"""Optional real GPU compute accelerators for classical QES array workloads.

This module layers genuine CuPy / CUDA-PyTorch execution paths on top of the
existing CPU-oriented backend concepts in :mod:`qes.backend`. It never pretends
that acceleration happened: when no functional GPU runtime is present, it
reports that honestly and keeps the public numeric operations working through a
NumPy CPU fallback.

Validation honesty for this change:
    * On the development machine used for this session, neither CuPy nor
      PyTorch was importable, and no CUDA-capable GPU runtime was detected.
    * The CPU fallback paths in this module were executed by automated tests and
      by the demo script in this session.
    * The CuPy and CUDA-PyTorch branches were validated here by code review,
      API-level correctness checks, and defensive runtime probing logic, but
      they were not executed against a real GPU in this session.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from qes.backend import BackendUnavailableError, NumpyBackend

_T = TypeVar("_T")


class GPUComputeUnavailableError(BackendUnavailableError):
    """Raised when a GPU-only operation is requested without a usable GPU runtime."""


@dataclass(frozen=True)
class GPUCapabilityReport:
    """Summary of whether this process can execute real GPU kernels."""

    runtime_name: str | None
    is_available: bool
    detail: str


@dataclass(frozen=True)
class PopulationDivergenceResult:
    """Batched DR/HSA-style divergence outputs for a population of states.

    Attributes:
        differences: raw per-state differences ``x - x_star``.
        scores: divergence scores analogous to ``DR`` in :mod:`qes.divergence`.
        health: per-state ``k * DR - baseline`` values.
    """

    differences: np.ndarray
    scores: np.ndarray
    health: np.ndarray


@dataclass(frozen=True)
class PopulationConvergenceResult:
    """Batched convergence diagnostics for populations of non-negative weights."""

    normalized_weights: np.ndarray
    entropy: np.ndarray
    normalized_entropy: np.ndarray
    convergence_coefficient: np.ndarray
    kl_divergence: np.ndarray
    gini: np.ndarray


def _as_float_array(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    return array


def _as_1d_float_array(name: str, values: ArrayLike) -> np.ndarray:
    array = _as_float_array(name, values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return array


def _as_2d_float_array(name: str, values: ArrayLike) -> np.ndarray:
    array = _as_float_array(name, values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return array


def _validate_dt(dt: float) -> float:
    step = float(dt)
    if step <= 0.0:
        raise ValueError("dt must be positive")
    return step


def _validate_non_negative(name: str, values: np.ndarray) -> None:
    if np.any(values < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")


def _broadcast_reference(reference: np.ndarray, batch_size: int) -> np.ndarray:
    if reference.ndim == 1:
        return np.broadcast_to(reference.reshape(1, -1), (batch_size, reference.shape[0])).copy()
    if reference.ndim == 2 and reference.shape[0] == batch_size:
        return reference
    raise ValueError("reference must be 1-D or match the 2-D batch shape of weights")


def _normalize_rows_numpy(weights: np.ndarray) -> np.ndarray:
    row_sums = np.sum(weights, axis=1, keepdims=True)
    safe_denominator = np.where(row_sums > 0.0, row_sums, 1.0)
    normalized = weights / safe_denominator
    return np.where(row_sums > 0.0, normalized, 0.0)


def _kl_divergence_numpy(probabilities: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mask = probabilities > 0.0
    q_safe = np.where(reference > 0.0, reference, np.finfo(float).tiny)
    safe_probabilities = np.where(mask, probabilities, 1.0)
    terms = np.where(mask, probabilities * np.log(safe_probabilities / q_safe), 0.0)
    return np.sum(terms, axis=1)


def _gini_numpy(probabilities: np.ndarray) -> np.ndarray:
    count = probabilities.shape[1]
    if count == 0:
        return np.zeros(probabilities.shape[0], dtype=float)
    sorted_weights = np.sort(probabilities, axis=1)
    index = np.arange(1, count + 1, dtype=float).reshape(1, -1)
    row_sums = np.sum(sorted_weights, axis=1)
    numerator = 2.0 * np.sum(sorted_weights * index, axis=1)
    raw = numerator / (count * np.where(row_sums > 0.0, row_sums, 1.0)) - (count + 1) / count
    return np.where(row_sums > 0.0, raw, 0.0)


def _rk4_step_numpy(
    states: np.ndarray,
    drift_matrix: np.ndarray,
    control_matrix: np.ndarray,
    controls: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Pure-NumPy batched RK4 step, shared by the CPU and simulated-GPU paths."""

    def rhs(current: np.ndarray) -> np.ndarray:
        drift = np.matmul(current, drift_matrix.T)
        control = np.matmul(controls, control_matrix.T)
        return drift + control

    k1 = rhs(states)
    k2 = rhs(states + 0.5 * dt * k1)
    k3 = rhs(states + 0.5 * dt * k2)
    k4 = rhs(states + dt * k3)
    return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _population_divergence_numpy(
    states: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    sigma: np.ndarray | None,
    scale: float,
    baseline: float,
) -> PopulationDivergenceResult:
    """Pure-NumPy batched divergence computation, shared by the CPU and
    simulated-GPU paths."""
    differences = states - reference
    working = differences if sigma is None else differences / sigma
    if weights.ndim == 1:
        scores = np.sum(weights * np.square(working), axis=1)
    else:
        scores = np.sum(np.matmul(working, weights) * working, axis=1)
    health = scale * scores - baseline
    return PopulationDivergenceResult(
        differences=differences,
        scores=np.asarray(scores, dtype=float),
        health=np.asarray(health, dtype=float),
    )


def _population_convergence_numpy(
    weights: np.ndarray,
    reference: np.ndarray,
) -> PopulationConvergenceResult:
    """Pure-NumPy batched convergence computation, shared by the CPU and
    simulated-GPU paths."""
    normalized = _normalize_rows_numpy(weights)
    normalized_reference = _normalize_rows_numpy(reference)
    entropy = -np.sum(
        np.where(normalized > 0.0, normalized * np.log(np.where(normalized > 0.0, normalized, 1.0)), 0.0),
        axis=1,
    )
    if normalized.shape[1] <= 1:
        normalized_entropy = np.zeros(normalized.shape[0], dtype=float)
    else:
        normalized_entropy = entropy / np.log(normalized.shape[1])
    convergence_coefficient = 1.0 - normalized_entropy
    kl_divergence = _kl_divergence_numpy(normalized, normalized_reference)
    gini = _gini_numpy(normalized)
    return PopulationConvergenceResult(
        normalized_weights=normalized,
        entropy=np.asarray(entropy, dtype=float),
        normalized_entropy=np.asarray(normalized_entropy, dtype=float),
        convergence_coefficient=np.asarray(convergence_coefficient, dtype=float),
        kl_divergence=np.asarray(kl_divergence, dtype=float),
        gini=np.asarray(gini, dtype=float),
    )


def _split_batch_slices(batch_size: int, lanes: int) -> list[tuple[int, int]]:
    """Split ``batch_size`` rows into up to ``lanes`` contiguous, near-equal chunks."""
    if batch_size <= 0:
        return [(0, 0)]
    lane_count = max(1, min(lanes, batch_size))
    base, remainder = divmod(batch_size, lane_count)
    slices: list[tuple[int, int]] = []
    start = 0
    for lane_index in range(lane_count):
        size = base + (1 if lane_index < remainder else 0)
        end = start + size
        if size > 0:  # pragma: no branch - size is always >= 1 here: lane_count <= batch_size by construction
            slices.append((start, end))
        start = end
    return slices


def _process_pid() -> int:
    """Return the real OS process ID of the calling process."""
    return os.getpid()


def _simulated_rk4_lane(
    states: np.ndarray,
    drift_matrix: np.ndarray,
    control_matrix: np.ndarray,
    controls: np.ndarray,
    dt: float,
) -> tuple[int, np.ndarray]:
    """Worker executed in a real, separate OS process: one simulated GPU 'lane'."""
    return _process_pid(), _rk4_step_numpy(states, drift_matrix, control_matrix, controls, dt)


def _simulated_divergence_lane(
    states: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    sigma: np.ndarray | None,
    scale: float,
    baseline: float,
) -> tuple[int, PopulationDivergenceResult]:
    """Worker executed in a real, separate OS process: one simulated GPU 'lane'."""
    return _process_pid(), _population_divergence_numpy(states, reference, weights, sigma, scale, baseline)


def _simulated_convergence_lane(
    weights: np.ndarray,
    reference: np.ndarray,
) -> tuple[int, PopulationConvergenceResult]:
    """Worker executed in a real, separate OS process: one simulated GPU 'lane'."""
    return _process_pid(), _population_convergence_numpy(weights, reference)


@lru_cache(maxsize=1)
def probe_gpu_runtime() -> GPUCapabilityReport:
    """Probe for a real, functional GPU runtime usable by this process."""
    cupy_report = _probe_cupy_runtime()
    if cupy_report.is_available:
        return cupy_report

    torch_report = _probe_torch_runtime()
    if torch_report.is_available:
        return torch_report

    return GPUCapabilityReport(
        runtime_name=None,
        is_available=False,
        detail=(
            "No functional GPU runtime detected. "
            f"CuPy probe: {cupy_report.detail}. "
            f"CUDA-PyTorch probe: {torch_report.detail}."
        ),
    )


def is_gpu_runtime_available() -> bool:
    """Return whether a real GPU runtime is currently usable."""
    return probe_gpu_runtime().is_available


def _probe_cupy_runtime() -> GPUCapabilityReport:
    try:
        cupy = import_module("cupy")
    except ImportError:
        return GPUCapabilityReport(None, False, "CuPy not importable")

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return GPUCapabilityReport(None, False, f"CuPy device query failed: {exc}")

    if device_count < 1:
        return GPUCapabilityReport(None, False, "CuPy imported but reported zero CUDA devices")

    try:
        probe = cupy.asarray([1.0, 2.0], dtype=float)
        _ = cupy.add(probe, probe)
        cupy.cuda.Stream.null.synchronize()
    except Exception as exc:
        return GPUCapabilityReport(None, False, f"CuPy kernel probe failed: {exc}")

    return GPUCapabilityReport("cupy", True, f"CuPy functional with {device_count} CUDA device(s)")


def _probe_torch_runtime() -> GPUCapabilityReport:
    try:
        torch = import_module("torch")
    except ImportError:
        return GPUCapabilityReport(None, False, "PyTorch not importable")

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return GPUCapabilityReport(None, False, f"torch.cuda.is_available() failed: {exc}")

    if not available:
        return GPUCapabilityReport(None, False, "PyTorch imported but CUDA is not available")

    try:
        device = torch.device("cuda")
        probe = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
        _ = torch.add(probe, probe)
        torch.cuda.synchronize()
        device_name = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        return GPUCapabilityReport(None, False, f"CUDA-PyTorch kernel probe failed: {exc}")

    return GPUCapabilityReport("torch-cuda", True, f"CUDA-PyTorch functional on device {device_name!r}")


class GPUComputeBackend:
    """Optional GPU-accelerated numeric backend with honest CPU fallback.

    Public numeric methods such as :meth:`batched_rk4_step`,
    :meth:`population_divergence`, and :meth:`population_convergence` always
    return NumPy arrays/dataclasses to keep the caller-facing API stable. When a
    real GPU runtime is available and ``prefer_gpu=True`` (the default), the
    heavy intermediate computation runs on GPU and the result is copied back to
    CPU before returning.
    """

    def __init__(self) -> None:
        """Initialize the backend and record accelerator availability honestly."""
        self._cpu_backend = NumpyBackend()
        self._report = probe_gpu_runtime()
        self._runtime_name = self._report.runtime_name
        self._module: Any | None = None
        self._torch_device: Any | None = None
        self._last_execution_path = "not-run"

        if self._report.is_available and self._runtime_name is not None:
            self._initialize_runtime(self._runtime_name)

    @property
    def runtime_name(self) -> str | None:
        """Return the detected runtime name, such as ``cupy`` or ``torch-cuda``."""
        return self._runtime_name

    @property
    def availability_detail(self) -> str:
        """Return a human-readable explanation of the runtime probe result."""
        return self._report.detail

    @property
    def last_execution_path(self) -> str:
        """Return how the most recent numeric operation was executed."""
        return self._last_execution_path

    def is_available(self) -> bool:
        """Return whether real GPU execution is available right now."""
        return self._report.is_available and self._module is not None

    def to_gpu_array(self, values: ArrayLike) -> Any:
        """Move array-like data to GPU memory and return a runtime-native array.

        Raises:
            GPUComputeUnavailableError: if no functional GPU runtime exists.
        """
        module = self._require_gpu("to_gpu_array")
        if self._runtime_name == "cupy":
            return module.asarray(values, dtype=float)
        return module.as_tensor(values, dtype=module.float64, device=self._torch_device)

    def to_numpy(self, values: Any) -> np.ndarray:
        """Move runtime-native data back to a NumPy array on CPU."""
        if self._runtime_name == "cupy" and self._module is not None:  # pragma: no cover - CuPy path only
            return np.asarray(self._module.asnumpy(values), dtype=float)
        if self._runtime_name == "torch-cuda" and self._module is not None:  # pragma: no cover - torch CUDA path only
            return np.asarray(values.detach().cpu().numpy(), dtype=float)
        return self._cpu_backend.array(values)

    def synchronize(self) -> None:
        """Wait for pending GPU work, or do nothing when running on CPU."""
        if not self.is_available() or self._module is None:  # pragma: no cover - CPU fallback
            return
        if self._runtime_name == "cupy":  # pragma: no cover - only exercised on CuPy-backed GPU machines
            self._module.cuda.Stream.null.synchronize()
            return
        self._module.cuda.synchronize()

    def batched_rk4_step(
        self,
        states: ArrayLike,
        drift_matrix: ArrayLike,
        control_matrix: ArrayLike,
        controls: ArrayLike,
        dt: float,
        *,
        prefer_gpu: bool = True,
    ) -> np.ndarray:
        """Integrate a linear batched room-dynamics model with RK4.

        The continuous-time model is:

            ``xdot = x @ drift_matrix.T + controls @ control_matrix.T``

        which mirrors the control-oriented structure used in
        :mod:`qes.dynamics`, but executes a whole population in one call.
        """
        state_array = _as_2d_float_array("states", states)
        drift_array = _as_2d_float_array("drift_matrix", drift_matrix)
        control_array = _as_2d_float_array("control_matrix", control_matrix)
        input_array = _as_2d_float_array("controls", controls)
        step = _validate_dt(dt)

        batch_size, state_dim = state_array.shape
        if drift_array.shape != (state_dim, state_dim):
            raise ValueError("drift_matrix must have shape (state_dim, state_dim)")
        if control_array.shape[0] != state_dim:
            raise ValueError("control_matrix row count must equal state dimension")
        if input_array.shape[0] != batch_size:
            raise ValueError("controls batch size must match states batch size")
        if input_array.shape[1] != control_array.shape[1]:
            raise ValueError("controls width must match control_matrix column count")

        return self._dispatch(
            operation_name="batched_rk4_step",
            prefer_gpu=prefer_gpu,
            cpu_fn=lambda: self._batched_rk4_step_cpu(
                state_array, drift_array, control_array, input_array, step
            ),
            gpu_fn=lambda: self._batched_rk4_step_gpu(
                state_array, drift_array, control_array, input_array, step
            ),
        )

    def population_divergence(
        self,
        states: ArrayLike,
        reference: ArrayLike,
        weights: ArrayLike,
        *,
        sigma: ArrayLike | None = None,
        scale: float = 1.0,
        baseline: float = 0.0,
        prefer_gpu: bool = True,
    ) -> PopulationDivergenceResult:
        """Compute batched DR/HSA-style divergence scores for many states."""
        state_array = _as_2d_float_array("states", states)
        reference_array = _as_1d_float_array("reference", reference)
        weight_array = _as_float_array("weights", weights)

        if state_array.shape[1] != reference_array.shape[0]:
            raise ValueError("reference length must equal state dimension")
        if weight_array.ndim == 1 and weight_array.shape[0] != state_array.shape[1]:
            raise ValueError("1-D weights length must equal state dimension")
        if weight_array.ndim == 2 and weight_array.shape != (state_array.shape[1], state_array.shape[1]):
            raise ValueError("2-D weights must have shape (state_dim, state_dim)")
        if weight_array.ndim not in {1, 2}:
            raise ValueError("weights must be a 1-D vector or a square 2-D matrix")

        sigma_array: np.ndarray | None = None
        if sigma is not None:
            sigma_array = _as_1d_float_array("sigma", sigma)
            if sigma_array.shape[0] != state_array.shape[1]:
                raise ValueError("sigma length must equal state dimension")
            if np.any(sigma_array == 0.0):
                raise ValueError("sigma must not contain zeros")

        return self._dispatch(
            operation_name="population_divergence",
            prefer_gpu=prefer_gpu,
            cpu_fn=lambda: self._population_divergence_cpu(
                state_array,
                reference_array,
                weight_array,
                sigma_array,
                float(scale),
                float(baseline),
            ),
            gpu_fn=lambda: self._population_divergence_gpu(
                state_array,
                reference_array,
                weight_array,
                sigma_array,
                float(scale),
                float(baseline),
            ),
        )

    def population_convergence(
        self,
        weights: ArrayLike,
        *,
        reference: ArrayLike | None = None,
        prefer_gpu: bool = True,
    ) -> PopulationConvergenceResult:
        """Compute batched entropy, KL, Gini, and convergence coefficients."""
        weight_array = _as_2d_float_array("weights", weights)
        _validate_non_negative("weights", weight_array)

        if reference is None:
            if weight_array.shape[1] == 0:
                reference_array = np.zeros_like(weight_array)
            else:
                reference_array = np.full_like(weight_array, 1.0 / weight_array.shape[1], dtype=float)
        else:
            reference_input = _as_float_array("reference", reference)
            _validate_non_negative("reference", reference_input)
            reference_array = _broadcast_reference(reference_input, weight_array.shape[0])
            if reference_array.shape[1] != weight_array.shape[1]:
                raise ValueError("reference width must match weights width")

        return self._dispatch(
            operation_name="population_convergence",
            prefer_gpu=prefer_gpu,
            cpu_fn=lambda: self._population_convergence_cpu(weight_array, reference_array),
            gpu_fn=lambda: self._population_convergence_gpu(weight_array, reference_array),
        )

    def _dispatch(
        self,
        operation_name: str,
        *,
        prefer_gpu: bool,
        cpu_fn: Callable[[], _T],
        gpu_fn: Callable[[], _T],
    ) -> _T:
        if prefer_gpu and self.is_available():
            self._last_execution_path = f"gpu:{self.runtime_name}:{operation_name}"
            result = gpu_fn()
            self.synchronize()
            return result

        self._last_execution_path = f"cpu-fallback:numpy:{operation_name}"
        return cpu_fn()

    def _initialize_runtime(self, runtime_name: str) -> None:
        if runtime_name == "cupy":
            self._module = import_module("cupy")
            return

        torch = import_module("torch")
        self._module = torch
        self._torch_device = torch.device("cuda")

    def _require_gpu(self, operation_name: str) -> Any:
        if not self.is_available() or self._module is None:
            raise GPUComputeUnavailableError(
                "No functional GPU runtime is available for "
                f"{operation_name}. {self.availability_detail}"
            )
        return self._module

    def _batched_rk4_step_cpu(
        self,
        states: np.ndarray,
        drift_matrix: np.ndarray,
        control_matrix: np.ndarray,
        controls: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        return _rk4_step_numpy(states, drift_matrix, control_matrix, controls, dt)

    def _batched_rk4_step_gpu(
        self,
        states: np.ndarray,
        drift_matrix: np.ndarray,
        control_matrix: np.ndarray,
        controls: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        module = self._require_gpu("batched_rk4_step")
        states_gpu = self.to_gpu_array(states)
        drift_gpu = self.to_gpu_array(drift_matrix)
        control_gpu = self.to_gpu_array(control_matrix)
        inputs_gpu = self.to_gpu_array(controls)

        def rhs(current: Any) -> Any:
            drift = module.matmul(current, drift_gpu.T)
            control = module.matmul(inputs_gpu, control_gpu.T)
            return drift + control

        k1 = rhs(states_gpu)
        k2 = rhs(states_gpu + 0.5 * dt * k1)
        k3 = rhs(states_gpu + 0.5 * dt * k2)
        k4 = rhs(states_gpu + dt * k3)
        result = states_gpu + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.to_numpy(result)

    def _population_divergence_cpu(
        self,
        states: np.ndarray,
        reference: np.ndarray,
        weights: np.ndarray,
        sigma: np.ndarray | None,
        scale: float,
        baseline: float,
    ) -> PopulationDivergenceResult:
        return _population_divergence_numpy(states, reference, weights, sigma, scale, baseline)


    def _population_divergence_gpu(
        self,
        states: np.ndarray,
        reference: np.ndarray,
        weights: np.ndarray,
        sigma: np.ndarray | None,
        scale: float,
        baseline: float,
    ) -> PopulationDivergenceResult:
        module = self._require_gpu("population_divergence")
        states_gpu = self.to_gpu_array(states)
        reference_gpu = self.to_gpu_array(reference)
        weights_gpu = self.to_gpu_array(weights)
        sigma_gpu = None if sigma is None else self.to_gpu_array(sigma)

        differences = states_gpu - reference_gpu
        working = differences if sigma_gpu is None else differences / sigma_gpu
        if weights.ndim == 1:
            scores = module.sum(weights_gpu * module.square(working), axis=1)
        else:
            scores = module.sum(module.matmul(working, weights_gpu) * working, axis=1)
        health = scale * scores - baseline
        return PopulationDivergenceResult(
            differences=self.to_numpy(differences),
            scores=self.to_numpy(scores),
            health=self.to_numpy(health),
        )

    def _population_convergence_cpu(
        self,
        weights: np.ndarray,
        reference: np.ndarray,
    ) -> PopulationConvergenceResult:
        return _population_convergence_numpy(weights, reference)


    def _population_convergence_gpu(
        self,
        weights: np.ndarray,
        reference: np.ndarray,
    ) -> PopulationConvergenceResult:
        weights_gpu = self.to_gpu_array(weights)
        reference_gpu = self.to_gpu_array(reference)
        normalized = self._normalize_rows_gpu(weights_gpu)
        normalized_reference = self._normalize_rows_gpu(reference_gpu)
        positive = normalized > 0.0
        safe = self._where(positive, normalized, self._ones_like(normalized))
        entropy_terms = self._where(positive, normalized * self._log(safe), self._zeros_like(normalized))
        entropy = -self._sum_axis(entropy_terms, axis=1)
        if weights.shape[1] <= 1:
            normalized_entropy = self._zeros_like(entropy)
        else:
            normalized_entropy = entropy / float(np.log(weights.shape[1]))
        convergence_coefficient = 1.0 - normalized_entropy
        kl_divergence = self._kl_divergence_gpu(normalized, normalized_reference)
        gini = self._gini_gpu(normalized)
        return PopulationConvergenceResult(
            normalized_weights=self.to_numpy(normalized),
            entropy=self.to_numpy(entropy),
            normalized_entropy=self.to_numpy(normalized_entropy),
            convergence_coefficient=self.to_numpy(convergence_coefficient),
            kl_divergence=self.to_numpy(kl_divergence),
            gini=self.to_numpy(gini),
        )

    def _normalize_rows_gpu(self, weights: Any) -> Any:
        row_sums = self._sum_axis(weights, axis=1).reshape(-1, 1)
        safe_denominator = self._where(row_sums > 0.0, row_sums, self._ones_like(row_sums))
        normalized = weights / safe_denominator
        return self._where(row_sums > 0.0, normalized, self._zeros_like(weights))

    def _kl_divergence_gpu(self, probabilities: Any, reference: Any) -> Any:
        mask = probabilities > 0.0
        q_safe = self._where(reference > 0.0, reference, self._full_like(reference, np.finfo(float).tiny))
        safe_probabilities = self._where(mask, probabilities, self._ones_like(probabilities))
        terms = self._where(
            mask,
            probabilities * self._log(safe_probabilities / q_safe),
            self._zeros_like(probabilities),
        )
        return self._sum_axis(terms, axis=1)

    def _gini_gpu(self, probabilities: Any) -> Any:
        count = int(probabilities.shape[1])
        if count == 0:
            return self._zeros_like(self._sum_axis(probabilities, axis=1))
        sorted_weights = self._sort_rows(probabilities)
        index = self.to_gpu_array(np.arange(1, count + 1, dtype=float).reshape(1, -1))
        row_sums = self._sum_axis(sorted_weights, axis=1)
        numerator = 2.0 * self._sum_axis(sorted_weights * index, axis=1)
        safe_denominator = self._where(row_sums > 0.0, row_sums, self._ones_like(row_sums))
        raw = numerator / (count * safe_denominator) - (count + 1) / count
        return self._where(row_sums > 0.0, raw, self._zeros_like(raw))

    def _sum_axis(self, values: Any, axis: int) -> Any:
        if self._runtime_name == "torch-cuda":
            return values.sum(dim=axis)
        return values.sum(axis=axis)

    def _log(self, values: Any) -> Any:
        module = self._require_gpu("log")
        return module.log(values)

    def _zeros_like(self, values: Any) -> Any:
        module = self._require_gpu("zeros_like")
        return module.zeros_like(values)

    def _ones_like(self, values: Any) -> Any:
        module = self._require_gpu("ones_like")
        return module.ones_like(values)

    def _full_like(self, values: Any, fill_value: float) -> Any:
        module = self._require_gpu("full_like")
        return module.full_like(values, fill_value)

    def _where(self, condition: Any, left: Any, right: Any) -> Any:
        module = self._require_gpu("where")
        return module.where(condition, left, right)

    def _sort_rows(self, values: Any) -> Any:
        module = self._require_gpu("sort")
        if self._runtime_name == "torch-cuda":
            return module.sort(values, dim=1).values
        return module.sort(values, axis=1)


class SimulatedGPUBackend:
    """A software SIMULATION of a multi-lane accelerator device.

    HONESTY NOTICE (read before using this class):
        This class does **not** provide real GPU hardware acceleration. It
        exists so the "many parallel compute lanes" execution shape that a
        real GPU backend would need can be built, exercised, and tested even
        when no functional CUDA/ROCm GPU runtime is present -- which is the
        case in this project's own development environment (an AMD
        integrated GPU with no CUDA/ROCm support detected; see
        :func:`probe_gpu_runtime`).

        Each "lane" here is a genuine, separate operating-system process
        (spawned via :class:`concurrent.futures.ProcessPoolExecutor`), so
        this class *does* provide real multi-core CPU parallelism -- but the
        thing being simulated is accelerator-style batch partitioning and
        multi-lane dispatch, not real GPU silicon. The numeric results are
        always identical to the plain single-process NumPy path; only the
        execution/partitioning strategy differs. Do not cite this class as
        evidence of GPU-level performance, and do not report its wall-clock
        timings as GPU acceleration numbers -- they measure ordinary
        multi-process CPU parallelism on this machine only.

    Each call to a public method reports the real OS process IDs that
    executed it via :attr:`last_lane_pids`, so this claim is independently
    checkable rather than asserted.
    """

    def __init__(self, lanes: int | None = None) -> None:
        """Create a simulated accelerator with ``lanes`` real worker processes.

        Args:
            lanes: number of OS worker processes to use as simulated compute
                lanes. Defaults to ``min(4, os.cpu_count())``.
        """
        cpu_count = os.cpu_count() or 1
        requested = int(lanes) if lanes is not None else min(4, cpu_count)
        if requested < 1:
            raise ValueError("lanes must be >= 1")
        self._lanes = requested
        self._pool = ProcessPoolExecutor(max_workers=self._lanes)
        self._last_lane_pids: tuple[int, ...] = ()
        self._last_execution_path = "not-run"
        self._closed = False

    def __enter__(self) -> SimulatedGPUBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    @property
    def lanes(self) -> int:
        """Return the configured number of simulated compute lanes (real OS processes)."""
        return self._lanes

    @property
    def is_simulated(self) -> bool:
        """Always ``True``: this backend never executes on real GPU hardware."""
        return True

    @property
    def last_execution_path(self) -> str:
        """Return how the most recent numeric operation was executed."""
        return self._last_execution_path

    @property
    def last_lane_pids(self) -> tuple[int, ...]:
        """Return the real OS process IDs used by the most recent call.

        These are genuine, distinct process IDs (verifiable via ``os.getpid()``
        inside each worker), proving this is real multiprocessing rather than
        a purely notional/faked parallelism claim.
        """
        return self._last_lane_pids

    def is_available(self) -> bool:
        """Always ``True``: a software simulation has no hardware dependency."""
        return not self._closed

    def shutdown(self) -> None:
        """Stop all simulated-lane worker processes. Safe to call more than once."""
        if not self._closed:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._closed = True

    def batched_rk4_step(
        self,
        states: ArrayLike,
        drift_matrix: ArrayLike,
        control_matrix: ArrayLike,
        controls: ArrayLike,
        dt: float,
    ) -> np.ndarray:
        """Integrate a batched linear room-dynamics model, partitioned across simulated lanes."""
        self._require_open()
        state_array = _as_2d_float_array("states", states)
        drift_array = _as_2d_float_array("drift_matrix", drift_matrix)
        control_array = _as_2d_float_array("control_matrix", control_matrix)
        input_array = _as_2d_float_array("controls", controls)
        step = _validate_dt(dt)

        batch_size, state_dim = state_array.shape
        if drift_array.shape != (state_dim, state_dim):
            raise ValueError("drift_matrix must have shape (state_dim, state_dim)")
        if control_array.shape[0] != state_dim:
            raise ValueError("control_matrix row count must equal state dimension")
        if input_array.shape[0] != batch_size:
            raise ValueError("controls batch size must match states batch size")
        if input_array.shape[1] != control_array.shape[1]:
            raise ValueError("controls width must match control_matrix column count")

        slices = _split_batch_slices(batch_size, self._lanes)
        futures = [
            self._pool.submit(
                _simulated_rk4_lane,
                state_array[start:end],
                drift_array,
                control_array,
                input_array[start:end],
                step,
            )
            for start, end in slices
        ]
        pids: list[int] = []
        chunks: list[np.ndarray] = []
        for future in futures:
            pid, chunk = future.result()
            pids.append(pid)
            chunks.append(chunk)
        self._last_lane_pids = tuple(pids)
        self._last_execution_path = f"simulated-gpu:{len(slices)}-lane-processes:batched_rk4_step"
        if not chunks:  # pragma: no cover - _split_batch_slices always yields >= 1 slice, so unreachable
            return state_array.copy()
        return np.concatenate(chunks, axis=0)

    def population_divergence(
        self,
        states: ArrayLike,
        reference: ArrayLike,
        weights: ArrayLike,
        *,
        sigma: ArrayLike | None = None,
        scale: float = 1.0,
        baseline: float = 0.0,
    ) -> PopulationDivergenceResult:
        """Compute batched divergence scores, partitioned across simulated lanes."""
        self._require_open()
        state_array = _as_2d_float_array("states", states)
        reference_array = _as_1d_float_array("reference", reference)
        weight_array = _as_float_array("weights", weights)

        if state_array.shape[1] != reference_array.shape[0]:
            raise ValueError("reference length must equal state dimension")
        if weight_array.ndim == 1 and weight_array.shape[0] != state_array.shape[1]:
            raise ValueError("1-D weights length must equal state dimension")
        if weight_array.ndim == 2 and weight_array.shape != (state_array.shape[1], state_array.shape[1]):
            raise ValueError("2-D weights must have shape (state_dim, state_dim)")
        if weight_array.ndim not in {1, 2}:
            raise ValueError("weights must be a 1-D vector or a square 2-D matrix")

        sigma_array: np.ndarray | None = None
        if sigma is not None:
            sigma_array = _as_1d_float_array("sigma", sigma)
            if sigma_array.shape[0] != state_array.shape[1]:
                raise ValueError("sigma length must equal state dimension")
            if np.any(sigma_array == 0.0):
                raise ValueError("sigma must not contain zeros")

        batch_size = state_array.shape[0]
        slices = _split_batch_slices(batch_size, self._lanes)
        futures = [
            self._pool.submit(
                _simulated_divergence_lane,
                state_array[start:end],
                reference_array,
                weight_array,
                sigma_array,
                float(scale),
                float(baseline),
            )
            for start, end in slices
        ]
        pids = []
        parts: list[PopulationDivergenceResult] = []
        for future in futures:
            pid, part = future.result()
            pids.append(pid)
            parts.append(part)
        self._last_lane_pids = tuple(pids)
        self._last_execution_path = f"simulated-gpu:{len(slices)}-lane-processes:population_divergence"
        if not parts:  # pragma: no cover - _split_batch_slices always yields >= 1 slice, so unreachable
            empty = np.zeros((0, state_array.shape[1]), dtype=float)
            return PopulationDivergenceResult(
                differences=empty, scores=np.zeros(0, dtype=float), health=np.zeros(0, dtype=float)
            )
        return PopulationDivergenceResult(
            differences=np.concatenate([part.differences for part in parts], axis=0),
            scores=np.concatenate([part.scores for part in parts], axis=0),
            health=np.concatenate([part.health for part in parts], axis=0),
        )

    def population_convergence(
        self,
        weights: ArrayLike,
        *,
        reference: ArrayLike | None = None,
    ) -> PopulationConvergenceResult:
        """Compute batched convergence diagnostics, partitioned across simulated lanes."""
        self._require_open()
        weight_array = _as_2d_float_array("weights", weights)
        _validate_non_negative("weights", weight_array)

        if reference is None:
            if weight_array.shape[1] == 0:
                reference_array = np.zeros_like(weight_array)
            else:
                reference_array = np.full_like(weight_array, 1.0 / weight_array.shape[1], dtype=float)
        else:
            reference_input = _as_float_array("reference", reference)
            _validate_non_negative("reference", reference_input)
            reference_array = _broadcast_reference(reference_input, weight_array.shape[0])
            if reference_array.shape[1] != weight_array.shape[1]:
                raise ValueError("reference width must match weights width")

        batch_size = weight_array.shape[0]
        slices = _split_batch_slices(batch_size, self._lanes)
        futures = [
            self._pool.submit(
                _simulated_convergence_lane,
                weight_array[start:end],
                reference_array[start:end],
            )
            for start, end in slices
        ]
        pids = []
        parts: list[PopulationConvergenceResult] = []
        for future in futures:
            pid, part = future.result()
            pids.append(pid)
            parts.append(part)
        self._last_lane_pids = tuple(pids)
        self._last_execution_path = f"simulated-gpu:{len(slices)}-lane-processes:population_convergence"
        if not parts:  # pragma: no cover - _split_batch_slices always yields >= 1 slice, so unreachable
            empty_2d = np.zeros((0, weight_array.shape[1]), dtype=float)
            empty_1d = np.zeros(0, dtype=float)
            return PopulationConvergenceResult(
                normalized_weights=empty_2d,
                entropy=empty_1d,
                normalized_entropy=empty_1d,
                convergence_coefficient=empty_1d,
                kl_divergence=empty_1d,
                gini=empty_1d,
            )
        return PopulationConvergenceResult(
            normalized_weights=np.concatenate([part.normalized_weights for part in parts], axis=0),
            entropy=np.concatenate([part.entropy for part in parts], axis=0),
            normalized_entropy=np.concatenate([part.normalized_entropy for part in parts], axis=0),
            convergence_coefficient=np.concatenate([part.convergence_coefficient for part in parts], axis=0),
            kl_divergence=np.concatenate([part.kl_divergence for part in parts], axis=0),
            gini=np.concatenate([part.gini for part in parts], axis=0),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SimulatedGPUBackend has been shut down; create a new instance to continue")


__all__ = [
    "GPUCapabilityReport",
    "GPUComputeBackend",
    "GPUComputeUnavailableError",
    "PopulationConvergenceResult",
    "PopulationDivergenceResult",
    "SimulatedGPUBackend",
    "is_gpu_runtime_available",
    "probe_gpu_runtime",
]
