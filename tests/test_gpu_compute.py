from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import qes.gpu_compute as gpu_compute
from qes.gpu_compute import (
    GPUCapabilityReport,
    GPUComputeBackend,
    GPUComputeUnavailableError,
    SimulatedGPUBackend,
)

GPU_AVAILABLE = gpu_compute.is_gpu_runtime_available()


def _manual_rk4(
    states: np.ndarray,
    drift_matrix: np.ndarray,
    control_matrix: np.ndarray,
    controls: np.ndarray,
    dt: float,
) -> np.ndarray:
    def rhs(current: np.ndarray) -> np.ndarray:
        return current @ drift_matrix.T + controls @ control_matrix.T

    k1 = rhs(states)
    k2 = rhs(states + 0.5 * dt * k1)
    k3 = rhs(states + 0.5 * dt * k2)
    k4 = rhs(states + dt * k3)
    return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _manual_divergence(
    states: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
    sigma: np.ndarray | None,
    scale: float,
    baseline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences = states - reference
    working = differences if sigma is None else differences / sigma
    if weights.ndim == 1:
        scores = np.sum(weights * working**2, axis=1)
    else:
        scores = np.sum((working @ weights) * working, axis=1)
    health = scale * scores - baseline
    return differences, scores, health


def _manual_convergence(
    weights: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_sums = np.sum(weights, axis=1, keepdims=True)
    normalized = np.where(row_sums > 0.0, weights / row_sums, 0.0)
    ref_sums = np.sum(reference, axis=1, keepdims=True)
    normalized_reference = np.where(ref_sums > 0.0, reference / ref_sums, 0.0)

    entropy = -np.sum(
        np.where(normalized > 0.0, normalized * np.log(np.where(normalized > 0.0, normalized, 1.0)), 0.0),
        axis=1,
    )
    normalized_entropy = entropy / np.log(weights.shape[1])
    convergence_coefficient = 1.0 - normalized_entropy

    q_safe = np.where(normalized_reference > 0.0, normalized_reference, np.finfo(float).tiny)
    safe_normalized = np.where(normalized > 0.0, normalized, 1.0)
    kl_divergence = np.sum(
        np.where(normalized > 0.0, normalized * np.log(safe_normalized / q_safe), 0.0),
        axis=1,
    )

    sorted_weights = np.sort(normalized, axis=1)
    index = np.arange(1, weights.shape[1] + 1, dtype=float).reshape(1, -1)
    gini = (2.0 * np.sum(sorted_weights * index, axis=1)) / weights.shape[1] - (
        weights.shape[1] + 1
    ) / weights.shape[1]
    return (
        normalized,
        entropy,
        normalized_entropy,
        convergence_coefficient,
        kl_divergence,
        gini,
    )


def test_backend_honestly_reports_runtime_probe_result() -> None:
    backend = GPUComputeBackend()
    report = gpu_compute.probe_gpu_runtime()
    assert backend.is_available() is report.is_available
    assert backend.runtime_name == report.runtime_name
    assert backend.availability_detail == report.detail


def test_gpu_only_method_raises_clear_error_when_runtime_is_forced_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_compute,
        "probe_gpu_runtime",
        lambda: GPUCapabilityReport(None, False, "forced test-only unavailability"),
    )
    backend = GPUComputeBackend()
    assert not backend.is_available()
    with pytest.raises(GPUComputeUnavailableError, match="forced test-only unavailability"):
        backend.to_gpu_array([1.0, 2.0, 3.0])


@pytest.mark.skipif(GPU_AVAILABLE, reason="This check is only meaningful on CPU-only environments")
def test_gpu_only_method_honestly_raises_on_actual_cpu_only_environment() -> None:
    backend = GPUComputeBackend()
    assert backend.is_available() is False
    with pytest.raises(GPUComputeUnavailableError, match="No functional GPU runtime is available"):
        backend.to_gpu_array([1.0, 2.0, 3.0])


def test_batched_rk4_step_cpu_fallback_matches_manual_result() -> None:
    backend = GPUComputeBackend()
    states = np.array([[0.4, -0.1], [0.2, 0.3]], dtype=float)
    drift_matrix = np.array([[0.0, 1.0], [-0.5, -0.2]], dtype=float)
    control_matrix = np.array([[1.0], [0.25]], dtype=float)
    controls = np.array([[0.2], [-0.1]], dtype=float)
    dt = 0.05

    result = backend.batched_rk4_step(
        states,
        drift_matrix,
        control_matrix,
        controls,
        dt,
        prefer_gpu=False,
    )

    np.testing.assert_allclose(result, _manual_rk4(states, drift_matrix, control_matrix, controls, dt))
    assert backend.last_execution_path == "cpu-fallback:numpy:batched_rk4_step"


def test_population_divergence_cpu_fallback_matches_manual_result() -> None:
    backend = GPUComputeBackend()
    states = np.array([[1.0, 2.0, 1.5], [0.5, 2.5, 0.5]], dtype=float)
    reference = np.array([0.75, 1.5, 1.0], dtype=float)
    weights = np.array([1.0, 0.5, 2.0], dtype=float)
    sigma = np.array([0.5, 1.0, 0.25], dtype=float)

    result = backend.population_divergence(
        states,
        reference,
        weights,
        sigma=sigma,
        scale=1.5,
        baseline=0.25,
        prefer_gpu=False,
    )
    differences, scores, health = _manual_divergence(
        states,
        reference,
        weights,
        sigma,
        scale=1.5,
        baseline=0.25,
    )

    np.testing.assert_allclose(result.differences, differences)
    np.testing.assert_allclose(result.scores, scores)
    np.testing.assert_allclose(result.health, health)
    assert backend.last_execution_path == "cpu-fallback:numpy:population_divergence"


def test_population_convergence_cpu_fallback_matches_manual_result() -> None:
    backend = GPUComputeBackend()
    weights = np.array([[0.2, 0.3, 0.5], [4.0, 1.0, 0.0]], dtype=float)
    reference = np.array([[1.0, 1.0, 1.0], [0.5, 0.25, 0.25]], dtype=float)

    result = backend.population_convergence(weights, reference=reference, prefer_gpu=False)
    (
        normalized,
        entropy,
        normalized_entropy,
        convergence_coefficient,
        kl_divergence,
        gini,
    ) = _manual_convergence(weights, reference)

    np.testing.assert_allclose(result.normalized_weights, normalized)
    np.testing.assert_allclose(result.entropy, entropy)
    np.testing.assert_allclose(result.normalized_entropy, normalized_entropy)
    np.testing.assert_allclose(result.convergence_coefficient, convergence_coefficient)
    np.testing.assert_allclose(result.kl_divergence, kl_divergence)
    np.testing.assert_allclose(result.gini, gini)
    assert backend.last_execution_path == "cpu-fallback:numpy:population_convergence"


def test_population_convergence_rejects_negative_weights() -> None:
    backend = GPUComputeBackend()
    with pytest.raises(ValueError, match="weights must contain only non-negative values"):
        backend.population_convergence([[1.0, -0.1, 0.2]], prefer_gpu=False)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="No functional GPU runtime available in this environment")
def test_gpu_execution_path_runs_and_matches_cpu_reference() -> None:
    backend = GPUComputeBackend()
    states = np.array([[0.4, -0.1], [0.2, 0.3]], dtype=float)
    drift_matrix = np.array([[0.0, 1.0], [-0.5, -0.2]], dtype=float)
    control_matrix = np.array([[1.0], [0.25]], dtype=float)
    controls = np.array([[0.2], [-0.1]], dtype=float)

    rk4_result = backend.batched_rk4_step(states, drift_matrix, control_matrix, controls, 0.05)
    np.testing.assert_allclose(rk4_result, _manual_rk4(states, drift_matrix, control_matrix, controls, 0.05))
    assert backend.last_execution_path.startswith("gpu:")

    divergence = backend.population_divergence(states, np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    _, expected_scores, _ = _manual_divergence(
        states,
        np.array([0.0, 0.0]),
        np.array([1.0, 2.0]),
        None,
        1.0,
        0.0,
    )
    np.testing.assert_allclose(divergence.scores, expected_scores)
    assert backend.last_execution_path.startswith("gpu:")

    convergence = backend.population_convergence(np.abs(states) + 0.5)
    assert convergence.normalized_weights.shape == states.shape
    assert backend.last_execution_path.startswith("gpu:")


# ---------------------------------------------------------------------------
# SimulatedGPUBackend: a real multi-process software SIMULATION of GPU-lane
# execution. It never provides real GPU hardware acceleration -- see its
# class docstring in qes.gpu_compute for the full honesty notice. These tests
# verify (a) numeric correctness against the same manual reference math used
# for GPUComputeBackend's CPU fallback, and (b) that execution genuinely
# spans multiple distinct real OS processes, so the "simulation" claim is
# independently checkable rather than merely asserted.
# ---------------------------------------------------------------------------


def _large_rk4_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(1234)
    states = rng.random((400, 12))
    drift_matrix = rng.random((12, 12)) * 0.01
    control_matrix = rng.random((12, 4)) * 0.01
    controls = rng.random((400, 4))
    return states, drift_matrix, control_matrix, controls, 0.01


def test_simulated_backend_is_always_available_and_labeled_as_simulated() -> None:
    with SimulatedGPUBackend(lanes=2) as backend:
        assert backend.is_available() is True
        assert backend.is_simulated is True
        assert backend.lanes == 2
        assert backend.last_execution_path == "not-run"


def test_simulated_backend_rk4_matches_manual_reference() -> None:
    states, drift_matrix, control_matrix, controls, dt = _large_rk4_inputs()
    with SimulatedGPUBackend(lanes=4) as backend:
        result = backend.batched_rk4_step(states, drift_matrix, control_matrix, controls, dt)
        np.testing.assert_allclose(result, _manual_rk4(states, drift_matrix, control_matrix, controls, dt))
        assert backend.last_execution_path.startswith("simulated-gpu:")
        assert len(backend.last_lane_pids) >= 1


def test_simulated_backend_uses_multiple_distinct_real_processes() -> None:
    states, drift_matrix, control_matrix, controls, dt = _large_rk4_inputs()
    with SimulatedGPUBackend(lanes=4) as backend:
        backend.batched_rk4_step(states, drift_matrix, control_matrix, controls, dt)
        # A large-enough batch with 4 lanes should genuinely fan out across
        # more than one real OS process (proving actual multiprocessing).
        assert len(set(backend.last_lane_pids)) > 1


def test_simulated_backend_divergence_matches_manual_reference() -> None:
    states = np.array([[1.0, 2.0, 1.5], [0.5, 2.5, 0.5], [2.0, 0.5, 1.0], [0.1, 0.2, 0.3]], dtype=float)
    reference = np.array([0.75, 1.5, 1.0], dtype=float)
    weights = np.array([1.0, 0.5, 2.0], dtype=float)
    sigma = np.array([0.5, 1.0, 0.25], dtype=float)

    with SimulatedGPUBackend(lanes=3) as backend:
        result = backend.population_divergence(
            states,
            reference,
            weights,
            sigma=sigma,
            scale=1.5,
            baseline=0.25,
        )
        differences, scores, health = _manual_divergence(
            states,
            reference,
            weights,
            sigma,
            scale=1.5,
            baseline=0.25,
        )
        np.testing.assert_allclose(result.differences, differences)
        np.testing.assert_allclose(result.scores, scores)
        np.testing.assert_allclose(result.health, health)
        assert backend.last_execution_path == "simulated-gpu:3-lane-processes:population_divergence"


def test_simulated_backend_convergence_matches_manual_reference() -> None:
    weights = np.array([[0.2, 0.3, 0.5], [4.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    reference = np.array([[1.0, 1.0, 1.0], [0.5, 0.25, 0.25], [2.0, 2.0, 2.0]], dtype=float)

    with SimulatedGPUBackend(lanes=2) as backend:
        result = backend.population_convergence(weights, reference=reference)
        (
            normalized,
            entropy,
            normalized_entropy,
            convergence_coefficient,
            kl_divergence,
            gini,
        ) = _manual_convergence(weights, reference)

        np.testing.assert_allclose(result.normalized_weights, normalized)
        np.testing.assert_allclose(result.entropy, entropy)
        np.testing.assert_allclose(result.normalized_entropy, normalized_entropy)
        np.testing.assert_allclose(result.convergence_coefficient, convergence_coefficient)
        np.testing.assert_allclose(result.kl_divergence, kl_divergence)
        np.testing.assert_allclose(result.gini, gini)


def test_simulated_backend_handles_batch_smaller_than_lane_count() -> None:
    states = np.array([[0.4, -0.1], [0.2, 0.3]], dtype=float)
    drift_matrix = np.array([[0.0, 1.0], [-0.5, -0.2]], dtype=float)
    control_matrix = np.array([[1.0], [0.25]], dtype=float)
    controls = np.array([[0.2], [-0.1]], dtype=float)

    with SimulatedGPUBackend(lanes=8) as backend:
        result = backend.batched_rk4_step(states, drift_matrix, control_matrix, controls, 0.05)
        np.testing.assert_allclose(result, _manual_rk4(states, drift_matrix, control_matrix, controls, 0.05))
        # Fewer rows than lanes: no empty lanes should be scheduled.
        assert len(backend.last_lane_pids) <= states.shape[0]


def test_simulated_backend_rejects_non_positive_lanes() -> None:
    with pytest.raises(ValueError, match="lanes must be >= 1"):
        SimulatedGPUBackend(lanes=0)


def test_simulated_backend_shutdown_blocks_further_use() -> None:
    backend = SimulatedGPUBackend(lanes=2)
    backend.shutdown()
    assert backend.is_available() is False
    with pytest.raises(RuntimeError, match="has been shut down"):
        backend.population_convergence([[1.0, 2.0, 3.0]])
    # Calling shutdown() again must be safe (idempotent).
    backend.shutdown()


def test_simulated_backend_is_honestly_documented_as_not_real_gpu_hardware() -> None:
    docstring = (SimulatedGPUBackend.__doc__ or "").lower()
    assert "does **not** provide real gpu hardware acceleration" in docstring
    assert "simulation" in docstring


# --------------------------------------------------------------------------
# Free-function validation-error and edge-case coverage.
# --------------------------------------------------------------------------


def test_as_float_array_rejects_scalar_input() -> None:
    with pytest.raises(ValueError, match="must have at least one dimension"):
        gpu_compute._as_float_array("weights", 3.0)


def test_as_1d_float_array_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="must be a 1-D array"):
        gpu_compute._as_1d_float_array("reference", [[1.0, 2.0]])


def test_as_2d_float_array_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="must be a 2-D array"):
        gpu_compute._as_2d_float_array("states", [1.0, 2.0])


def test_validate_dt_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        gpu_compute._validate_dt(0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        gpu_compute._validate_dt(-0.5)


def test_broadcast_reference_broadcasts_1d_input() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    broadcast = gpu_compute._broadcast_reference(reference, 4)
    assert broadcast.shape == (4, 3)
    np.testing.assert_allclose(broadcast[0], reference)
    np.testing.assert_allclose(broadcast[-1], reference)


def test_broadcast_reference_accepts_matching_2d_batch() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    broadcast = gpu_compute._broadcast_reference(reference, 2)
    assert broadcast is reference


def test_broadcast_reference_rejects_mismatched_2d_batch() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="must be 1-D or match the 2-D batch shape"):
        gpu_compute._broadcast_reference(reference, 3)


def test_gini_numpy_returns_zeros_for_zero_width_input() -> None:
    probabilities = np.zeros((3, 0), dtype=float)
    result = gpu_compute._gini_numpy(probabilities)
    np.testing.assert_allclose(result, np.zeros(3))


def test_population_divergence_numpy_uses_matrix_weights_branch() -> None:
    states = np.array([[1.0, 2.0], [0.5, -1.0]])
    reference = np.array([0.0, 0.0])
    weights = np.eye(2)
    result = gpu_compute._population_divergence_numpy(states, reference, weights, None, 1.0, 0.0)
    expected_scores = np.sum(states**2, axis=1)
    np.testing.assert_allclose(result.scores, expected_scores)


def test_population_convergence_numpy_zero_entropy_for_single_column() -> None:
    weights = np.array([[1.0], [2.0]])
    reference = np.array([[1.0], [1.0]])
    result = gpu_compute._population_convergence_numpy(weights, reference)
    np.testing.assert_allclose(result.normalized_entropy, np.zeros(2))


def test_split_batch_slices_handles_non_positive_batch_size() -> None:
    assert gpu_compute._split_batch_slices(0, 4) == [(0, 0)]
    assert gpu_compute._split_batch_slices(-3, 4) == [(0, 0)]


def test_split_batch_slices_never_exceeds_batch_size() -> None:
    slices = gpu_compute._split_batch_slices(3, 100)
    assert len(slices) == 3
    assert slices[-1][1] == 3


# --------------------------------------------------------------------------
# Direct unit-testing of lane-worker functions, bypassing the process pool.
#
# These plain, module-level functions only show as "covered" to coverage.py
# when called in the *current* process: when dispatched through
# ProcessPoolExecutor, they run in real child OS processes that coverage.py
# does not instrument by default. Calling them directly here is a legitimate
# way to exercise their logic (they are simple, pickláble, side-effect-free
# functions -- calling them directly is equivalent to calling them through
# the pool from a correctness standpoint).
# --------------------------------------------------------------------------


def test_process_pid_returns_current_process_id() -> None:
    assert gpu_compute._process_pid() == os.getpid()


def test_simulated_rk4_lane_worker_matches_manual_reference() -> None:
    states = np.array([[0.4, -0.1], [0.2, 0.3]], dtype=float)
    drift_matrix = np.array([[0.0, 1.0], [-0.5, -0.2]], dtype=float)
    control_matrix = np.array([[1.0], [0.25]], dtype=float)
    controls = np.array([[0.2], [-0.1]], dtype=float)

    pid, chunk = gpu_compute._simulated_rk4_lane(states, drift_matrix, control_matrix, controls, 0.05)

    assert pid == os.getpid()
    np.testing.assert_allclose(chunk, _manual_rk4(states, drift_matrix, control_matrix, controls, 0.05))


def test_simulated_divergence_lane_worker_matches_manual_reference() -> None:
    states = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=float)
    reference = np.array([0.0, 0.5], dtype=float)
    weights = np.array([1.0, 2.0], dtype=float)

    pid, result = gpu_compute._simulated_divergence_lane(states, reference, weights, None, 1.0, 0.0)
    differences, scores, health = _manual_divergence(states, reference, weights, None, 1.0, 0.0)

    assert pid == os.getpid()
    np.testing.assert_allclose(result.differences, differences)
    np.testing.assert_allclose(result.scores, scores)
    np.testing.assert_allclose(result.health, health)


def test_simulated_convergence_lane_worker_matches_manual_reference() -> None:
    weights = np.array([[0.2, 0.3, 0.5], [4.0, 1.0, 0.0]], dtype=float)
    reference = np.array([[1.0, 1.0, 1.0], [0.5, 0.25, 0.25]], dtype=float)

    pid, result = gpu_compute._simulated_convergence_lane(weights, reference)
    normalized, entropy, normalized_entropy, coefficient, kl_divergence, gini = _manual_convergence(
        weights, reference
    )

    assert pid == os.getpid()
    np.testing.assert_allclose(result.normalized_weights, normalized)
    np.testing.assert_allclose(result.entropy, entropy)
    np.testing.assert_allclose(result.normalized_entropy, normalized_entropy)
    np.testing.assert_allclose(result.convergence_coefficient, coefficient)
    np.testing.assert_allclose(result.kl_divergence, kl_divergence)
    np.testing.assert_allclose(result.gini, gini)


# --------------------------------------------------------------------------
# probe_gpu_runtime() branch coverage via monkeypatched probe functions.
# --------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _clear_probe_cache():
    gpu_compute.probe_gpu_runtime.cache_clear()
    yield
    gpu_compute.probe_gpu_runtime.cache_clear()


def test_probe_gpu_runtime_prefers_cupy_when_available(monkeypatch, _clear_probe_cache) -> None:
    cupy_report = GPUCapabilityReport("cupy", True, "fake cupy available")
    monkeypatch.setattr(gpu_compute, "_probe_cupy_runtime", lambda: cupy_report)
    monkeypatch.setattr(
        gpu_compute, "_probe_torch_runtime", lambda: GPUCapabilityReport(None, False, "unused")
    )
    assert gpu_compute.probe_gpu_runtime() is cupy_report


def test_probe_gpu_runtime_falls_back_to_torch_when_cupy_unavailable(monkeypatch, _clear_probe_cache) -> None:
    torch_report = GPUCapabilityReport("torch-cuda", True, "fake torch available")
    monkeypatch.setattr(
        gpu_compute, "_probe_cupy_runtime", lambda: GPUCapabilityReport(None, False, "no cupy")
    )
    monkeypatch.setattr(gpu_compute, "_probe_torch_runtime", lambda: torch_report)
    assert gpu_compute.probe_gpu_runtime() is torch_report


def test_probe_gpu_runtime_reports_neither_available(monkeypatch, _clear_probe_cache) -> None:
    monkeypatch.setattr(
        gpu_compute, "_probe_cupy_runtime", lambda: GPUCapabilityReport(None, False, "no cupy detail")
    )
    monkeypatch.setattr(
        gpu_compute, "_probe_torch_runtime", lambda: GPUCapabilityReport(None, False, "no torch detail")
    )
    report = gpu_compute.probe_gpu_runtime()
    assert report.is_available is False
    assert report.runtime_name is None
    assert "no cupy detail" in report.detail
    assert "no torch detail" in report.detail


# --------------------------------------------------------------------------
# Fake cupy/torch module test doubles.
#
# Neither CuPy nor PyTorch is installed in this environment. To exercise the
# real-GPU-only code paths for coverage purposes, we install small, ethical
# test-double modules into sys.modules (import_module() checks the module
# cache first). This is standard unit-test mocking, not a violation of this
# project's documentation-honesty rules: it never claims real GPU hardware
# was used, it only lets us verify the *logic* of the GPU branches runs
# correctly if a real accelerator module were present.
# --------------------------------------------------------------------------


class _FakeCupyModule:
    """A fake 'cupy' module that proxies straight through to real NumPy.

    CuPy's ndarray API mirrors NumPy's for every operation this module uses
    (asarray, matmul, square, sum, where, ones_like, zeros_like, full_like,
    log, sort, add), so a thin passthrough is sufficient and honest: it
    exercises the exact call shape gpu_compute.py uses without claiming to
    execute anything on real GPU hardware.
    """

    def __init__(self, device_count: int = 1, raise_on_device_count: bool = False):
        self._device_count = device_count
        self._raise_on_device_count = raise_on_device_count
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=self._get_device_count),
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)),
        )

    def _get_device_count(self) -> int:
        if self._raise_on_device_count:
            raise RuntimeError("fake cupy device query failure")
        return self._device_count

    def __getattr__(self, name: str):
        return getattr(np, name)

    def asnumpy(self, values):
        return np.asarray(values)


class _FakeTensor:
    """A minimal fake torch.Tensor wrapper backed by a real NumPy array."""

    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array, dtype=float)

    @property
    def T(self):
        return _FakeTensor(self._array.T)

    @property
    def shape(self):
        return self._array.shape

    def sum(self, dim: int | None = None):
        return _FakeTensor(self._array.sum(axis=dim))

    def reshape(self, *shape):
        return _FakeTensor(self._array.reshape(*shape))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __array__(self, dtype=None):
        return self._array.astype(dtype) if dtype is not None else self._array

    def __add__(self, other):
        return _FakeTensor(self._array + _FakeTensor._value(other))

    __radd__ = __add__

    def __sub__(self, other):
        return _FakeTensor(self._array - _FakeTensor._value(other))

    def __rsub__(self, other):
        return _FakeTensor(_FakeTensor._value(other) - self._array)

    def __mul__(self, other):
        return _FakeTensor(self._array * _FakeTensor._value(other))

    __rmul__ = __mul__

    def __neg__(self):
        return _FakeTensor(-self._array)

    def __truediv__(self, other):
        return _FakeTensor(self._array / _FakeTensor._value(other))

    def __rtruediv__(self, other):
        return _FakeTensor(_FakeTensor._value(other) / self._array)

    def __gt__(self, other):
        return _FakeTensor(self._array > _FakeTensor._value(other))

    @staticmethod
    def _value(other):
        return other._array if isinstance(other, _FakeTensor) else other


class _FakeSortResult:
    def __init__(self, values: np.ndarray):
        self.values = _FakeTensor(values)


class _FakeTorchModule:
    """A fake 'torch' module exposing just enough surface for gpu_compute.py."""

    float64 = "float64"

    def __init__(self, cuda_available: bool = True, raise_on_is_available: bool = False):
        self._cuda_available = cuda_available
        self._raise_on_is_available = raise_on_is_available
        self.cuda = SimpleNamespace(
            is_available=self._is_available,
            synchronize=lambda: None,
            get_device_name=lambda index: "Fake CUDA Device",
        )

    def _is_available(self) -> bool:
        if self._raise_on_is_available:
            raise RuntimeError("fake torch.cuda.is_available failure")
        return self._cuda_available

    def device(self, name: str):
        return f"device:{name}"

    def as_tensor(self, values, dtype=None, device=None):
        return _FakeTensor(np.asarray(values, dtype=float))

    def tensor(self, values, dtype=None, device=None):
        return _FakeTensor(np.asarray(values, dtype=float))

    def matmul(self, left, right):
        return _FakeTensor(np.matmul(_FakeTensor._value(left), _FakeTensor._value(right)))

    def square(self, values):
        return _FakeTensor(np.square(_FakeTensor._value(values)))

    def add(self, left, right):
        return _FakeTensor(_FakeTensor._value(left) + _FakeTensor._value(right))

    def where(self, condition, left, right):
        return _FakeTensor(
            np.where(_FakeTensor._value(condition), _FakeTensor._value(left), _FakeTensor._value(right))
        )

    def ones_like(self, values):
        return _FakeTensor(np.ones_like(_FakeTensor._value(values)))

    def zeros_like(self, values):
        return _FakeTensor(np.zeros_like(_FakeTensor._value(values)))

    def full_like(self, values, fill_value):
        return _FakeTensor(np.full_like(_FakeTensor._value(values), fill_value))

    def log(self, values):
        return _FakeTensor(np.log(_FakeTensor._value(values)))

    def sort(self, values, dim: int = -1):
        return _FakeSortResult(np.sort(_FakeTensor._value(values), axis=dim))


def test_probe_cupy_runtime_reports_not_importable_when_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.delitem(sys.modules, "cupy", raising=False)

    def _raise_import_error(name: str):
        raise ImportError("no module named cupy")

    monkeypatch.setattr(gpu_compute, "import_module", _raise_import_error)
    report = gpu_compute._probe_cupy_runtime()
    assert report.is_available is False
    assert "not importable" in report.detail


def test_probe_cupy_runtime_reports_device_query_failure(monkeypatch) -> None:
    fake_cupy = _FakeCupyModule(raise_on_device_count=True)
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_cupy)
    report = gpu_compute._probe_cupy_runtime()
    assert report.is_available is False
    assert "device query failed" in report.detail


def test_probe_cupy_runtime_reports_zero_devices(monkeypatch) -> None:
    fake_cupy = _FakeCupyModule(device_count=0)
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_cupy)
    report = gpu_compute._probe_cupy_runtime()
    assert report.is_available is False
    assert "zero CUDA devices" in report.detail


def test_probe_cupy_runtime_reports_kernel_probe_failure(monkeypatch) -> None:
    fake_cupy = _FakeCupyModule(device_count=1)

    def _broken_asarray(*args, **kwargs):
        raise RuntimeError("fake kernel failure")

    fake_cupy.asarray = _broken_asarray
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_cupy)
    report = gpu_compute._probe_cupy_runtime()
    assert report.is_available is False
    assert "kernel probe failed" in report.detail


def test_probe_cupy_runtime_reports_full_success(monkeypatch) -> None:
    fake_cupy = _FakeCupyModule(device_count=2)
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_cupy)
    report = gpu_compute._probe_cupy_runtime()
    assert report.is_available is True
    assert report.runtime_name == "cupy"
    assert "2 CUDA device(s)" in report.detail


def test_probe_torch_runtime_reports_not_importable_when_missing(monkeypatch) -> None:
    def _raise_import_error(name: str):
        raise ImportError("no module named torch")

    monkeypatch.setattr(gpu_compute, "import_module", _raise_import_error)
    report = gpu_compute._probe_torch_runtime()
    assert report.is_available is False
    assert "not importable" in report.detail


def test_probe_torch_runtime_reports_is_available_failure(monkeypatch) -> None:
    fake_torch = _FakeTorchModule(raise_on_is_available=True)
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_torch)
    report = gpu_compute._probe_torch_runtime()
    assert report.is_available is False
    assert "torch.cuda.is_available() failed" in report.detail


def test_probe_torch_runtime_reports_cuda_not_available(monkeypatch) -> None:
    fake_torch = _FakeTorchModule(cuda_available=False)
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_torch)
    report = gpu_compute._probe_torch_runtime()
    assert report.is_available is False
    assert "CUDA is not available" in report.detail


def test_probe_torch_runtime_reports_kernel_probe_failure(monkeypatch) -> None:
    fake_torch = _FakeTorchModule(cuda_available=True)

    def _broken_tensor(*args, **kwargs):
        raise RuntimeError("fake torch kernel failure")

    fake_torch.tensor = _broken_tensor
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_torch)
    report = gpu_compute._probe_torch_runtime()
    assert report.is_available is False
    assert "CUDA-PyTorch kernel probe failed" in report.detail


def test_probe_torch_runtime_reports_full_success(monkeypatch) -> None:
    fake_torch = _FakeTorchModule(cuda_available=True)
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_torch)
    report = gpu_compute._probe_torch_runtime()
    assert report.is_available is True
    assert report.runtime_name == "torch-cuda"
    assert "Fake CUDA Device" in report.detail


# --------------------------------------------------------------------------
# End-to-end GPUComputeBackend coverage using the fake cupy runtime.
# --------------------------------------------------------------------------


@pytest.fixture
def fake_cupy_backend(monkeypatch):
    """A GPUComputeBackend wired to a fake, always-available cupy runtime."""
    fake_cupy = _FakeCupyModule(device_count=1)
    monkeypatch.setattr(gpu_compute, "import_module", lambda name: fake_cupy if name == "cupy" else None)
    gpu_compute.probe_gpu_runtime.cache_clear()
    backend = GPUComputeBackend()
    assert backend.runtime_name == "cupy"
    assert backend.is_available() is True
    try:
        yield backend
    finally:
        gpu_compute.probe_gpu_runtime.cache_clear()


def test_fake_cupy_backend_batched_rk4_step_matches_manual_reference(fake_cupy_backend) -> None:
    states = np.array([[0.4, -0.1], [0.2, 0.3]], dtype=float)
    drift_matrix = np.array([[0.0, 1.0], [-0.5, -0.2]], dtype=float)
    control_matrix = np.array([[1.0], [0.25]], dtype=float)
    controls = np.array([[0.2], [-0.1]], dtype=float)

    result = fake_cupy_backend.batched_rk4_step(states, drift_matrix, control_matrix, controls, 0.05)

    np.testing.assert_allclose(result, _manual_rk4(states, drift_matrix, control_matrix, controls, 0.05))
    assert fake_cupy_backend.last_execution_path == "gpu:cupy:batched_rk4_step"


def test_fake_cupy_backend_population_divergence_1d_and_2d_weights(fake_cupy_backend) -> None:
    states = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=float)
    reference = np.array([0.0, 0.5], dtype=float)

    result_1d = fake_cupy_backend.population_divergence(states, reference, np.array([1.0, 2.0]))
    differences, scores, health = _manual_divergence(states, reference, np.array([1.0, 2.0]), None, 1.0, 0.0)
    np.testing.assert_allclose(result_1d.differences, differences)
    np.testing.assert_allclose(result_1d.scores, scores)
    np.testing.assert_allclose(result_1d.health, health)

    weights_2d = np.eye(2)
    result_2d = fake_cupy_backend.population_divergence(states, reference, weights_2d, sigma=np.array([1.0, 2.0]))
    differences2, scores2, health2 = _manual_divergence(states, reference, weights_2d, np.array([1.0, 2.0]), 1.0, 0.0)
    np.testing.assert_allclose(result_2d.differences, differences2)
    np.testing.assert_allclose(result_2d.scores, scores2)
    np.testing.assert_allclose(result_2d.health, health2)


def test_fake_cupy_backend_population_convergence_normal_case(fake_cupy_backend) -> None:
    weights = np.array([[0.2, 0.3, 0.5], [4.0, 1.0, 0.0]], dtype=float)
    reference = np.array([[1.0, 1.0, 1.0], [0.5, 0.25, 0.25]], dtype=float)

    result = fake_cupy_backend.population_convergence(weights, reference=reference)
    normalized, entropy, normalized_entropy, coefficient, kl_divergence, gini = _manual_convergence(
        weights, reference
    )

    np.testing.assert_allclose(result.normalized_weights, normalized)
    np.testing.assert_allclose(result.entropy, entropy)
    np.testing.assert_allclose(result.normalized_entropy, normalized_entropy)
    np.testing.assert_allclose(result.convergence_coefficient, coefficient)
    np.testing.assert_allclose(result.kl_divergence, kl_divergence)
    np.testing.assert_allclose(result.gini, gini)


def test_fake_cupy_backend_population_convergence_single_column_is_zero_entropy(fake_cupy_backend) -> None:
    weights = np.array([[2.0], [3.0]])
    result = fake_cupy_backend.population_convergence(weights)
    np.testing.assert_allclose(result.normalized_entropy, np.zeros(2))


def test_fake_cupy_backend_to_numpy_and_synchronize_roundtrip(fake_cupy_backend) -> None:
    gpu_array = fake_cupy_backend.to_gpu_array([1.0, 2.0, 3.0])
    back = fake_cupy_backend.to_numpy(gpu_array)
    np.testing.assert_allclose(back, [1.0, 2.0, 3.0])
    fake_cupy_backend.synchronize()  # must not raise


# --------------------------------------------------------------------------
# End-to-end GPUComputeBackend coverage using the fake torch-cuda runtime.
# --------------------------------------------------------------------------


@pytest.fixture
def fake_torch_backend(monkeypatch):
    """A GPUComputeBackend wired to a fake, always-available torch-cuda runtime."""
    fake_torch = _FakeTorchModule(cuda_available=True)

    def _fake_import(name: str):
        if name == "cupy":
            raise ImportError("cupy not installed")
        if name == "torch":
            return fake_torch
        raise ImportError(name)

    monkeypatch.setattr(gpu_compute, "import_module", _fake_import)
    gpu_compute.probe_gpu_runtime.cache_clear()
    backend = GPUComputeBackend()
    assert backend.runtime_name == "torch-cuda"
    assert backend.is_available() is True
    try:
        yield backend
    finally:
        gpu_compute.probe_gpu_runtime.cache_clear()


def test_fake_torch_backend_batched_rk4_step_matches_manual_reference(fake_torch_backend) -> None:
    states = np.array([[0.4, -0.1], [0.2, 0.3]], dtype=float)
    drift_matrix = np.array([[0.0, 1.0], [-0.5, -0.2]], dtype=float)
    control_matrix = np.array([[1.0], [0.25]], dtype=float)
    controls = np.array([[0.2], [-0.1]], dtype=float)

    result = fake_torch_backend.batched_rk4_step(states, drift_matrix, control_matrix, controls, 0.05)

    np.testing.assert_allclose(result, _manual_rk4(states, drift_matrix, control_matrix, controls, 0.05))
    assert fake_torch_backend.last_execution_path == "gpu:torch-cuda:batched_rk4_step"


def test_fake_torch_backend_population_convergence_normal_case(fake_torch_backend) -> None:
    weights = np.array([[0.2, 0.3, 0.5], [4.0, 1.0, 0.0]], dtype=float)
    reference = np.array([[1.0, 1.0, 1.0], [0.5, 0.25, 0.25]], dtype=float)

    result = fake_torch_backend.population_convergence(weights, reference=reference)
    normalized, entropy, normalized_entropy, coefficient, kl_divergence, gini = _manual_convergence(
        weights, reference
    )

    np.testing.assert_allclose(result.normalized_weights, normalized)
    np.testing.assert_allclose(result.entropy, entropy)
    np.testing.assert_allclose(result.normalized_entropy, normalized_entropy)
    np.testing.assert_allclose(result.convergence_coefficient, coefficient)
    np.testing.assert_allclose(result.kl_divergence, kl_divergence)
    np.testing.assert_allclose(result.gini, gini)


def test_fake_torch_backend_to_numpy_and_synchronize_roundtrip(fake_torch_backend) -> None:
    gpu_tensor = fake_torch_backend.to_gpu_array([1.0, 2.0, 3.0])
    back = fake_torch_backend.to_numpy(gpu_tensor)
    np.testing.assert_allclose(back, [1.0, 2.0, 3.0])
    fake_torch_backend.synchronize()  # must not raise


# --------------------------------------------------------------------------
# GPUComputeBackend validation-error and edge-case coverage (CPU path).
# --------------------------------------------------------------------------


def test_backend_batched_rk4_step_rejects_bad_shapes() -> None:
    backend = GPUComputeBackend()
    states = np.zeros((2, 2))
    drift = np.zeros((2, 2))
    control = np.zeros((2, 1))
    controls = np.zeros((2, 1))

    with pytest.raises(ValueError, match="drift_matrix must have shape"):
        backend.batched_rk4_step(states, np.zeros((3, 3)), control, controls, 0.1, prefer_gpu=False)
    with pytest.raises(ValueError, match="control_matrix row count"):
        backend.batched_rk4_step(states, drift, np.zeros((3, 1)), controls, 0.1, prefer_gpu=False)
    with pytest.raises(ValueError, match="controls batch size"):
        backend.batched_rk4_step(states, drift, control, np.zeros((3, 1)), 0.1, prefer_gpu=False)
    with pytest.raises(ValueError, match="controls width"):
        backend.batched_rk4_step(states, drift, control, np.zeros((2, 2)), 0.1, prefer_gpu=False)


def test_backend_population_divergence_rejects_bad_shapes() -> None:
    backend = GPUComputeBackend()
    states = np.zeros((2, 2))

    with pytest.raises(ValueError, match="reference length must equal state dimension"):
        backend.population_divergence(states, np.zeros(3), np.ones(2), prefer_gpu=False)
    with pytest.raises(ValueError, match="1-D weights length must equal state dimension"):
        backend.population_divergence(states, np.zeros(2), np.ones(3), prefer_gpu=False)
    with pytest.raises(ValueError, match="2-D weights must have shape"):
        backend.population_divergence(states, np.zeros(2), np.ones((3, 3)), prefer_gpu=False)
    with pytest.raises(ValueError, match="weights must be a 1-D vector or a square 2-D matrix"):
        backend.population_divergence(states, np.zeros(2), np.ones((2, 2, 2)), prefer_gpu=False)


def test_backend_population_divergence_without_sigma_skips_sigma_validation() -> None:
    backend = GPUComputeBackend()
    states = np.array([[1.0, 2.0], [0.5, -1.0]])
    result = backend.population_divergence(states, np.zeros(2), np.ones(2), prefer_gpu=False)
    assert result.scores.shape == (2,)


def test_backend_population_divergence_rejects_bad_sigma() -> None:
    backend = GPUComputeBackend()
    states = np.zeros((2, 2))

    with pytest.raises(ValueError, match="sigma length must equal state dimension"):
        backend.population_divergence(states, np.zeros(2), np.ones(2), sigma=np.ones(3), prefer_gpu=False)
    with pytest.raises(ValueError, match="sigma must not contain zeros"):
        backend.population_divergence(states, np.zeros(2), np.ones(2), sigma=np.zeros(2), prefer_gpu=False)


def test_backend_population_convergence_defaults_reference_when_none() -> None:
    backend = GPUComputeBackend()
    weights = np.array([[1.0, 1.0, 2.0], [3.0, 1.0, 0.0]])
    result = backend.population_convergence(weights, prefer_gpu=False)
    assert result.normalized_weights.shape == weights.shape


def test_backend_population_convergence_zero_width_reference_default() -> None:
    backend = GPUComputeBackend()
    weights = np.zeros((2, 0))
    result = backend.population_convergence(weights, prefer_gpu=False)
    assert result.normalized_weights.shape == (2, 0)


def test_backend_population_convergence_rejects_reference_width_mismatch() -> None:
    backend = GPUComputeBackend()
    weights = np.ones((2, 3))
    with pytest.raises(ValueError, match="reference width must match weights width"):
        backend.population_convergence(weights, reference=np.ones((2, 4)), prefer_gpu=False)


def test_backend_require_gpu_raises_when_unavailable() -> None:
    backend = GPUComputeBackend()
    if backend.is_available():
        pytest.skip("A real GPU runtime is available in this environment")
    with pytest.raises(GPUComputeUnavailableError):
        backend.to_gpu_array([1.0, 2.0])


# --------------------------------------------------------------------------
# SimulatedGPUBackend validation-error coverage (mirrors GPUComputeBackend).
# --------------------------------------------------------------------------


def test_simulated_backend_batched_rk4_step_rejects_bad_shapes() -> None:
    with SimulatedGPUBackend(lanes=2) as backend:
        states = np.zeros((2, 2))
        drift = np.zeros((2, 2))
        control = np.zeros((2, 1))
        controls = np.zeros((2, 1))

        with pytest.raises(ValueError, match="drift_matrix must have shape"):
            backend.batched_rk4_step(states, np.zeros((3, 3)), control, controls, 0.1)
        with pytest.raises(ValueError, match="control_matrix row count"):
            backend.batched_rk4_step(states, drift, np.zeros((3, 1)), controls, 0.1)
        with pytest.raises(ValueError, match="controls batch size"):
            backend.batched_rk4_step(states, drift, control, np.zeros((3, 1)), 0.1)
        with pytest.raises(ValueError, match="controls width"):
            backend.batched_rk4_step(states, drift, control, np.zeros((2, 2)), 0.1)


def test_simulated_backend_population_divergence_rejects_bad_shapes() -> None:
    with SimulatedGPUBackend(lanes=2) as backend:
        states = np.zeros((2, 2))
        with pytest.raises(ValueError, match="reference length must equal state dimension"):
            backend.population_divergence(states, np.zeros(3), np.ones(2))
        with pytest.raises(ValueError, match="1-D weights length must equal state dimension"):
            backend.population_divergence(states, np.zeros(2), np.ones(3))
        with pytest.raises(ValueError, match="2-D weights must have shape"):
            backend.population_divergence(states, np.zeros(2), np.ones((3, 3)))
        with pytest.raises(ValueError, match="weights must be a 1-D vector or a square 2-D matrix"):
            backend.population_divergence(states, np.zeros(2), np.ones((2, 2, 2)))


def test_simulated_backend_population_divergence_without_sigma() -> None:
    with SimulatedGPUBackend(lanes=2) as backend:
        states = np.array([[1.0, 2.0], [0.5, -1.0]])
        result = backend.population_divergence(states, np.zeros(2), np.ones(2))
        assert result.scores.shape == (2,)


def test_simulated_backend_population_divergence_rejects_bad_sigma() -> None:
    with SimulatedGPUBackend(lanes=2) as backend:
        states = np.zeros((2, 2))
        with pytest.raises(ValueError, match="sigma length must equal state dimension"):
            backend.population_divergence(states, np.zeros(2), np.ones(2), sigma=np.ones(3))
        with pytest.raises(ValueError, match="sigma must not contain zeros"):
            backend.population_divergence(states, np.zeros(2), np.ones(2), sigma=np.zeros(2))
