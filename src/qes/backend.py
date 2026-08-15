"""Phase 10 compute backend abstraction for classical QES array execution.

QES uses "quantum/universe" language for possibility-space search, but this
module is an ordinary classical execution layer. It lets room/state-vector code
target a backend interface instead of hard-coding NumPy calls. In this
environment the guaranteed implementation is CPU-based NumPy execution; optional
GPU-style backends detect real support and otherwise fail honestly.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np


class BackendError(RuntimeError):
    """Base class for backend-related errors."""


class BackendUnavailableError(BackendError):
    """Raised when an optional backend cannot execute in the current environment."""


class UnknownBackendError(BackendError):
    """Raised when a backend name is not registered."""


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary of an operation benchmark on one backend.

    Attributes:
        backend_name: backend used to execute the operation.
        operation: operation name or callable name that was benchmarked.
        repeats: number of timed repetitions.
        total_seconds: total wall-clock time across all repetitions.
        average_seconds: average wall-clock time per repetition.
        result_shape: numpy-visible shape of the final result, when available.
    """

    backend_name: str
    operation: str
    repeats: int
    total_seconds: float
    average_seconds: float
    result_shape: tuple[int, ...] | None


def _normalize_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize an array shape specification."""
    if isinstance(shape, int):
        normalized: tuple[int, ...] = (shape,)
    else:
        normalized = tuple(shape)

    if not normalized:
        raise ValueError("shape must contain at least one dimension")
    if any((not isinstance(dim, int)) or dim < 0 for dim in normalized):
        raise ValueError("shape dimensions must be non-negative integers")
    return normalized


def _normalize_backend_name(name: str) -> str:
    """Normalize a backend identifier for registry lookup."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("backend name must be a non-empty string")
    return normalized


class ComputeBackend(ABC):
    """Abstract classical array backend for QES compute operations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier such as ``cpu`` or ``gpu``."""

    @abstractmethod
    def array(self, data: Any) -> Any:
        """Create a backend-native array from array-like input."""

    @abstractmethod
    def zeros(self, shape: int | Sequence[int]) -> Any:
        """Create a zero-filled backend-native array."""

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Compute a matrix multiplication on the backend."""

    @abstractmethod
    def elementwise(self, op_name: str, *arrays: Any) -> Any:
        """Apply a named elementwise operation on backend arrays."""

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert a backend-native array into a NumPy array."""

    def synchronize(self) -> None:
        """Block until pending backend work completes.

        CPU NumPy execution is synchronous, so the default implementation is a
        no-op. Optional accelerator backends override this when needed.
        """
        return


class NumpyBackend(ComputeBackend):
    """Working CPU backend backed by ordinary NumPy arrays."""

    _BINARY_OPS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.divide,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "power": np.power,
    }
    _UNARY_OPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "abs": np.abs,
        "exp": np.exp,
        "negative": np.negative,
        "sqrt": np.sqrt,
    }

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "cpu"

    def array(self, data: Any) -> np.ndarray:
        """Create a NumPy array with QES's standard float dtype."""
        return np.asarray(data, dtype=float)

    def zeros(self, shape: int | Sequence[int]) -> np.ndarray:
        """Create a zero-filled NumPy array."""
        return np.zeros(_normalize_shape(shape), dtype=float)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        """Compute a matrix multiplication using NumPy."""
        return np.matmul(self.array(a), self.array(b))

    def elementwise(self, op_name: str, *arrays: Any) -> np.ndarray:
        """Apply a supported elementwise NumPy operation."""
        normalized = op_name.strip().lower()
        if normalized in self._UNARY_OPS:
            if len(arrays) != 1:
                raise ValueError(f"elementwise operation {normalized!r} expects exactly 1 array")
            return self._UNARY_OPS[normalized](self.array(arrays[0]))

        if normalized in self._BINARY_OPS:
            if len(arrays) < 2:
                raise ValueError(f"elementwise operation {normalized!r} expects at least 2 arrays")
            result = self.array(arrays[0])
            operator = self._BINARY_OPS[normalized]
            for array in arrays[1:]:
                result = operator(result, self.array(array))
            return result

        supported = sorted([*self._UNARY_OPS.keys(), *self._BINARY_OPS.keys()])
        raise ValueError(f"unsupported elementwise operation {op_name!r}; supported operations: {supported}")

    def to_numpy(self, array: Any) -> np.ndarray:
        """Return a NumPy view/copy of the given array-like object."""
        return np.asarray(array, dtype=float)


class GPUBackend(ComputeBackend):
    """Optional GPU-style backend that only activates when a real runtime exists.

    The implementation honestly detects either CuPy or CUDA-enabled PyTorch.
    If neither can execute on a GPU in the current environment, construction
    raises :class:`BackendUnavailableError` instead of pretending acceleration
    occurred.
    """

    _BINARY_NAMES = {"add", "subtract", "multiply", "divide", "maximum", "minimum", "power"}
    _UNARY_NAMES = {"abs", "exp", "negative", "sqrt"}

    def __init__(self) -> None:
        """Initialize a GPU runtime or raise if none is available."""
        self._runtime: str
        self._module: Any
        self._torch_device: Any | None = None
        self._runtime, self._module = self._load_runtime()

    @property
    def name(self) -> str:
        """Return the backend identifier."""
        return "gpu"

    @property
    def runtime_name(self) -> str:
        """Return the detected accelerator runtime name."""
        return self._runtime

    def array(self, data: Any) -> Any:
        """Create a backend-native accelerator array."""
        if self._runtime == "cupy":
            return self._module.asarray(data, dtype=float)
        return self._module.as_tensor(data, dtype=self._module.float64, device=self._torch_device)

    def zeros(self, shape: int | Sequence[int]) -> Any:
        """Create a zero-filled accelerator array."""
        normalized = _normalize_shape(shape)
        if self._runtime == "cupy":
            return self._module.zeros(normalized, dtype=float)
        return self._module.zeros(normalized, dtype=self._module.float64, device=self._torch_device)

    def matmul(self, a: Any, b: Any) -> Any:
        """Compute a matrix multiplication on the accelerator runtime."""
        return self._module.matmul(self.array(a), self.array(b))

    def elementwise(self, op_name: str, *arrays: Any) -> Any:
        """Apply a supported elementwise operation on the accelerator runtime."""
        normalized = op_name.strip().lower()

        if normalized in self._UNARY_NAMES:
            if len(arrays) != 1:
                raise ValueError(f"elementwise operation {normalized!r} expects exactly 1 array")
            return self._apply_unary(normalized, self.array(arrays[0]))

        if normalized in self._BINARY_NAMES:
            if len(arrays) < 2:
                raise ValueError(f"elementwise operation {normalized!r} expects at least 2 arrays")
            result = self.array(arrays[0])
            for array in arrays[1:]:
                result = self._apply_binary(normalized, result, self.array(array))
            return result

        supported = sorted([*self._UNARY_NAMES, *self._BINARY_NAMES])
        raise ValueError(f"unsupported elementwise operation {op_name!r}; supported operations: {supported}")

    def to_numpy(self, array: Any) -> np.ndarray:
        """Move accelerator data back to a NumPy array on CPU."""
        if self._runtime == "cupy":
            return self._module.asnumpy(array)
        return array.detach().cpu().numpy()

    def synchronize(self) -> None:
        """Wait for pending accelerator work to complete before timing results."""
        if self._runtime == "cupy":
            self._module.cuda.Stream.null.synchronize()
            return
        self._module.cuda.synchronize()

    def _apply_unary(self, op_name: str, array: Any) -> Any:
        if op_name == "abs":
            return self._module.abs(array)
        if op_name == "exp":
            return self._module.exp(array)
        if op_name == "negative":
            return self._module.negative(array)
        return self._module.sqrt(array)

    def _apply_binary(self, op_name: str, left: Any, right: Any) -> Any:
        if op_name == "add":
            return self._module.add(left, right)
        if op_name == "subtract":
            return self._module.subtract(left, right)
        if op_name == "multiply":
            return self._module.multiply(left, right)
        if op_name == "divide":
            return self._module.divide(left, right)
        if op_name == "maximum":
            return self._module.maximum(left, right)
        if op_name == "minimum":
            return self._module.minimum(left, right)
        if self._runtime.startswith("torch"):
            return self._module.pow(left, right)
        return self._module.power(left, right)

    def _load_runtime(self) -> tuple[str, Any]:
        cupy_runtime = self._load_cupy()
        if cupy_runtime is not None:
            return cupy_runtime

        torch_runtime = self._load_torch()
        if torch_runtime is not None:
            return torch_runtime

        raise BackendUnavailableError(
            "GPU backend not available in this environment; no usable CuPy or "
            "CUDA-enabled PyTorch runtime was detected"
        )

    def _load_cupy(self) -> tuple[str, Any] | None:
        try:
            cupy = import_module("cupy")
        except ImportError:
            return None

        try:
            device_count = int(cupy.cuda.runtime.getDeviceCount())
        except Exception:
            return None

        if device_count < 1:
            return None
        return ("cupy", cupy)

    def _load_torch(self) -> tuple[str, Any] | None:
        try:
            torch = import_module("torch")
        except ImportError:
            return None

        try:
            available = bool(torch.cuda.is_available())
        except Exception:
            return None

        if not available:
            return None

        self._torch_device = torch.device("cuda")
        return ("torch-cuda", torch)


class BackendRegistry:
    """Factory registry for backend instances requested by name."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._factories: dict[str, Callable[[], ComputeBackend]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], ComputeBackend],
        aliases: Sequence[str] = (),
    ) -> None:
        """Register a backend factory under one primary name and optional aliases."""
        names = [_normalize_backend_name(name), *(_normalize_backend_name(alias) for alias in aliases)]
        for backend_name in names:
            if backend_name in self._factories:
                raise ValueError(f"backend {backend_name!r} is already registered")
        for backend_name in names:
            self._factories[backend_name] = factory

    def create(self, name: str, fallback_name: str | None = None) -> ComputeBackend:
        """Create a backend by name, optionally falling back to another backend."""
        normalized = _normalize_backend_name(name)
        if normalized not in self._factories:
            known = ", ".join(sorted(self._factories))
            raise UnknownBackendError(f"unknown backend {name!r}; registered backends: {known}")

        try:
            return self._factories[normalized]()
        except BackendUnavailableError:
            if fallback_name is None:
                raise

        fallback = _normalize_backend_name(fallback_name) if fallback_name is not None else None
        if fallback == normalized:
            raise BackendUnavailableError(f"backend {name!r} is unavailable and fallback points to itself")
        return self.create(fallback_name)

    def names(self) -> tuple[str, ...]:
        """Return the registered backend names and aliases."""
        return tuple(sorted(self._factories))


DEFAULT_BACKEND_REGISTRY = BackendRegistry()
DEFAULT_BACKEND_REGISTRY.register("cpu", NumpyBackend, aliases=("numpy",))
DEFAULT_BACKEND_REGISTRY.register("gpu", GPUBackend, aliases=("cuda", "rocm", "accelerator"))


def get_backend(
    name: str,
    *,
    fallback_name: str | None = None,
    registry: BackendRegistry | None = None,
) -> ComputeBackend:
    """Return a backend instance from a registry by name.

    Args:
        name: requested backend identifier.
        fallback_name: optional alternate backend name used when the requested
            backend exists but is unavailable in the current environment.
        registry: optional registry override; defaults to the module registry.
    """
    active_registry = DEFAULT_BACKEND_REGISTRY if registry is None else registry
    return active_registry.create(name, fallback_name=fallback_name)


def benchmark_backend(
    backend: ComputeBackend,
    operation: str | Callable[..., Any],
    *args: Any,
    repeats: int = 5,
) -> BenchmarkResult:
    """Benchmark one backend operation using measured wall-clock time.

    Args:
        backend: backend used to execute the operation.
        operation: backend method name such as ``"matmul"`` or a callable that
            accepts ``(backend, *args)``.
        *args: arguments forwarded to the operation.
        repeats: number of repetitions to time; must be at least 1.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    if isinstance(operation, str):
        operation_name = operation
        if not hasattr(backend, operation_name):
            raise ValueError(f"backend {backend.name!r} has no operation {operation_name!r}")
        runner = getattr(backend, operation_name)
    else:
        operation_name = getattr(operation, "__name__", "callable")

        def runner(*runner_args: Any) -> Any:
            return operation(backend, *runner_args)

    total_seconds = 0.0
    last_result: Any = None
    for _ in range(repeats):
        t_start = time.perf_counter()
        last_result = runner(*args)
        backend.synchronize()
        total_seconds += time.perf_counter() - t_start

    result_shape: tuple[int, ...] | None = None
    if last_result is not None:
        result_shape = tuple(backend.to_numpy(last_result).shape)

    return BenchmarkResult(
        backend_name=backend.name,
        operation=operation_name,
        repeats=repeats,
        total_seconds=total_seconds,
        average_seconds=total_seconds / repeats,
        result_shape=result_shape,
    )


__all__ = [
    "BackendError",
    "BackendRegistry",
    "BackendUnavailableError",
    "BenchmarkResult",
    "ComputeBackend",
    "DEFAULT_BACKEND_REGISTRY",
    "GPUBackend",
    "NumpyBackend",
    "UnknownBackendError",
    "benchmark_backend",
    "get_backend",
]
