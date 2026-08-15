from __future__ import annotations

import numpy as np
import pytest

from qes.backend import (
    BackendRegistry,
    BackendUnavailableError,
    ComputeBackend,
    GPUBackend,
    NumpyBackend,
    UnknownBackendError,
    benchmark_backend,
    get_backend,
)


class ConstantBackend(ComputeBackend):
    @property
    def name(self) -> str:
        return "constant"

    def array(self, data: object) -> np.ndarray:
        return np.asarray(data, dtype=float)

    def zeros(self, shape: int | tuple[int, ...]) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        return np.zeros(shape, dtype=float)

    def matmul(self, a: object, b: object) -> np.ndarray:
        return np.matmul(self.array(a), self.array(b))

    def elementwise(self, op_name: str, *arrays: object) -> np.ndarray:
        if op_name != "add":
            raise ValueError("constant backend only supports add")
        if len(arrays) < 2:
            raise ValueError("constant backend add expects at least two arrays")
        result = self.array(arrays[0])
        for array in arrays[1:]:
            result = result + self.array(array)
        return result

    def to_numpy(self, array: object) -> np.ndarray:
        return np.asarray(array, dtype=float)


def test_numpy_backend_name_is_cpu() -> None:
    assert NumpyBackend().name == "cpu"


def test_numpy_backend_array_coerces_to_float_ndarray() -> None:
    result = NumpyBackend().array([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))


def test_numpy_backend_zeros_accepts_integer_shape() -> None:
    result = NumpyBackend().zeros(3)
    np.testing.assert_allclose(result, np.zeros(3))


def test_numpy_backend_zeros_accepts_tuple_shape() -> None:
    result = NumpyBackend().zeros((2, 3))
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, np.zeros((2, 3)))


@pytest.mark.parametrize("shape", [(), (-1,), (2, -3), (2.5,)])  # type: ignore[list-item]
def test_numpy_backend_zeros_rejects_invalid_shapes(shape: object) -> None:
    with pytest.raises(ValueError):
        NumpyBackend().zeros(shape)  # type: ignore[arg-type]


def test_numpy_backend_matmul_matches_numpy() -> None:
    backend = NumpyBackend()
    left = [[1.0, 2.0], [3.0, 4.0]]
    right = [[2.0, 0.0], [1.0, 2.0]]
    np.testing.assert_allclose(backend.matmul(left, right), np.matmul(left, right))


def test_numpy_backend_matmul_raises_for_incompatible_shapes() -> None:
    backend = NumpyBackend()
    with pytest.raises(ValueError):
        backend.matmul(np.ones((2, 3)), np.ones((4, 2)))


def test_numpy_backend_elementwise_add_supports_multiple_arrays() -> None:
    backend = NumpyBackend()
    result = backend.elementwise("add", [1, 2], [3, 4], [5, 6])
    np.testing.assert_allclose(result, np.array([9.0, 12.0]))


def test_numpy_backend_elementwise_multiply_matches_numpy() -> None:
    backend = NumpyBackend()
    result = backend.elementwise("multiply", [1, 2], [3, 4])
    np.testing.assert_allclose(result, np.array([3.0, 8.0]))


def test_numpy_backend_elementwise_maximum_matches_numpy() -> None:
    backend = NumpyBackend()
    result = backend.elementwise("maximum", [1, 5, 0], [2, 4, 3])
    np.testing.assert_allclose(result, np.maximum([1, 5, 0], [2, 4, 3]))


def test_numpy_backend_elementwise_negative_is_supported() -> None:
    backend = NumpyBackend()
    result = backend.elementwise("negative", [1, -2, 3])
    np.testing.assert_allclose(result, np.array([-1.0, 2.0, -3.0]))


def test_numpy_backend_elementwise_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match="unsupported elementwise operation"):
        NumpyBackend().elementwise("unknown", [1.0], [2.0])


def test_numpy_backend_elementwise_requires_enough_arrays_for_binary_op() -> None:
    with pytest.raises(ValueError, match="expects at least 2 arrays"):
        NumpyBackend().elementwise("add", [1.0])


def test_numpy_backend_elementwise_requires_exactly_one_array_for_unary_op() -> None:
    with pytest.raises(ValueError, match="expects exactly 1 array"):
        NumpyBackend().elementwise("sqrt", [1.0], [4.0])


def test_numpy_backend_to_numpy_returns_ndarray() -> None:
    result = NumpyBackend().to_numpy([[1, 2], [3, 4]])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_benchmark_backend_times_named_operation() -> None:
    backend = NumpyBackend()
    left = backend.array(np.eye(8))
    right = backend.array(np.eye(8))
    result = benchmark_backend(backend, "matmul", left, right, repeats=3)
    assert result.backend_name == "cpu"
    assert result.operation == "matmul"
    assert result.repeats == 3
    assert result.total_seconds >= result.average_seconds > 0.0
    assert result.result_shape == (8, 8)


def test_benchmark_backend_times_callable_operation() -> None:
    backend = NumpyBackend()

    def add_then_square(active_backend: ComputeBackend, left: object, right: object) -> np.ndarray:
        added = active_backend.elementwise("add", left, right)
        return active_backend.elementwise("multiply", added, added)

    result = benchmark_backend(backend, add_then_square, [1, 2], [3, 4], repeats=2)
    assert result.operation == "add_then_square"
    assert result.repeats == 2
    assert result.result_shape == (2,)


def test_benchmark_backend_rejects_non_positive_repeats() -> None:
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        benchmark_backend(NumpyBackend(), "zeros", (2, 2), repeats=0)


def test_benchmark_backend_rejects_unknown_named_operation() -> None:
    with pytest.raises(ValueError, match="has no operation"):
        benchmark_backend(NumpyBackend(), "does_not_exist", repeats=1)


def test_backend_registry_creates_registered_backend() -> None:
    registry = BackendRegistry()
    registry.register("cpu", NumpyBackend, aliases=("numpy",))
    backend = registry.create("numpy")
    assert isinstance(backend, NumpyBackend)


def test_backend_registry_rejects_duplicate_names() -> None:
    registry = BackendRegistry()
    registry.register("cpu", NumpyBackend)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("cpu", ConstantBackend)


def test_backend_registry_names_are_sorted() -> None:
    registry = BackendRegistry()
    registry.register("gpu", ConstantBackend)
    registry.register("cpu", NumpyBackend, aliases=("numpy",))
    assert registry.names() == ("cpu", "gpu", "numpy")


def test_backend_registry_unknown_backend_raises_clear_error() -> None:
    with pytest.raises(UnknownBackendError, match="unknown backend"):
        get_backend("tpu")


def test_backend_registry_can_fallback_to_cpu_when_gpu_unavailable() -> None:
    backend = get_backend("gpu", fallback_name="cpu")
    assert isinstance(backend, NumpyBackend)


def test_backend_registry_self_fallback_still_raises_unavailable_error() -> None:
    registry = BackendRegistry()
    registry.register("gpu", GPUBackend)
    with pytest.raises(BackendUnavailableError, match="fallback points to itself"):
        registry.create("gpu", fallback_name="gpu")


def test_gpu_backend_raises_honest_unavailable_error_in_this_environment() -> None:
    with pytest.raises(BackendUnavailableError, match="not available in this environment"):
        GPUBackend()


def test_backend_name_normalization_rejects_blank_names() -> None:
    registry = BackendRegistry()
    with pytest.raises(ValueError, match="non-empty string"):
        registry.register("   ", NumpyBackend)


def test_backend_registry_unavailable_without_fallback_still_raises() -> None:
    registry = BackendRegistry()

    def unavailable() -> ComputeBackend:
        raise BackendUnavailableError("no runtime")

    registry.register("gpu", unavailable)
    with pytest.raises(BackendUnavailableError, match="no runtime"):
        registry.create("gpu")


def test_benchmark_backend_handles_callable_returning_none() -> None:
    result = benchmark_backend(NumpyBackend(), lambda backend: None, repeats=1)
    assert result.result_shape is None


def test_gpu_backend_cupy_runtime_methods_cover_supported_operations() -> None:
    class FakeCupy:
        float64 = np.float64

        def __init__(self) -> None:
            self.synced = False
            self.cuda = type(
                "Cuda",
                (),
                {
                    "Stream": type(
                        "StreamNamespace",
                        (),
                        {"null": type("NullStream", (), {"synchronize": self._sync})()},
                    )()
                },
            )()

        def _sync(self) -> None:
            self.synced = True

        @staticmethod
        def asarray(data: object, dtype: object = float) -> np.ndarray:
            return np.asarray(data, dtype=dtype)

        @staticmethod
        def zeros(shape: tuple[int, ...], dtype: object = float) -> np.ndarray:
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def matmul(a: object, b: object) -> np.ndarray:
            return np.matmul(np.asarray(a), np.asarray(b))

        @staticmethod
        def asnumpy(array: object) -> np.ndarray:
            return np.asarray(array)

        abs = staticmethod(np.abs)
        exp = staticmethod(np.exp)
        negative = staticmethod(np.negative)
        sqrt = staticmethod(np.sqrt)
        add = staticmethod(np.add)
        subtract = staticmethod(np.subtract)
        multiply = staticmethod(np.multiply)
        divide = staticmethod(np.divide)
        maximum = staticmethod(np.maximum)
        minimum = staticmethod(np.minimum)
        power = staticmethod(np.power)

    backend = GPUBackend.__new__(GPUBackend)
    backend._runtime = "cupy"
    backend._module = FakeCupy()
    backend._torch_device = None

    assert backend.name == "gpu"
    assert backend.runtime_name == "cupy"
    np.testing.assert_allclose(backend.array([1, 2]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(backend.zeros((1, 2)), np.zeros((1, 2)))
    np.testing.assert_allclose(backend.matmul([[1.0]], [[2.0]]), np.array([[2.0]]))
    np.testing.assert_allclose(backend.elementwise("abs", [-1.0, 2.0]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(backend.elementwise("exp", [0.0, 1.0]), np.exp([0.0, 1.0]))
    np.testing.assert_allclose(backend.elementwise("negative", [1.0, -2.0]), np.array([-1.0, 2.0]))
    np.testing.assert_allclose(backend.elementwise("sqrt", [1.0, 4.0]), np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        backend.elementwise("add", [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]),
        np.array([9.0, 12.0]),
    )
    np.testing.assert_allclose(backend.elementwise("subtract", [5.0, 4.0], [1.0, 2.0]), np.array([4.0, 2.0]))
    np.testing.assert_allclose(backend.elementwise("multiply", [2.0, 3.0], [4.0, 5.0]), np.array([8.0, 15.0]))
    np.testing.assert_allclose(backend.elementwise("divide", [8.0, 9.0], [2.0, 3.0]), np.array([4.0, 3.0]))
    np.testing.assert_allclose(
        backend.elementwise("maximum", [1.0, 5.0], [2.0, 4.0]),
        np.array([2.0, 5.0]),
    )
    np.testing.assert_allclose(
        backend.elementwise("minimum", [1.0, 5.0], [2.0, 4.0]),
        np.array([1.0, 4.0]),
    )
    np.testing.assert_allclose(
        backend.elementwise("power", [2.0, 3.0], [2.0, 2.0]),
        np.array([4.0, 9.0]),
    )
    with pytest.raises(ValueError, match="expects exactly 1 array"):
        backend.elementwise("sqrt", [1.0], [4.0])
    with pytest.raises(ValueError, match="expects at least 2 arrays"):
        backend.elementwise("add", [1.0])
    with pytest.raises(ValueError, match="unsupported elementwise operation"):
        backend.elementwise("bogus", [1.0])
    np.testing.assert_allclose(backend.to_numpy(np.array([7.0])), np.array([7.0]))
    backend.synchronize()
    assert backend._module.synced is True


def test_gpu_backend_torch_runtime_methods_cover_tensor_paths() -> None:
    class FakeTensor:
        def __init__(self, value: object) -> None:
            self.value = np.asarray(value, dtype=float)

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    class FakeTorch:
        float64 = np.float64

        def __init__(self) -> None:
            self.cuda = type("Cuda", (), {"synchronize": self._sync})()
            self.synced = False

        def _sync(self) -> None:
            self.synced = True

        @staticmethod
        def device(name: str) -> str:
            return name

        @staticmethod
        def as_tensor(data: object, dtype: object = None, device: object = None) -> FakeTensor:
            return FakeTensor(data)

        @staticmethod
        def zeros(shape: tuple[int, ...], dtype: object = None, device: object = None) -> FakeTensor:
            return FakeTensor(np.zeros(shape, dtype=float))

        @staticmethod
        def _unwrap(value: object) -> np.ndarray:
            return value.value if isinstance(value, FakeTensor) else np.asarray(value, dtype=float)

        @classmethod
        def matmul(cls, a: object, b: object) -> FakeTensor:
            return FakeTensor(np.matmul(cls._unwrap(a), cls._unwrap(b)))

        @classmethod
        def add(cls, a: object, b: object) -> FakeTensor:
            return FakeTensor(cls._unwrap(a) + cls._unwrap(b))

        @classmethod
        def subtract(cls, a: object, b: object) -> FakeTensor:
            return FakeTensor(cls._unwrap(a) - cls._unwrap(b))

        @classmethod
        def multiply(cls, a: object, b: object) -> FakeTensor:
            return FakeTensor(cls._unwrap(a) * cls._unwrap(b))

        @classmethod
        def divide(cls, a: object, b: object) -> FakeTensor:
            return FakeTensor(cls._unwrap(a) / cls._unwrap(b))

        @classmethod
        def maximum(cls, a: object, b: object) -> FakeTensor:
            return FakeTensor(np.maximum(cls._unwrap(a), cls._unwrap(b)))

        @classmethod
        def minimum(cls, a: object, b: object) -> FakeTensor:
            return FakeTensor(np.minimum(cls._unwrap(a), cls._unwrap(b)))

        @classmethod
        def pow(cls, a: object, b: object) -> FakeTensor:
            return FakeTensor(np.power(cls._unwrap(a), cls._unwrap(b)))

        @classmethod
        def abs(cls, a: object) -> FakeTensor:
            return FakeTensor(np.abs(cls._unwrap(a)))

        @classmethod
        def exp(cls, a: object) -> FakeTensor:
            return FakeTensor(np.exp(cls._unwrap(a)))

        @classmethod
        def negative(cls, a: object) -> FakeTensor:
            return FakeTensor(-cls._unwrap(a))

        @classmethod
        def sqrt(cls, a: object) -> FakeTensor:
            return FakeTensor(np.sqrt(cls._unwrap(a)))

    backend = GPUBackend.__new__(GPUBackend)
    backend._runtime = "torch-cuda"
    backend._module = FakeTorch()
    backend._torch_device = "cuda"

    np.testing.assert_allclose(backend.to_numpy(backend.array([1.0, 2.0])), np.array([1.0, 2.0]))
    np.testing.assert_allclose(backend.to_numpy(backend.zeros(2)), np.zeros(2))
    np.testing.assert_allclose(
        backend.to_numpy(backend.elementwise("power", [2.0], [3.0])),
        np.array([8.0]),
    )
    backend.synchronize()
    assert backend._module.synced is True


def test_gpu_backend_runtime_detection_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import qes.backend as backend_module

    backend = GPUBackend.__new__(GPUBackend)
    backend._torch_device = None
    original_load_cupy = GPUBackend._load_cupy
    original_load_torch = GPUBackend._load_torch

    monkeypatch.setattr(GPUBackend, "_load_cupy", lambda self: ("cupy", object()))
    monkeypatch.setattr(GPUBackend, "_load_torch", lambda self: ("torch-cuda", object()))
    assert backend._load_runtime()[0] == "cupy"

    monkeypatch.setattr(GPUBackend, "_load_cupy", lambda self: None)
    assert backend._load_runtime()[0] == "torch-cuda"
    monkeypatch.setattr(GPUBackend, "_load_cupy", original_load_cupy)
    monkeypatch.setattr(GPUBackend, "_load_torch", original_load_torch)

    class FakeCupy:
        class cuda:
            class runtime:
                @staticmethod
                def getDeviceCount() -> int:
                    return 1

    monkeypatch.setattr(backend_module, "import_module", lambda name: FakeCupy)
    assert backend._load_cupy() == ("cupy", FakeCupy)

    class NoDevicesCupy:
        class cuda:
            class runtime:
                @staticmethod
                def getDeviceCount() -> int:
                    return 0

    monkeypatch.setattr(backend_module, "import_module", lambda name: NoDevicesCupy)
    assert backend._load_cupy() is None

    class BrokenCupy:
        class cuda:
            class runtime:
                @staticmethod
                def getDeviceCount() -> int:
                    raise RuntimeError("boom")

    monkeypatch.setattr(backend_module, "import_module", lambda name: BrokenCupy)
    assert backend._load_cupy() is None

    def raise_import_error(name: str) -> object:
        raise ImportError

    monkeypatch.setattr(backend_module, "import_module", raise_import_error)
    assert backend._load_cupy() is None

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

        @staticmethod
        def device(name: str) -> str:
            return name

    monkeypatch.setattr(backend_module, "import_module", lambda name: FakeTorch)
    runtime_name, runtime_module = backend._load_torch()
    assert runtime_name == "torch-cuda"
    assert runtime_module is FakeTorch
    assert backend._torch_device == "cuda"

    class UnavailableTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(backend_module, "import_module", lambda name: UnavailableTorch)
    assert backend._load_torch() is None

    class BrokenTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                raise RuntimeError("boom")

    monkeypatch.setattr(backend_module, "import_module", lambda name: BrokenTorch)
    assert backend._load_torch() is None

    monkeypatch.setattr(backend_module, "import_module", raise_import_error)
    assert backend._load_torch() is None
