import numpy as np

from qes.gpu_compute import GPUComputeBackend, PopulationConvergenceResult, SimulatedGPUBackend


def test_to_numpy_cpu_path_returns_numpy_array():
    backend = GPUComputeBackend()
    arr = backend.to_numpy([1.0, 2.0, 3.0])
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)


def test_simulated_population_convergence_zero_width():
    with SimulatedGPUBackend(lanes=2) as sim:
        weights = np.zeros((2, 0), dtype=float)
        result = sim.population_convergence(weights)
        assert isinstance(result, PopulationConvergenceResult)
        assert result.normalized_weights.shape == (2, 0)
        assert result.entropy.shape == (2,)
        assert np.all(result.entropy == 0.0)


def test_simulated_population_convergence_with_default_reference():
    with SimulatedGPUBackend(lanes=2) as sim:
        weights = np.ones((2, 3), dtype=float)
        result = sim.population_convergence(weights)
        assert result.normalized_weights.shape == (2, 3)
        assert result.entropy.shape == (2,)


def test_simulated_population_convergence_bad_reference_width_raises():
    with SimulatedGPUBackend(lanes=2) as sim:
        weights = np.ones((2, 3), dtype=float)
        import pytest

        with pytest.raises(ValueError):
            sim.population_convergence(weights, reference=np.array([0.5, 0.5]))
