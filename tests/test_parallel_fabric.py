import numpy as np
import pytest

from qes.parallel_fabric import (
    ErrorCorrectionDomain,
    ExecutionDomain,
    MonitoringDomain,
    ParallelComputeDomain,
    PredictionDomain,
    SynchronizationDomain,
)


def test_parallel_compute_domain_map_sequential_and_threaded():
    domain = ParallelComputeDomain(max_workers=None)
    assert domain.map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    threaded = ParallelComputeDomain(max_workers=4)
    assert sorted(threaded.map(lambda x: x * 2, [1, 2, 3])) == [2, 4, 6]


def test_parallel_compute_domain_map_single_item_uses_sequential_path():
    domain = ParallelComputeDomain(max_workers=4)
    assert domain.map(lambda x: x + 1, [5]) == [6]


def test_matrix_multiplex():
    matrices = [np.eye(2), 2 * np.eye(2)]
    vectors = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    results = ParallelComputeDomain.matrix_multiplex(matrices, vectors)
    np.testing.assert_allclose(results[0], [1.0, 1.0])
    np.testing.assert_allclose(results[1], [2.0, 2.0])


def test_synchronization_barrier():
    assert SynchronizationDomain.barrier([1, 2, 3]) == [1, 2, 3]


def test_temporal_match_within_tolerance():
    a = [1.0, 2.0, 3.0]
    b = [1.05, 2.5, 3.02]
    matches = SynchronizationDomain.temporal_match(a, b, tolerance=0.1)
    assert list(matches) == [0, 2]


def test_forecast_next_empty_and_single_and_multi():
    assert PredictionDomain.forecast_next([]) == 0.0
    assert PredictionDomain.forecast_next([5.0]) == 5.0
    assert PredictionDomain.forecast_next([1.0, 3.0]) == pytest.approx(5.0)


def test_trajectory_extrapolates_horizon():
    traj = PredictionDomain.trajectory([1.0, 2.0], horizon=3)
    assert len(traj) == 3
    assert traj[0] == pytest.approx(3.0)


def test_probability_drift_discounts_and_normalizes():
    drifted = PredictionDomain.probability_drift([1.0, 1.0, 1.0], discount=0.5)
    assert drifted.sum() == pytest.approx(1.0)
    assert drifted[-1] > drifted[0]


def test_probability_drift_zero_total_returns_undivided():
    drifted = PredictionDomain.probability_drift([0.0, 0.0], discount=0.5)
    np.testing.assert_allclose(drifted, [0.0, 0.0])


def test_execution_domain_picks_best_branch_and_normalizes():
    branches = [lambda: np.array([1.0, 1.0]), lambda: np.array([3.0, 1.0])]
    result = ExecutionDomain.execute_branches(branches, score_fn=lambda o: o.sum())
    assert result.branch == 1
    assert result.output.sum() == pytest.approx(1.0)


def test_execution_domain_negative_sum_not_normalized():
    branches = [lambda: np.array([-1.0, -1.0])]
    result = ExecutionDomain.execute_branches(branches, score_fn=lambda o: o.sum())
    np.testing.assert_allclose(result.output, [-1.0, -1.0])


def test_monitoring_dashboard_empty_and_populated():
    empty = MonitoringDomain.dashboard([])
    assert empty == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    stats = MonitoringDomain.dashboard([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(3.0)


def test_stability_probe():
    assert MonitoringDomain.stability_probe([], threshold=0.1)
    assert MonitoringDomain.stability_probe([1.0, 1.0, 1.0], threshold=0.01)
    assert not MonitoringDomain.stability_probe([1.0, 100.0], threshold=0.01)


def test_drift_monitor():
    assert MonitoringDomain.drift_monitor(1.0, 5.0, threshold=1.0)
    assert not MonitoringDomain.drift_monitor(1.0, 1.5, threshold=1.0)


def test_ecc_correct_median():
    replicas = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([9.0, 9.0])]
    corrected = ErrorCorrectionDomain.ecc_correct(replicas)
    np.testing.assert_allclose(corrected, [1.0, 1.0])


def test_tri_cycle_correct_clips_and_damps():
    x = np.array([10.0, -10.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    corrected = ErrorCorrectionDomain.tri_cycle_correct(x, lower, upper)
    assert np.all(corrected >= lower) and np.all(corrected <= upper)


def test_integrity_validate_true_and_false():
    good_replicas = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    assert ErrorCorrectionDomain.integrity_validate(good_replicas)

    bad_replicas = [np.array([1.0, 1.0]), np.array([100.0, 100.0])]
    assert not ErrorCorrectionDomain.integrity_validate(bad_replicas)


def test_parallel_compute_domain_and_matrix_multiplex_validation():
    with pytest.raises(TypeError, match="max_workers must be an integer or None"):
        ParallelComputeDomain(max_workers=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_workers must be > 0"):
        ParallelComputeDomain(max_workers=0)

    domain = ParallelComputeDomain()
    with pytest.raises(TypeError, match="fn must be callable"):
        domain.map(1, [1, 2])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="same length"):
        ParallelComputeDomain.matrix_multiplex([np.eye(2)], [])
    assert ParallelComputeDomain.matrix_multiplex([], []) == []
    with pytest.raises(ValueError, match="each matrix must be two-dimensional"):
        ParallelComputeDomain.matrix_multiplex([np.array([1.0, 2.0])], [np.array([1.0, 2.0])])
    with pytest.raises(ValueError, match="each vector must be one-dimensional"):
        ParallelComputeDomain.matrix_multiplex([np.eye(2)], [np.array([[1.0], [2.0]])])
    with pytest.raises(ValueError, match="shape mismatch"):
        ParallelComputeDomain.matrix_multiplex([np.eye(2)], [np.array([1.0, 2.0, 3.0])])
    with pytest.raises(ValueError, match="contain only finite values"):
        ParallelComputeDomain.matrix_multiplex(
            [np.array([[1.0, np.inf], [0.0, 1.0]])],
            [np.array([1.0, 1.0])],
        )


def test_matrix_multiplex_falls_back_for_mixed_shapes():
    results = ParallelComputeDomain.matrix_multiplex(
        [np.eye(2), np.eye(3)],
        [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])],
    )
    np.testing.assert_allclose(results[0], [1.0, 2.0])
    np.testing.assert_allclose(results[1], [1.0, 2.0, 3.0])


def test_synchronization_and_prediction_validation():
    with pytest.raises(ValueError, match="series_a and series_b must have the same shape"):
        SynchronizationDomain.temporal_match([1.0], [1.0, 2.0], tolerance=0.1)
    with pytest.raises(ValueError, match="tolerance must be finite and >= 0"):
        SynchronizationDomain.temporal_match([1.0], [1.0], tolerance=-0.1)

    with pytest.raises(TypeError, match="horizon must be an integer"):
        PredictionDomain.trajectory([1.0], horizon=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizon must be >= 0"):
        PredictionDomain.trajectory([1.0], horizon=-1)
    assert PredictionDomain.trajectory([1.0, 2.0], horizon=0) == []
    assert PredictionDomain.trajectory([], horizon=2) == [0.0, 0.0]
    assert PredictionDomain.trajectory([5.0], horizon=2) == [5.0, 5.0]

    with pytest.raises(ValueError, match="weights must be >= 0"):
        PredictionDomain.probability_drift([1.0, -1.0])
    with pytest.raises(ValueError, match="discount must be finite and in \\[0, 1\\]"):
        PredictionDomain.probability_drift([1.0], discount=1.5)
    assert PredictionDomain.probability_drift([], discount=0.5).tolist() == []


def test_execution_domain_validation_paths():
    with pytest.raises(ValueError, match="branches must be non-empty"):
        ExecutionDomain.execute_branches([], score_fn=lambda o: 0.0)
    with pytest.raises(TypeError, match="score_fn must be callable"):
        ExecutionDomain.execute_branches([lambda: np.array([1.0])], score_fn=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="branches must contain callables"):
        ExecutionDomain.execute_branches([1], score_fn=lambda o: 0.0)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="one-dimensional"):
        ExecutionDomain.execute_branches([lambda: np.array([[1.0]])], score_fn=lambda o: 0.0)
    with pytest.raises(ValueError, match="contain only finite values"):
        ExecutionDomain.execute_branches([lambda: np.array([np.inf])], score_fn=lambda o: 0.0)
    with pytest.raises(ValueError, match="return finite scores"):
        ExecutionDomain.execute_branches([lambda: np.array([1.0])], score_fn=lambda o: float("nan"))


def test_monitoring_and_error_correction_validation():
    with pytest.raises(ValueError, match="series must be one-dimensional"):
        MonitoringDomain.dashboard([[1.0], [2.0]])
    with pytest.raises(ValueError, match="series must contain only finite values"):
        MonitoringDomain.dashboard([1.0, np.inf])
    with pytest.raises(ValueError, match="threshold must be finite and >= 0"):
        MonitoringDomain.stability_probe([1.0], threshold=-0.1)
    with pytest.raises(ValueError, match="baseline and current must be finite"):
        MonitoringDomain.drift_monitor(float("nan"), 1.0, threshold=0.1)
    with pytest.raises(ValueError, match="threshold must be finite and >= 0"):
        MonitoringDomain.drift_monitor(1.0, 2.0, threshold=-1.0)

    with pytest.raises(ValueError, match="replicas must be non-empty"):
        ErrorCorrectionDomain.ecc_correct([])
    with pytest.raises(ValueError, match="same shape"):
        ErrorCorrectionDomain.ecc_correct([np.array([1.0]), np.array([1.0, 2.0])])
    with pytest.raises(ValueError, match="contain only finite values"):
        ErrorCorrectionDomain.ecc_correct([np.array([1.0]), np.array([np.inf])])
    with pytest.raises(ValueError, match="matching shapes"):
        ErrorCorrectionDomain.tri_cycle_correct(np.array([1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="contain only finite values"):
        ErrorCorrectionDomain.tri_cycle_correct(
            np.array([1.0]),
            np.array([0.0]),
            np.array([np.inf]),
        )
    with pytest.raises(ValueError, match="lower must be <= upper"):
        ErrorCorrectionDomain.tri_cycle_correct(
            np.array([1.0]),
            np.array([2.0]),
            np.array([1.0]),
        )
    with pytest.raises(ValueError, match="tolerance must be finite and >= 0"):
        ErrorCorrectionDomain.integrity_validate([np.array([1.0])], tolerance=-1.0)
