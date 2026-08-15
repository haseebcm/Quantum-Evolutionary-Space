import numpy as np

from qes.digital_twin import DigitalTwin


def test_twin_residual_is_difference():
    twin = DigitalTwin()
    residual = twin.twin_residual(observed=np.array([1.0, 2.0]), predicted=np.array([0.5, 2.5]))
    np.testing.assert_allclose(residual, [0.5, -0.5])


def test_weighted_twin_error_uses_identity_when_no_covariance():
    twin = DigitalTwin()
    residual = np.array([1.0, 2.0])
    # identity R^-1 => e^T e = 1+4=5
    assert twin.weighted_twin_error(residual) == 5.0


def test_weighted_twin_error_uses_provided_covariance():
    twin = DigitalTwin(noise_covariance=np.diag([2.0, 2.0]))
    residual = np.array([2.0, 2.0])
    # R^-1 = diag(0.5, 0.5); e^T R^-1 e = 2*0.5*4=4
    assert twin.weighted_twin_error(residual) == 4.0


def test_evidence_update_favors_higher_likelihood_room():
    weights = [0.5, 0.5]
    likelihoods = [0.9, 0.1]
    updated = DigitalTwin.evidence_update(weights, likelihoods)
    assert updated[0] > updated[1]
    assert abs(updated.sum() - 1.0) < 1e-9


def test_evidence_update_handles_zero_likelihoods_gracefully():
    weights = [0.5, 0.5]
    likelihoods = [0.0, 0.0]
    updated = DigitalTwin.evidence_update(weights, likelihoods)
    np.testing.assert_allclose(updated, [0.5, 0.5])


def test_likelihood_gaussian_peaks_at_zero_residual():
    cov = np.eye(2)
    at_zero = DigitalTwin.likelihood_gaussian(np.zeros(2), cov)
    away = DigitalTwin.likelihood_gaussian(np.array([3.0, 3.0]), cov)
    assert at_zero > away


def test_record_residual_bounds_history_to_window():
    twin = DigitalTwin(window=3)
    for delta in [1.0, 2.0, 3.0, 4.0]:
        twin.record_residual(delta)
    assert len(twin._residual_history) == 3
    assert twin._residual_history == [2.0, 3.0, 4.0]


def test_rolling_mean_error_is_zero_with_no_history():
    twin = DigitalTwin()
    assert twin.rolling_mean_error() == 0.0


def test_rolling_mean_error_averages_recorded_residuals():
    twin = DigitalTwin(window=10)
    for delta in [1.0, 3.0]:
        twin.record_residual(delta)
    assert twin.rolling_mean_error() == 2.0


def test_detect_drift_flags_sustained_high_error():
    twin = DigitalTwin(window=5)
    for delta in [10.0, 10.0, 10.0]:
        twin.record_residual(delta)
    assert twin.detect_drift(threshold=5.0)
    assert not twin.detect_drift(threshold=15.0)
