import numpy as np
import pytest

from qes.validation import (
    FailureRecoveryLoop,
    FunctionalSafetySystem,
    IntegrityValidator,
    ValidationFramework,
)


def test_validation_framework_reports_failures():
    framework = ValidationFramework()
    framework.add_check("non_negative", lambda x: np.all(x >= 0))
    framework.add_check("bounded", lambda x: np.all(x <= 10))

    report_ok = framework.validate(np.array([1.0, 2.0]))
    assert report_ok.passed
    assert report_ok.failures == []

    report_bad = framework.validate(np.array([-1.0, 20.0]))
    assert not report_bad.passed
    assert set(report_bad.failures) == {"non_negative", "bounded"}


def test_functional_safety_system_margin_rejects_negative():
    with pytest.raises(ValueError):
        FunctionalSafetySystem(lower=np.array([0.0]), upper=np.array([1.0]), margin=-0.1)


def test_functional_safety_system_rejects_shape_and_bound_mismatches():
    with pytest.raises(ValueError, match="same shape"):
        FunctionalSafetySystem(lower=np.array([0.0]), upper=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="upper must be >="):
        FunctionalSafetySystem(lower=np.array([1.0]), upper=np.array([0.0]))


def test_functional_safety_system_scores_center_highest():
    system = FunctionalSafetySystem(lower=np.array([-1.0]), upper=np.array([1.0]), margin=0.1)
    center_score = system.safety_score(np.array([0.0]))
    edge_score = system.safety_score(np.array([1.0]))
    assert center_score > edge_score
    assert system.is_safe(np.array([0.0]))
    assert not system.is_safe(np.array([1.0]), threshold=0.5)


def test_failure_recovery_loop_no_op_when_not_failed():
    loop = FailureRecoveryLoop(lower=np.array([-1.0]), upper=np.array([1.0]))
    outcome = loop.recover(np.array([0.5]))
    assert not outcome.was_null
    np.testing.assert_allclose(outcome.recovered_state, [0.5])


def test_failure_recovery_loop_recovers_out_of_bounds_state():
    loop = FailureRecoveryLoop(lower=np.array([-1.0]), upper=np.array([1.0]))
    outcome = loop.recover(np.array([100.0]))
    assert outcome.was_null
    assert np.all(outcome.recovered_state <= 1.0)


def test_failure_recovery_loop_recovers_non_finite_state():
    loop = FailureRecoveryLoop(lower=np.array([-1.0]), upper=np.array([1.0]))
    outcome = loop.recover(np.array([np.nan]))
    assert outcome.was_null


def test_failure_recovery_loop_with_custom_regenerate_fn():
    loop = FailureRecoveryLoop(lower=np.array([-1.0]), upper=np.array([1.0]))
    outcome = loop.recover(np.array([100.0]), regenerate_fn=lambda base: base + 0.5)
    assert outcome.was_null
    np.testing.assert_allclose(outcome.recovered_state, [0.5])


def test_failure_recovery_loop_default_baseline_is_midpoint():
    loop = FailureRecoveryLoop(lower=np.array([-2.0]), upper=np.array([2.0]))
    np.testing.assert_allclose(loop.baseline, [0.0])


def test_failure_recovery_loop_rejects_invalid_shapes_bounds_and_baseline():
    with pytest.raises(ValueError, match="same shape"):
        FailureRecoveryLoop(lower=np.array([0.0]), upper=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="upper must be >="):
        FailureRecoveryLoop(lower=np.array([1.0]), upper=np.array([0.0]))
    with pytest.raises(ValueError, match="baseline must match"):
        FailureRecoveryLoop(
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            baseline=np.array([0.0, 1.0]),
        )


def test_failure_recovery_loop_is_failed_detects_bound_violations():
    loop = FailureRecoveryLoop(lower=np.array([-1.0]), upper=np.array([1.0]))
    assert loop.is_failed(np.array([-2.0])) is True
    assert loop.is_failed(np.array([2.0])) is True
    assert loop.is_failed(np.array([0.0])) is False


def test_integrity_validator_consensus_and_consistency():
    replicas = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    consensus = IntegrityValidator.consensus(replicas)
    np.testing.assert_allclose(consensus, [1.0, 1.0])
    assert IntegrityValidator.is_consistent(replicas)

    bad_replicas = [np.array([1.0, 1.0]), np.array([50.0, 50.0])]
    assert not IntegrityValidator.is_consistent(bad_replicas)


def test_integrity_validator_consensus_empty_raises():
    with pytest.raises(ValueError):
        IntegrityValidator.consensus([])
