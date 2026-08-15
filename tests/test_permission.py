import numpy as np
import pytest

from qes.permission import (
    AdaptivePermission,
    GenesisPermission,
    accelerated_cci,
    cascade_collapse_index,
    exceedance,
    permission_margin,
    soft_permission,
    violation_energy,
)


def test_violation_energy_zero_inside_bounds():
    x = np.array([0.5, -0.5])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    assert violation_energy(x, lower, upper) == 0.0


def test_violation_energy_positive_outside_bounds():
    x = np.array([2.0, -3.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    # over: (2-1)^2=1 ; under: (-1 - -3)^2=4
    assert violation_energy(x, lower, upper) == 5.0


def test_exceedance_matches_manual_calc():
    x = np.array([2.0, 0.0, -3.0])
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(exceedance(x, lower, upper), [1.0, 0.0, 2.0])


def test_cascade_collapse_index_without_coupling():
    e = np.array([1.0, 2.0])
    w = np.array([0.5, 0.5])
    assert cascade_collapse_index(e, w) == 1.5


def test_cascade_collapse_index_with_coupling_adds_penalty():
    e = np.array([1.0, 1.0])
    w = np.array([1.0, 1.0])
    coupling = np.eye(2)
    cci = cascade_collapse_index(e, w, coupling, gamma=1.0)
    # base = 2, coupled penalty = ||[1,1]||^2 = 2 -> total 4
    assert cci == 4.0


def test_accelerated_cci_only_adds_positive_rates():
    base_cci = 1.0
    w = np.array([1.0, 1.0])
    e_rate = np.array([2.0, -5.0])
    result = accelerated_cci(base_cci, w, e_rate, eta=1.0)
    assert result == 1.0 + 2.0  # negative rate contributes nothing


def test_permission_margin_shrinks_with_cci():
    x = np.array([0.0])
    lower = np.array([-1.0])
    upper = np.array([1.0])
    m0 = permission_margin(x, lower, upper, cci=0.0)
    m1 = permission_margin(x, lower, upper, cci=1.0)
    # at x=0 the margin to both bounds is 0.5 of the span
    assert m0 == pytest.approx(0.5)
    assert m1 == pytest.approx(0.25)


def test_soft_permission_decays_with_violation_and_cci():
    high = soft_permission(phi=0.0, cci=0.0, theta=1.0)
    low = soft_permission(phi=5.0, cci=5.0, theta=1.0)
    assert high == 1.0
    assert 0.0 <= low < high


def test_genesis_permission_evaluate_admits_within_bounds():
    gate = GenesisPermission(theta=1.0)
    x = np.zeros(2)
    lower = -np.ones(2)
    upper = np.ones(2)
    w = np.array([1.0, 1.0])
    result = gate.evaluate(x, lower, upper, w)
    assert result.admitted
    assert result.hard_permission
    assert result.soft_permission == 1.0


def test_genesis_permission_rejects_outside_bounds():
    gate = GenesisPermission(theta=1.0)
    x = np.array([5.0, 5.0])
    lower = -np.ones(2)
    upper = np.ones(2)
    w = np.array([1.0, 1.0])
    result = gate.evaluate(x, lower, upper, w)
    assert not result.admitted
    assert not result.hard_permission
    assert result.phi > 0


def test_adaptive_permission_tightens_theta_when_admissions_too_easy():
    gate = AdaptivePermission(theta=1.0, target_rate=0.2, adapt_rate=0.5, window=5)
    x = np.zeros(2)
    lower = -np.ones(2)
    upper = np.ones(2)
    w = np.array([1.0, 1.0])
    for _ in range(5):
        gate.evaluate(x, lower, upper, w)
    # Always admitted (rate=1.0) with a low target -> theta should shrink.
    assert gate.theta < 1.0


def test_adaptive_permission_relaxes_theta_when_admissions_too_rare():
    gate = AdaptivePermission(
        theta=1.0, target_rate=0.9, adapt_rate=0.5, window=5, theta_max=10.0
    )
    x = np.array([5.0, 5.0])
    lower = -np.ones(2)
    upper = np.ones(2)
    w = np.array([1.0, 1.0])
    for _ in range(5):
        gate.evaluate(x, lower, upper, w)
    # Never admitted (rate=0.0) with a high target -> theta should grow.
    assert gate.theta > 1.0


def test_adaptive_permission_admission_rate_tracks_history():
    gate = AdaptivePermission(theta=1.0, window=3)
    x = np.zeros(2)
    lower = -np.ones(2)
    upper = np.ones(2)
    w = np.array([1.0, 1.0])
    gate.evaluate(x, lower, upper, w)
    assert gate.admission_rate == 1.0


def test_adaptive_permission_theta_respects_min_and_max_bounds():
    gate = AdaptivePermission(
        theta=1.0, target_rate=0.0, adapt_rate=100.0, theta_min=0.5, theta_max=2.0
    )
    x = np.zeros(2)
    lower = -np.ones(2)
    upper = np.ones(2)
    w = np.array([1.0, 1.0])
    for _ in range(10):
        gate.evaluate(x, lower, upper, w)
    assert 0.5 <= gate.theta <= 2.0


def test_adaptive_permission_history_stays_bounded_to_window():
    gate = AdaptivePermission(theta=1.0, window=3)
    x = np.zeros(2)
    lower = -np.ones(2)
    upper = np.ones(2)
    w = np.array([1.0, 1.0])
    for _ in range(5):
        gate.evaluate(x, lower, upper, w)
    assert len(gate._history) == 3


def test_adaptive_permission_admission_rate_zero_before_any_evaluation():
    gate = AdaptivePermission(theta=1.0)
    assert gate.admission_rate == 0.0
