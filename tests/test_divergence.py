import numpy as np

from qes.divergence import DSA, HSA_CRITICAL, HSA_SINGULAR, HSA_STABLE, hsa_state


def test_divergence_zero_when_at_reference():
    dsa = DSA()
    result = dsa.update(
        x=np.zeros(2), x_star=np.zeros(2), dt=1.0, w=np.eye(2)
    )
    assert result.dr == 0.0
    np.testing.assert_allclose(result.d, [0.0, 0.0])


def test_dr_uses_domain_nullified_metric():
    dsa = DSA()
    x = np.array([1.0, 2.0])
    x_star = np.zeros(2)
    w = np.diag([1.0, 0.5])
    result = dsa.update(x, x_star, dt=1.0, w=w)
    # DR = d^T W d = 1*1 + 2*2*0.5 = 1 + 2 = 3
    assert result.dr == 3.0


def test_velocity_and_acceleration_are_finite_differenced():
    dsa = DSA()
    dsa.update(x=np.array([0.0]), x_star=np.array([0.0]), dt=1.0, w=np.eye(1))
    r2 = dsa.update(x=np.array([1.0]), x_star=np.array([0.0]), dt=1.0, w=np.eye(1))
    assert r2.d_dot[0] == 1.0
    r3 = dsa.update(x=np.array([3.0]), x_star=np.array([0.0]), dt=1.0, w=np.eye(1))
    # d_dot goes from 1.0 -> 2.0, so acceleration = 1.0
    assert r3.d_dot[0] == 2.0
    assert r3.d_ddot[0] == 1.0


def test_health_uses_scale_and_baseline():
    dsa = DSA()
    result = dsa.update(
        x=np.array([2.0]), x_star=np.array([0.0]), dt=1.0, w=np.eye(1), k=2.0, baseline=1.0
    )
    # DR = 4, health = k*DR - baseline = 2*4 - 1 = 7
    assert result.health == 7.0


def test_hsa_state_classifies_stable_critical_singular():
    assert hsa_state(-1.0) == HSA_STABLE
    assert hsa_state(0.0) == HSA_CRITICAL
    assert hsa_state(1.0) == HSA_SINGULAR


def test_hsa_state_respects_critical_band_tolerance():
    assert hsa_state(1e-9, critical_band=1e-6) == HSA_CRITICAL
    assert hsa_state(1e-3, critical_band=1e-6) == HSA_SINGULAR


def test_divergence_result_state_method_matches_hsa_state():
    dsa = DSA()
    result = dsa.update(x=np.array([2.0]), x_star=np.zeros(1), dt=1.0, w=np.eye(1))
    assert result.state() == hsa_state(result.health)


def test_dr_accepts_1d_weight_vector():
    dsa = DSA()
    x = np.array([1.0, 2.0])
    x_star = np.zeros(2)
    w = np.array([1.0, 0.5])
    result = dsa.update(x, x_star, dt=1.0, w=w)
    # DR_w = sum_j w_j * d_j^2 = 1*1 + 0.5*4 = 3
    assert result.dr == 3.0


def test_dr_uses_standardized_form_when_sigma_given():
    dsa = DSA()
    x = np.array([2.0])
    x_star = np.array([0.0])
    result = dsa.update(x, x_star, dt=1.0, w=np.eye(1), sigma=np.array([2.0]))
    # z = d / sigma = 1.0, DR = z^T W z = 1.0
    assert result.dr == 1.0


def test_reset_clears_previous_divergence_history():
    dsa = DSA()
    dsa.update(x=np.array([1.0]), x_star=np.zeros(1), dt=1.0, w=np.eye(1))
    dsa.reset()
    assert dsa._prev_d is None
    assert dsa._prev_d_dot is None
    # Velocity should restart from zero as if freshly created.
    result = dsa.update(x=np.array([5.0]), x_star=np.zeros(1), dt=1.0, w=np.eye(1))
    assert result.d_dot[0] == 0.0
