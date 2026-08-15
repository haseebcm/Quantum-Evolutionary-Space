import numpy as np

from qes.dynamics import RoomDynamics


def test_step_discrete_default_transition_applies_drift_and_control():
    dyn = RoomDynamics(drift=lambda x, t: np.array([1.0, 1.0]))
    x = np.zeros(2)
    u = np.array([0.5, 0.5])
    result = dyn.step_discrete(x, u=u)
    np.testing.assert_allclose(result, [1.5, 1.5])


def test_step_discrete_uses_custom_transition_when_provided():
    def transition(x, u, xi, theta, t):
        return x * 2

    dyn = RoomDynamics(transition=transition)
    result = dyn.step_discrete(np.array([1.0, 2.0]))
    np.testing.assert_allclose(result, [2.0, 4.0])


def test_step_discrete_adds_disturbance():
    dyn = RoomDynamics()
    x = np.zeros(2)
    xi = np.array([0.1, -0.1])
    result = dyn.step_discrete(x, xi=xi)
    np.testing.assert_allclose(result, xi)


def test_step_continuous_matches_drift_plus_control():
    dyn = RoomDynamics(drift=lambda x, t: np.array([2.0]), control_matrix=np.array([[3.0]]))
    xdot = dyn.step_continuous(np.zeros(1), u=np.array([1.0]))
    np.testing.assert_allclose(xdot, [5.0])


def test_integrate_advances_state_by_euler_step():
    dyn = RoomDynamics(drift=lambda x, t: np.array([1.0]))
    x_next = dyn.integrate(np.zeros(1), u=None, t=0.0, dt=0.5)
    np.testing.assert_allclose(x_next, [0.5])


def test_integrate_rk4_matches_analytic_solution_of_linear_decay():
    # xdot = -x (drift only), exact solution x(t) = x0 * exp(-t).
    dyn = RoomDynamics(drift=lambda x, t: -x)
    x_next = dyn.integrate_rk4(np.array([1.0]), u=None, t=0.0, dt=0.1)
    np.testing.assert_allclose(x_next, [np.exp(-0.1)], atol=1e-5)


def test_integrate_rk4_more_accurate_than_euler_for_larger_steps():
    dyn = RoomDynamics(drift=lambda x, t: -x)
    exact = np.exp(-1.0)
    euler = dyn.integrate(np.array([1.0]), u=None, t=0.0, dt=1.0)[0]
    rk4 = dyn.integrate_rk4(np.array([1.0]), u=None, t=0.0, dt=1.0)[0]
    assert abs(rk4 - exact) < abs(euler - exact)


def test_stochastic_noise_has_requested_dimension_and_scale():
    rng = np.random.default_rng(0)
    noise = RoomDynamics.stochastic_noise(dim=1000, sigma=2.0, rng=rng)
    assert noise.shape == (1000,)
    assert 1.5 < np.std(noise) < 2.5


def test_step_stochastic_combines_rk4_drift_and_noise():
    dyn = RoomDynamics(drift=lambda x, t: -x)
    rng = np.random.default_rng(0)
    deterministic = dyn.integrate_rk4(np.array([1.0]), u=None, t=0.0, dt=0.1)
    result = dyn.step_stochastic(np.array([1.0]), t=0.0, dt=0.1, sigma=0.0, rng=rng)
    # With sigma=0 the noise term vanishes, so this should match the deterministic RK4 step.
    np.testing.assert_allclose(result, deterministic)
