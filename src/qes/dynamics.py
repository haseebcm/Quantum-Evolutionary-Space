"""QES dynamics (docs/QES-architecture.md, section 5).

Within each room, discrete-time evolution:

    x_i(t + dt) = F_{E_i}(x_i(t), u_i(t), xi_i(t), theta_i)

Continuous form:

    xdot_i = f_i(x_i, t) + B_i * u_i

A QES room is not a static container — it is an evolving dynamical system.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

# F(x, u, xi, theta, t) -> x_next
TransitionFn = Callable[[np.ndarray, np.ndarray, np.ndarray, dict, float], np.ndarray]
# f(x, t) -> xdot contribution
DriftFn = Callable[[np.ndarray, float], np.ndarray]


class RoomDynamics:
    """Wraps a room's governing transition function F_{E_i}."""

    def __init__(
        self,
        transition: TransitionFn | None = None,
        drift: DriftFn | None = None,
        control_matrix: np.ndarray | None = None,
    ):
        self.transition = transition
        self.drift = drift
        self.control_matrix = control_matrix

    def step_discrete(
        self,
        x: np.ndarray,
        u: np.ndarray | None = None,
        xi: np.ndarray | None = None,
        theta: dict | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """x_i(t + dt) = F_{E_i}(x_i(t), u_i(t), xi_i(t), theta_i)."""
        n = x.shape[0]
        u = np.zeros(n) if u is None else np.asarray(u, dtype=float)
        xi = np.zeros(n) if xi is None else np.asarray(xi, dtype=float)
        theta = theta or {}
        if self.transition is not None:
            return np.asarray(self.transition(x, u, xi, theta, t), dtype=float)
        # Default transition: additive drift + control + disturbance.
        drift = self.drift(x, t) if self.drift is not None else np.zeros(n)
        b = self.control_matrix if self.control_matrix is not None else np.eye(n)
        return x + drift + b @ u + xi

    def step_continuous(
        self, x: np.ndarray, u: np.ndarray | None = None, t: float = 0.0
    ) -> np.ndarray:
        """xdot_i = f_i(x_i, t) + B_i * u_i."""
        n = x.shape[0]
        u = np.zeros(n) if u is None else np.asarray(u, dtype=float)
        drift = self.drift(x, t) if self.drift is not None else np.zeros(n)
        b = self.control_matrix if self.control_matrix is not None else np.eye(n)
        return drift + b @ u

    def integrate(
        self, x: np.ndarray, u: np.ndarray | None, t: float, dt: float
    ) -> np.ndarray:
        """Simple forward-Euler integration of the continuous form."""
        return x + dt * self.step_continuous(x, u, t)

    def integrate_rk4(
        self, x: np.ndarray, u: np.ndarray | None, t: float, dt: float
    ) -> np.ndarray:
        """Classical 4th-order Runge-Kutta integration of xdot = f(x,t) + B*u.

        Higher-order alternative to `integrate()` (forward Euler): RK4's
        local truncation error is O(dt^5) vs. Euler's O(dt^2), so rooms with
        fast/stiff dynamics stay numerically stable at much larger step
        sizes `dt`, at the cost of four `step_continuous` evaluations per
        tick instead of one.
        """
        k1 = self.step_continuous(x, u, t)
        k2 = self.step_continuous(x + 0.5 * dt * k1, u, t + 0.5 * dt)
        k3 = self.step_continuous(x + 0.5 * dt * k2, u, t + 0.5 * dt)
        k4 = self.step_continuous(x + dt * k3, u, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def stochastic_noise(
        dim: int,
        sigma: float | np.ndarray = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw a Gaussian disturbance xi ~ N(0, sigma^2 I) for the `xi` term of
        `step_discrete`, modelling process noise / stochastic dynamics."""
        rng = rng or np.random.default_rng()
        return rng.normal(0.0, 1.0, size=dim) * np.asarray(sigma, dtype=float)

    def step_stochastic(
        self,
        x: np.ndarray,
        t: float,
        dt: float,
        sigma: float | np.ndarray = 0.0,
        u: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """RK4-integrate the deterministic drift, then add a stochastic
        disturbance scaled by sqrt(dt) (Euler-Maruyama noise scaling) --
        combines a high-order deterministic integrator with process noise
        in one call."""
        deterministic = self.integrate_rk4(x, u, t, dt)
        noise = self.stochastic_noise(x.shape[0], sigma, rng) * np.sqrt(dt)
        return deterministic + noise
