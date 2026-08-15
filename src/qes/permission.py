"""Genesis permission kernel (docs/QES-architecture.md, sections 9-14).

    Omega       = { x : l <= x <= u }
    Phi(x)      = sum_j [ max(0, x_j - u_j)^2 + max(0, l_j - x_j)^2 ]     (violation energy)
    P           = { x in Omega : Phi(x) = 0  AND  CCI(x) < Theta }        (Genesis-permitted region)

    Pi(R_i)     = 1 if x_i in P_i else 0                                  (hard permission operator)
    pi(x)       = exp(-alpha * Phi(x)) * exp(-beta * max(0, CCI(x) - Theta))  (soft permission field)

    CCI         = w^T e + gamma * ||A e||^2                              (Cascade Collapse Index)
    CCI_up      = CCI + eta * sum_j w_j * max(0, r_j),  r_j = de_j/dt     (accelerated CCI)

    M(t)        = min_j( (x_j-l_j)/(u_j-l_j), (u_j-x_j)/(u_j-l_j) ) * 1/(1+CCI(t))  (permission margin)

    chi_i       = 1[x_i in Omega_i] * 1[Phi_i <= eps_Phi] * 1[CCI_i < Theta_i] * 1[M_i >= M_min]
                                                                           (complete admission kernel)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def violation_energy(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Phi(x) = sum_j [ max(0, x_j - u_j)^2 + max(0, l_j - x_j)^2 ]."""
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    over = np.maximum(0.0, x - upper)
    under = np.maximum(0.0, lower - x)
    return float(np.sum(over ** 2 + under ** 2))


def exceedance(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Per-component exceedance e_j = max(0, x_j-u_j) + max(0, l_j-x_j) used by CCI."""
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.maximum(0.0, x - upper) + np.maximum(0.0, lower - x)


def cascade_collapse_index(
    e: np.ndarray,
    w: np.ndarray,
    coupling: np.ndarray | None = None,
    gamma: float = 1.0,
) -> float:
    """CCI = w^T e + gamma * ||A e||^2."""
    e = np.asarray(e, dtype=float)
    w = np.asarray(w, dtype=float)
    base = float(w @ e)
    if coupling is None:
        return base
    coupled = coupling @ e
    return base + gamma * float(np.dot(coupled, coupled))


def accelerated_cci(
    cci: float, w: np.ndarray, e_rate: np.ndarray, eta: float = 1.0
) -> float:
    """CCI_up = CCI + eta * sum_j w_j * max(0, r_j), r_j = de_j/dt."""
    w = np.asarray(w, dtype=float)
    e_rate = np.asarray(e_rate, dtype=float)
    return cci + eta * float(np.sum(w * np.maximum(0.0, e_rate)))


def permission_margin(
    x: np.ndarray, lower: np.ndarray, upper: np.ndarray, cci: float
) -> float:
    """M(t) = min_j( (x_j-l_j)/(u_j-l_j), (u_j-x_j)/(u_j-l_j) ) / (1 + CCI(t))."""
    x = np.asarray(x, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    span = upper - lower
    span = np.where(span == 0, np.finfo(float).eps, span)
    lower_margin = (x - lower) / span
    upper_margin = (upper - x) / span
    m = float(np.min(np.minimum(lower_margin, upper_margin)))
    return m / (1.0 + cci)


def soft_permission(phi: float, cci: float, theta: float, alpha: float = 1.0, beta: float = 1.0) -> float:
    """pi(x) = exp(-alpha*Phi) * exp(-beta*max(0, CCI-Theta))."""
    return float(np.exp(-alpha * phi) * np.exp(-beta * max(0.0, cci - theta)))


@dataclass
class PermissionResult:
    """Bundled result of evaluating the full permission kernel for one room."""

    phi: float
    cci: float
    margin: float
    hard_permission: bool  # Pi_i
    soft_permission: float  # pi_i
    admitted: bool  # chi_i


class GenesisPermission:
    """The Genesis permission gate: evaluates Pi, pi, CCI, M, and chi for a room."""

    def __init__(
        self,
        theta: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        eps_phi: float = 1e-9,
        m_min: float = 0.0,
    ):
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps_phi = eps_phi
        self.m_min = m_min

    def evaluate(
        self,
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        w: np.ndarray | None = None,
        coupling: np.ndarray | None = None,
    ) -> PermissionResult:
        """Evaluate the full permission kernel."""
        x_arr = np.asarray(x, dtype=float)
        l_arr = np.asarray(lower, dtype=float)
        u_arr = np.asarray(upper, dtype=float)

        over = np.maximum(0.0, x_arr - u_arr)
        under = np.maximum(0.0, l_arr - x_arr)
        phi = float(np.sum(over ** 2 + under ** 2))

        e = over + under
        if coupling is None:
            base = float(np.sum(e)) if w is None else float(np.dot(np.asarray(w, dtype=float), e))
            cci = base
        else:
            w_val = np.ones_like(e) if w is None else np.asarray(w, dtype=float)
            base = float(np.dot(w_val, e))
            coupled = coupling @ e
            cci = base + self.gamma * float(np.dot(coupled, coupled))

        span = u_arr - l_arr
        span = np.where(span == 0, np.finfo(float).eps, span)
        lower_margin = (x_arr - l_arr) / span
        upper_margin = (u_arr - x_arr) / span
        margin_min = float(np.min(np.minimum(lower_margin, upper_margin)))
        margin = margin_min / (1.0 + cci)

        hard = (phi == 0.0) and (cci < self.theta)
        soft = float(np.exp(-self.alpha * phi) * np.exp(-self.beta * max(0.0, cci - self.theta)))
        admitted = (
            phi <= self.eps_phi
            and cci < self.theta
            and margin >= self.m_min
        )
        return PermissionResult(
            phi=phi,
            cci=cci,
            margin=margin,
            hard_permission=hard,
            soft_permission=soft,
            admitted=admitted,
        )


class AdaptivePermission(GenesisPermission):
    """A Genesis permission gate whose admission threshold Theta adapts over
    time based on the recent admission history.

    This is an "extreme" advancement over the fixed-Theta base gate: if
    admissions are consistently easy (admission rate above `target_rate`),
    Theta tightens (shrinks) to raise the bar; if admissions are too rare,
    Theta relaxes (grows) up to `theta_max`, subject to a floor of
    `theta_min`. The adaptation itself is governed -- bounded and rate
    limited by `adapt_rate` -- so the gate cannot runaway to always-admit or
    always-reject.
    """

    def __init__(
        self,
        theta: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        eps_phi: float = 1e-9,
        m_min: float = 0.0,
        target_rate: float = 0.5,
        adapt_rate: float = 0.05,
        theta_min: float = 1e-6,
        theta_max: float | None = None,
        window: int = 20,
    ):
        super().__init__(theta, alpha, beta, gamma, eps_phi, m_min)
        self.target_rate = target_rate
        self.adapt_rate = adapt_rate
        self.theta_min = theta_min
        self.theta_max = theta_max if theta_max is not None else theta * 10.0
        self.window = window
        self._history: list = []

    def evaluate(
        self,
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        w: np.ndarray | None = None,
        coupling: np.ndarray | None = None,
    ) -> PermissionResult:
        """Evaluate the gate at the current Theta, then adapt Theta for next time."""
        result = super().evaluate(x, lower, upper, w, coupling)
        self._history.append(result.admitted)
        if len(self._history) > self.window:
            del self._history[0]
        rate = sum(self._history) / len(self._history)
        # Admissions too easy (rate above target) -> tighten (shrink) Theta.
        # Admissions too hard (rate below target) -> relax (grow) Theta.
        adjustment = 1.0 - self.adapt_rate * (rate - self.target_rate)
        self.theta = float(np.clip(self.theta * adjustment, self.theta_min, self.theta_max))
        return result

    @property
    def admission_rate(self) -> float:
        """Recent admission rate over the trailing `window` evaluations."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)
