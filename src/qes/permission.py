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
        parameters = (theta, alpha, beta, gamma, eps_phi, m_min)
        if not all(np.isfinite(value) for value in parameters):
            raise ValueError("permission parameters must be finite")
        if theta < 0 or any(value < 0 for value in (alpha, beta, gamma, eps_phi)):
            raise ValueError("permission threshold and penalties must be nonnegative")
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps_phi = eps_phi
        self.m_min = m_min

    def evaluate_batch(self, candidates: list[tuple]) -> list[PermissionResult]:
        """Evaluate candidates using the gate's documented population policy."""
        # Preserve custom gate overrides. Small or heterogeneous populations use
        # the scalar path; batching low-dimensional populations avoids repeated
        # NumPy dispatch without weakening validation at the public boundary.
        if len(candidates) >= 8 and type(self).evaluate is GenesisPermission.evaluate and all(
            3 <= len(candidate) <= 5 for candidate in candidates
        ):
            dimensions = [np.asarray(candidate[0]).shape for candidate in candidates]
            if all(shape == dimensions[0] for shape in dimensions) and (
                len(dimensions[0]) == 1 and 0 < dimensions[0][0] <= 16
            ):
                return self._evaluate_population([
                    tuple(candidate) + (None,) * (5 - len(candidate)) for candidate in candidates
                ])
        return [self.evaluate(*candidate) for candidate in candidates]

    def _evaluate_population(self, candidates: list[tuple]) -> list[PermissionResult]:
        x = np.stack([np.asarray(c[0], dtype=float) for c in candidates])
        lower = np.stack([np.asarray(c[1], dtype=float) for c in candidates])
        upper = np.stack([np.asarray(c[2], dtype=float) for c in candidates])
        if lower.shape != x.shape or upper.shape != x.shape:
            raise ValueError("permission vectors must share a one-dimensional shape")
        if not (np.isfinite(x).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise ValueError("permission state and bounds must be finite")
        if (lower > upper).any():
            raise ValueError("permission bounds require lower <= upper")
        weights = np.stack([np.ones(x.shape[1]) if c[3] is None else np.asarray(c[3], dtype=float)
                            for c in candidates])
        if weights.shape != x.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("CCI weights must be finite, nonnegative and match the state")
        coupling = np.stack([np.zeros((x.shape[1], x.shape[1])) if c[4] is None
                             else np.asarray(c[4], dtype=float) for c in candidates])
        if coupling.shape != (len(candidates), x.shape[1], x.shape[1]) or not np.isfinite(coupling).all():
            raise ValueError("coupling must be a finite square matrix matching the state")
        over = np.maximum(0.0, x - upper)
        under = np.maximum(0.0, lower - x)
        exceed = over + under
        phi = (over ** 2 + under ** 2).sum(axis=1)
        coupled = np.matmul(coupling, exceed[..., None])[..., 0]
        cci = (weights * exceed).sum(axis=1) + self.gamma * (coupled ** 2).sum(axis=1)
        span = upper - lower
        span = np.where(span == 0, np.finfo(float).eps, span)
        margin = np.minimum((x - lower) / span, (upper - x) / span).min(axis=1) / (1 + cci)
        hard = (phi == 0) & (cci < self.theta)
        soft = np.exp(-self.alpha * phi) * np.exp(-self.beta * np.maximum(0, cci - self.theta))
        admitted = ((x >= lower) & (x <= upper)).all(axis=1) & (phi <= self.eps_phi) & (
            cci < self.theta) & (margin >= self.m_min)
        return [PermissionResult(float(phi[i]), float(cci[i]), float(margin[i]),
                                 bool(hard[i]), float(soft[i]), bool(admitted[i]))
                for i in range(len(candidates))]

    def inspect(
        self, x: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        w: np.ndarray | None = None, coupling: np.ndarray | None = None,
    ) -> PermissionResult:
        """Validate a candidate without adapting the gate or changing its history."""
        return GenesisPermission.evaluate(self, x, lower, upper, w, coupling)

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

        if x_arr.ndim != 1 or x_arr.size == 0 or l_arr.shape != x_arr.shape or u_arr.shape != x_arr.shape:
            raise ValueError("permission vectors must be nonempty and share a one-dimensional shape")
        if not all(np.all(np.isfinite(value)) for value in (x_arr, l_arr, u_arr)) or np.any(l_arr > u_arr):
            raise ValueError("permission state and bounds must be finite with lower <= upper")
        if w is not None and (np.asarray(w).shape != x_arr.shape or not np.all(np.isfinite(w)) or np.any(np.asarray(w) < 0)):
            raise ValueError("CCI weights must be finite, nonnegative and match the state")
        if coupling is not None and (
            np.asarray(coupling).shape != (x_arr.size, x_arr.size) or not np.all(np.isfinite(coupling))
        ):
            raise ValueError("coupling must be a finite square matrix matching the state")
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
            bool(np.all((x_arr >= l_arr) & (x_arr <= u_arr)))
            and phi <= self.eps_phi
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
        if not np.isfinite(target_rate) or not 0 <= target_rate <= 1:
            raise ValueError("target_rate must be in [0,1]")
        if not np.isfinite(adapt_rate) or adapt_rate < 0:
            raise ValueError("adapt_rate must be finite and nonnegative")
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError("window must be a positive integer")
        if not np.isfinite(theta_min) or theta_min <= 0:
            raise ValueError("theta_min must be finite and positive")
        if theta_max is not None and (not np.isfinite(theta_max) or theta_max < theta_min):
            raise ValueError("theta_max must be finite and >= theta_min")
        self.target_rate = target_rate
        self.adapt_rate = adapt_rate
        self.theta_min = theta_min
        self.theta_max = theta_max if theta_max is not None else theta * 10.0
        self.window = window
        self._history: list = []

    def evaluate_batch(self, candidates: list[tuple]) -> list[PermissionResult]:
        """Use one threshold for the population, then adapt once on its admission rate."""
        results = [self.inspect(*candidate) for candidate in candidates]
        if results:
            self._history.append(sum(result.admitted for result in results) / len(results))
            self._adapt()
        return results

    def _adapt(self) -> None:
        if len(self._history) > self.window:
            del self._history[0]
        rate = sum(self._history) / len(self._history)
        adjustment = 1.0 - self.adapt_rate * (rate - self.target_rate)
        self.theta = float(np.clip(self.theta * adjustment, self.theta_min, self.theta_max))

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
        self._adapt()
        return result

    @property
    def admission_rate(self) -> float:
        """Recent admission rate over the trailing `window` evaluations."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)
