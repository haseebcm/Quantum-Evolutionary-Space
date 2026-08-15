"""ACROS orchestration law and room lifecycle (docs/QES-architecture.md, sections 25-26, 29-30).

ACROS correction law:
    xdot = f(x, t) - K * grad[ Phi(x) + mu * CCI(x) ]
    xdot_i = f_i(x_i, t) - chi_i * K_i * grad[ Phi_i + mu_i * CCI_i ]

ACROS V13 meta-adaptation:
    (A, w, Theta)* = argmin[ Risk + Inconsistency + Instability ]

Room lifecycle (section 29):
    State(R_i) in { Seed, Active, Shadow, Frozen, Collapsed, Merged, Validated }

QES master operator (section 30):
    Q_{t+dt} = C o S o P o D o E o G (Q_t, Y_{t+dt})
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

VALID_STATES = {
    "Seed",
    "Active",
    "Shadow",
    "Frozen",
    "Collapsed",
    "Merged",
    "Validated",
}

# Allowed lifecycle transitions (section 29 example paths).
_ALLOWED_TRANSITIONS = {
    "Seed": {"Active", "Collapsed"},
    "Active": {"Validated", "Shadow", "Frozen", "Collapsed", "Merged"},
    "Shadow": {"Collapsed", "Active", "Validated"},
    "Frozen": {"Active", "Collapsed"},
    "Collapsed": set(),
    "Merged": {"Active", "Validated"},
    "Validated": set(),
}


class RoomLifecycle:
    """Validates and applies room lifecycle state transitions."""

    @staticmethod
    def can_transition(current: str, target: str) -> bool:
        if current not in VALID_STATES or target not in VALID_STATES:
            raise ValueError(f"unknown lifecycle state(s): {current!r} -> {target!r}")
        return target in _ALLOWED_TRANSITIONS[current]

    @staticmethod
    def transition(room, target: str):
        """Move `room` to `target` state if the transition is allowed."""
        if not RoomLifecycle.can_transition(room.state, target):
            raise ValueError(f"illegal transition {room.state!r} -> {target!r}")
        room.state = target
        return room

    @staticmethod
    def merge(room_a, room_b, merged_state: str = "Merged"):
        """Active_A + Active_B -> Merged_C: combine two rooms into a child room."""
        child = room_a.clone(
            x=(room_a.x + room_b.x) / 2.0,
            lineage=room_a.lineage + [room_a.id] + room_b.lineage + [room_b.id],
            state=merged_state,
        )
        return child


class Acros:
    """ACROS: the self-stabilizing orchestration law applied inside each room."""

    def __init__(
        self,
        gradient_fn: Callable[[np.ndarray, dict], np.ndarray],
        gain: float = 1.0,
        mu: float = 1.0,
    ):
        """
        Args:
            gradient_fn: callable(x, lower_upper_context) -> grad[Phi + mu*CCI],
                approximating grad[Phi(x) + mu*CCI(x)] with respect to x.
            gain: correction gain K_i.
            mu: weight mu_i on the CCI term inside the gradient.
        """
        self.gradient_fn = gradient_fn
        self.gain = gain
        self.mu = mu

    def correction(self, x: np.ndarray, chi: float, context: dict | None = None) -> np.ndarray:
        """chi_i * K_i * grad[Phi_i + mu_i * CCI_i]."""
        grad = self.gradient_fn(x, context or {})
        return chi * self.gain * np.asarray(grad, dtype=float)

    def state_derivative(
        self,
        x: np.ndarray,
        t: float,
        drift_fn: Callable[[np.ndarray, float], np.ndarray],
        chi: float,
        context: dict | None = None,
    ) -> np.ndarray:
        """xdot_i = f_i(x_i, t) - chi_i * K_i * grad[Phi_i + mu_i * CCI_i]."""
        drift = np.asarray(drift_fn(x, t), dtype=float)
        return drift - self.correction(x, chi, context)

    @staticmethod
    def meta_adapt(
        candidates: list,
        risk_fn: Callable[[object], float],
        inconsistency_fn: Callable[[object], float],
        instability_fn: Callable[[object], float],
    ):
        """(A, w, Theta)* = argmin[Risk + Inconsistency + Instability] over candidates."""
        return min(
            candidates,
            key=lambda c: risk_fn(c) + inconsistency_fn(c) + instability_fn(c),
        )


class AdaptiveAcros(Acros):
    """Acros with an Adam-style adaptive gain: per-component first/second
    moment estimates of the correction gradient replace the fixed gain `K`.

    This is an "extreme" advancement of the base ACROS correction law: rather
    than a single scalar gain `K_i`, each state component gets its own
    effective step size that shrinks in high-curvature/noisy directions and
    grows in flat, consistent ones -- the same idea behind Adam in gradient
    based optimization, applied to the room correction law:

        m_t = b1*m_{t-1} + (1-b1)*g_t
        v_t = b2*v_{t-1} + (1-b2)*g_t^2
        mhat_t = m_t / (1-b1^t), vhat_t = v_t / (1-b2^t)
        correction_t = chi_i * K_i * mhat_t / (sqrt(vhat_t) + eps)
    """

    def __init__(
        self,
        gradient_fn: Callable[[np.ndarray, dict], np.ndarray],
        gain: float = 1.0,
        mu: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(gradient_fn, gain, mu)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._t: int = 0

    def reset(self) -> None:
        """Clear the accumulated moment estimates (e.g. when a room is reborn)."""
        self._m = None
        self._v = None
        self._t = 0

    def correction(self, x: np.ndarray, chi: float, context: dict | None = None) -> np.ndarray:
        """chi_i * K_i * mhat / (sqrt(vhat) + eps) -- Adam-adapted correction."""
        grad = np.asarray(self.gradient_fn(x, context or {}), dtype=float)
        m = self._m if self._m is not None else np.zeros_like(grad)
        v = self._v if self._v is not None else np.zeros_like(grad)
        self._t += 1
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
        self._m, self._v = m, v
        m_hat = m / (1.0 - self.beta1 ** self._t)
        v_hat = v / (1.0 - self.beta2 ** self._t)
        return chi * self.gain * m_hat / (np.sqrt(v_hat) + self.eps)
