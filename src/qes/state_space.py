"""MCC — Multi-Component Coordinate state space (docs/QES-architecture.md, section 3).

The neutral state-space foundation used before DSA, DR, and HSA:

    x(t) = [x_1(t), ..., x_n(t)]^T,   x(t) in M = R^n
    L <= x(t) <= U

Inside QES, every virtual reality owns its own space M_i = { x_i in R^(n_i) },
and rooms need not share the same dimensionality.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class MCCStateSpace:
    """Bounded coordinate space for a single room's state vector."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        reference: np.ndarray,
        names: Sequence[str] | None = None,
    ):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.reference = np.asarray(reference, dtype=float)
        if not (self.lower.shape == self.upper.shape == self.reference.shape):
            raise ValueError("lower, upper, and reference must share the same shape")
        if self.lower.ndim != 1 or self.lower.size == 0 or not all(
            np.all(np.isfinite(value)) for value in (self.lower, self.upper, self.reference)
        ):
            raise ValueError("state-space vectors must be finite and one-dimensional")
        if np.any(self.lower > self.upper):
            raise ValueError("lower bounds must not exceed upper bounds")
        if names is not None and len(names) != self.lower.shape[0]:
            raise ValueError("names must have one entry per state dimension")
        self.names: list[str] | None = list(names) if names is not None else None

    @property
    def dim(self) -> int:
        return self.lower.shape[0]

    def within_envelope(self, x: np.ndarray) -> np.ndarray:
        """Elementwise boolean mask: L_j <= x_j <= U_j."""
        x = np.asarray(x, dtype=float)
        if x.shape != self.lower.shape or not np.all(np.isfinite(x)):
            raise ValueError("state must be finite and match the state-space shape")
        return (x >= self.lower) & (x <= self.upper)

    def is_admissible(self, x: np.ndarray) -> bool:
        """True iff x lies entirely within [L, U] (the raw domain Omega)."""
        return bool(np.all(self.within_envelope(x)))

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Project x back into [L, U]."""
        x = np.asarray(x, dtype=float)
        self.within_envelope(x)
        return np.clip(x, self.lower, self.upper)

    def volume(self) -> float:
        """Lebesgue measure of the bounding box Omega = prod_j (U_j - L_j)."""
        return float(np.prod(self.upper - self.lower))

    def index_of(self, name: str) -> int:
        """Look up the dimension index for a named coordinate (requires `names`)."""
        if self.names is None:
            raise ValueError("this state space has no named dimensions")
        return self.names.index(name)

    def labeled(self, x: np.ndarray) -> dict:
        """Return `x` as a name -> value dict (requires `names`)."""
        if self.names is None:
            raise ValueError("this state space has no named dimensions")
        return dict(zip(self.names, np.asarray(x, dtype=float).tolist(), strict=True))
