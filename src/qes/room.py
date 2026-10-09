"""Canonical QES room definition (docs/QES-architecture.md, section 2).

A room needs more than a state vector — it carries its state, its laws
(active equations), its constraints, its allocated compute, and its history.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_id_counter = itertools.count(1)


def _next_id() -> str:
    return f"R-{next(_id_counter):05d}"


@dataclass
class Room:
    """R_i = (x_i, x_i*, L_i, U_i, a_i, W_i, E_i, theta_i, G_i, C_i, rho_i, M_i, l_i).

    Attributes:
        x: current virtual state vector.
        x_star: reference / equilibrium state vector.
        lower: lower bound of the allowed state envelope L_i.
        upper: upper bound of the allowed state envelope U_i.
        activation: domain activation/nullification vector a_i (values in [0, 1]).
        equations: active equation population E_i (list of Equation ids/objects).
        theta: equation/model parameters theta_i.
        gates: constraints/gates G_i (arbitrary metadata, e.g. thresholds).
        couplings: coupling structure C_i (matrix describing cross-variable coupling).
        compute: allocated compute resources rho_i (dict of resource -> amount).
        memory: memory M_i (arbitrary payload / history buffer).
        lineage: provenance chain l_i (ids of ancestor rooms).
        id: unique room identifier.
        state: lifecycle state (see orchestrator.RoomLifecycle).
        weight: belief / viability weight p_i used within a QESSpace.
    """

    x: np.ndarray
    x_star: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    activation: np.ndarray
    equations: list = field(default_factory=list)
    theta: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    couplings: np.ndarray | None = None
    compute: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    lineage: list = field(default_factory=list)
    id: str = field(default_factory=_next_id)
    state: str = "Seed"
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.x_star = np.asarray(self.x_star, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.activation = np.asarray(self.activation, dtype=float)
        n = self.x.shape[0]
        if self.couplings is None:
            self.couplings = np.eye(n)

    def validate(self) -> None:
        """Validate execution inputs while allowing invalid rooms for diagnostics."""
        arrays = (self.x, self.x_star, self.lower, self.upper, self.activation)
        if self.x.ndim != 1 or self.x.size == 0:
            raise ValueError("room state must be a nonempty one-dimensional vector")
        if any(value.shape != self.x.shape or not np.isfinite(value).all() for value in arrays):
            raise ValueError("room vectors must be finite and share the state shape")
        if (self.lower > self.upper).any():
            raise ValueError("room lower bounds must not exceed upper bounds")
        if ((self.activation < 0) | (self.activation > 1)).any():
            raise ValueError("room activation must lie in [0, 1]")
        if self.couplings is not None and (
            self.couplings.shape != (self.dim, self.dim) or not np.isfinite(self.couplings).all()
        ):
            raise ValueError("room coupling matrix must be finite and square with the state dimension")

    @property
    def dim(self) -> int:
        """Dimensionality n_i of this room's state space M_i = R^n_i."""
        return self.x.shape[0]

    @property
    def generation(self) -> int:
        """Branching depth: number of ancestors in `lineage` (0 for a seed room)."""
        return len(self.lineage)

    def tag(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata to this room's memory under a `tags` namespace."""
        self.memory.setdefault("tags", {})[key] = value

    def get_tag(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata previously stored via `tag()`."""
        return self.memory.get("tags", {}).get(key, default)

    def effective_state(self) -> np.ndarray:
        """x~_i = A_i * x_i, where A_i = diag(a_i)."""
        return self.activation * self.x

    def clone(self, **overrides: Any) -> Room:
        """Create a child room inheriting this room's fields, with overrides.

        The child's lineage is extended with this room's id.
        """
        fields: dict = dict(
            x=self.x.copy(),
            x_star=self.x_star.copy(),
            lower=self.lower.copy(),
            upper=self.upper.copy(),
            activation=self.activation.copy(),
            equations=copy.deepcopy(self.equations),
            theta=copy.deepcopy(self.theta),
            gates=copy.deepcopy(self.gates),
            couplings=None if self.couplings is None else self.couplings.copy(),
            compute=copy.deepcopy(self.compute),
            memory=copy.deepcopy(self.memory),
            lineage=self.lineage + [self.id],
            state="Seed",
            weight=self.weight,
        )
        fields.update(overrides)
        return Room(**fields)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Room(id={self.id!r}, state={self.state!r}, dim={self.dim}, weight={self.weight:.4g})"
