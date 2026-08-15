"""Reality generation operator (docs/QES-architecture.md, section 17).

    B(R_i) = { R_i1, R_i2, ..., R_im }
    R_ij   = G(R_i, zeta_ij, E_ij, a_ij)

One virtual reality can generate thousands of children, and each child can
itself branch further (R -> R_i -> R_ij -> R_ijk), forming a compute tree
rather than a flat simulation pool.

Two branching strategies are provided:

- `branch()`: i.i.d. Gaussian perturbation zeta_ij (or a custom
  `perturb_fn`) -- fast, unstructured Monte Carlo sampling of nearby
  realities.
- `branch_latin_hypercube()`: stratified Latin Hypercube sampling across a
  bounding box -- guarantees even coverage of every state dimension across
  the generated children, which Monte Carlo sampling cannot guarantee for
  small `count`, at the cost of needing explicit per-dimension bounds.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from qes.room import Room


class RealityGenerator:
    """Branches a parent room into a population of child realities."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def perturb(self, room: Room, scale: float = 0.05) -> np.ndarray:
        """zeta_ij: a default Gaussian perturbation/scenario vector for the state."""
        return self.rng.normal(0.0, scale, size=room.dim)

    def branch(
        self,
        room: Room,
        count: int,
        scale: float = 0.05,
        equations: Sequence | None = None,
        activations: Sequence[np.ndarray] | None = None,
        perturb_fn: Callable[[Room], np.ndarray] | None = None,
    ) -> list:
        """B(R_i) = { R_i1, ..., R_im }.

        Args:
            room: parent room R_i.
            count: number of children m to generate.
            scale: perturbation scale used by the default perturb function.
            equations: optional list of equation populations E_ij, one per child
                (cycled if shorter than count).
            activations: optional list of domain activation vectors a_ij, one per
                child (cycled if shorter than count).
            perturb_fn: optional custom zeta_ij generator, called as fn(room).
        """
        children = []
        for j in range(count):
            zeta = (
                perturb_fn(room)
                if perturb_fn is not None
                else self.perturb(room, scale)
            )
            overrides: dict = {"x": room.x + zeta}
            if equations:
                overrides["equations"] = list(equations[j % len(equations)])
            if activations:
                overrides["activation"] = np.asarray(activations[j % len(activations)])
            child = room.clone(**overrides)
            child.state = "Active"
            children.append(child)
        return children

    def latin_hypercube_samples(
        self, dim: int, count: int, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """Stratified Latin Hypercube samples of shape (count, dim) within [lower, upper].

        Each of the `dim` axes is split into `count` equal strata; one
        sample is drawn uniformly within a random stratum per axis, and the
        stratum-to-sample assignment is independently shuffled per axis, so
        every stratum on every axis is populated exactly once -- unlike
        i.i.d. sampling, which can leave gaps for small `count`.
        """
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        samples = np.empty((count, dim), dtype=float)
        strata_edges = np.linspace(0.0, 1.0, count + 1)
        for d in range(dim):
            offsets = self.rng.uniform(0.0, 1.0, size=count)
            points = strata_edges[:-1] + offsets * (strata_edges[1:] - strata_edges[:-1])
            self.rng.shuffle(points)
            samples[:, d] = lower[d] + points * (upper[d] - lower[d])
        return samples

    def branch_latin_hypercube(
        self,
        room: Room,
        count: int,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> list:
        """Branch into `count` children whose states are Latin-Hypercube-sampled
        across [`lower`, `upper`] (defaults to the room's own bounds), guaranteeing
        stratified coverage of the room's state envelope rather than i.i.d. noise."""
        lower = room.lower if lower is None else np.asarray(lower, dtype=float)
        upper = room.upper if upper is None else np.asarray(upper, dtype=float)
        samples = self.latin_hypercube_samples(room.dim, count, lower, upper)
        children = []
        for j in range(count):
            child = room.clone(x=samples[j])
            child.state = "Active"
            children.append(child)
        return children
