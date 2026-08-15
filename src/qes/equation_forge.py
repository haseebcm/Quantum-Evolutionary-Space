"""SGEE Equation Forge (docs/QES-architecture.md, sections 15-16).

    D_n   = { V, C, I }                     (captured domain: variables, constraints, interactions)
    Lambda = f(C)
    Phi_I  = sum_{r,s} V_r (x) V_s * I_rs
    eps_0  = Lambda * Phi_I - Delta          (seed equation)

    E_j'          = M_E(E_j, dtheta, dD, dC)  (equation mutation)
    Parent(E_j')  = E_j                       (lineage tracked)

Each room owns a population E_i = { E_i1, ..., E_im } of equation hypotheses;
QES therefore branches along two dimensions: reality branching and equation
branching.

Beyond asexual mutation, `EquationForge.crossover()` recombines two parent
equations' parameters (sexual recombination across lineages), and
`spawn_next_generation()` combines elitism + crossover + mutation into a
single full generational-replacement step, turning the equation population
into a genuine evolutionary search rather than a mutate-only random walk.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_eq_id_counter = itertools.count(1)


def _next_eq_id() -> str:
    return f"E-{next(_eq_id_counter):05d}"


@dataclass
class Equation:
    """A single equation hypothesis governing a room's evolution.

    Attributes:
        theta: parameters of this equation.
        fitness: viability/quality score (higher is better); updated externally
            as the equation is tested (e.g. by suppression/selection).
        parent: id of the equation this one mutated from (None for seeds).
        generation: mutation depth from the original seed (0 for seeds).
        fn: optional callable implementing the equation's transition/residual.
    """

    theta: dict = field(default_factory=dict)
    fitness: float = 0.0
    parent: str | None = None
    generation: int = 0
    fn: Callable[..., Any] | None = None
    id: str = field(default_factory=_next_eq_id)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Equation(id={self.id!r}, gen={self.generation}, fitness={self.fitness:.4g})"


def seed_equation(
    variables: np.ndarray, interactions: np.ndarray, delta: float = 0.0, lam: float = 1.0
) -> float:
    """eps_0 = Lambda * Phi_I - Delta, with Phi_I = sum_rs V_r V_s I_rs."""
    v = np.asarray(variables, dtype=float)
    i_mat = np.asarray(interactions, dtype=float)
    phi_i = float(v @ i_mat @ v)
    return lam * phi_i - delta


class EquationForge:
    """SGEE: generates seed equations and recursively mutates equation populations."""

    def __init__(self, mutation_rate: float = 0.1, rng: np.random.Generator | None = None):
        self.mutation_rate = mutation_rate
        self.rng = rng or np.random.default_rng()

    def seed(self, theta: dict | None = None, fn: Callable | None = None) -> Equation:
        """Create a generation-0 seed equation E_i(0)."""
        return Equation(theta=dict(theta or {}), generation=0, fn=fn)

    def mutate(self, equation: Equation, scale: float = 1.0) -> Equation:
        """E_j' = M_E(E_j, dtheta, dD, dC): perturb numeric parameters, retain lineage."""
        mutated_theta = {}
        for key, value in equation.theta.items():
            if isinstance(value, (int, float)):
                noise = self.rng.normal(0.0, self.mutation_rate * scale)
                mutated_theta[key] = value + noise
            else:
                mutated_theta[key] = value
        return Equation(
            theta=mutated_theta,
            fitness=0.0,
            parent=equation.id,
            generation=equation.generation + 1,
            fn=equation.fn,
        )

    def spawn_population(self, base: Equation, size: int) -> list:
        """Generate m mutated descendants: E_i = { E_i1, ..., E_im }."""
        return [self.mutate(base) for _ in range(size)]

    def crossover(self, parent_a: Equation, parent_b: Equation) -> Equation:
        """Recombine two equations' numeric parameters into one child.

        For each shared numeric key in theta, the child's value is drawn
        uniformly between the two parents' values (a form of genetic
        recombination/BLX crossover); non-numeric or non-shared keys are
        inherited from `parent_a`. This complements `mutate()` (asexual
        perturbation) with sexual recombination across two lineages, and
        tracks both parents via `Equation.parent` (dominant lineage) plus
        `theta["_co_parent"]` (the secondary parent's id).
        """
        child_theta: dict = dict(parent_a.theta)
        for key, value_a in parent_a.theta.items():
            value_b = parent_b.theta.get(key)
            if (
                isinstance(value_a, (int, float))
                and not isinstance(value_a, bool)
                and isinstance(value_b, (int, float))
                and not isinstance(value_b, bool)
            ):
                weight = self.rng.uniform(0.0, 1.0)
                child_theta[key] = weight * value_a + (1.0 - weight) * value_b
        child_theta["_co_parent"] = parent_b.id
        return Equation(
            theta=child_theta,
            fitness=0.0,
            parent=parent_a.id,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            fn=parent_a.fn,
        )

    def spawn_next_generation(
        self, population: list, size: int, elite_fraction: float = 0.2
    ) -> list:
        """Produce the next generation from a fitness-scored `population`.

        The top `elite_fraction` of `population` (by fitness) survive
        unchanged (elitism); the remaining slots up to `size` are filled by
        crossing over pairs sampled from the elite pool and mutating the
        result, giving a full generational replacement step that combines
        elitism, crossover, and mutation in one call.
        """
        if not population:
            return []
        ranked = sorted(population, key=lambda e: e.fitness, reverse=True)
        elite_count = max(1, int(round(len(ranked) * elite_fraction)))
        elite = ranked[:elite_count]
        next_gen = list(elite[: min(size, len(elite))])
        while len(next_gen) < size:
            a, b = self.rng.choice(len(elite), size=2, replace=True)
            child = self.crossover(elite[a], elite[b])
            next_gen.append(self.mutate(child, scale=0.5))
        return next_gen[:size]

    def suppress(self, population: list, min_fitness: float) -> list:
        """Equation Suppression: keep only equations meeting a minimum fitness."""
        return [eq for eq in population if eq.fitness >= min_fitness]

    def lineage(self, equation: Equation, registry: dict) -> list:
        """Walk parent pointers through `registry` (id -> Equation) to the seed."""
        chain = [equation]
        current = equation
        while current.parent is not None and current.parent in registry:
            current = registry[current.parent]
            chain.append(current)
        return list(reversed(chain))
