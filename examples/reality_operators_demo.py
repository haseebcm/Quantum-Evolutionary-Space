"""Phase 3 demo: next-generation Reality Generator (operators + families).

Demonstrates the reality operators added in `qes.reality_operators` --
mutation, crossover, interpolation, extrapolation, inversion, structured
perturbation, dimensional transformation, topology transformation,
equation substitution, and parameter transformation -- plus grouping a
branching batch into a `RealityFamily` with family-level statistics
instead of a flat, unrelated room list.

Run with:  python examples/reality_operators_demo.py
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.reality_operators import (
    RealityFamilyBuilder,
    crossover,
    dimensional_transformation,
    equation_substitution,
    extrapolate,
    interpolate,
    inversion,
    mutation,
    parameter_transformation,
    perturbation,
    topology_transformation,
)
from qes.room import Room


def make_room(x: np.ndarray, dim: int = 3) -> Room:
    return Room(
        x=x,
        x_star=np.ones(dim) * 0.5,
        lower=-np.ones(dim) * 3,
        upper=np.ones(dim) * 3,
        activation=np.ones(dim),
    )


def main() -> None:
    rng = np.random.default_rng(5)
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES REALITY OPERATORS (Phase 3)")
    print("=" * 60)

    parent_a = make_room(np.zeros(3))
    parent_b = make_room(np.ones(3))

    print(f"[mutation]         {mutation(parent_a, rng, scale=0.2).x.round(3)}")
    print(f"[crossover]        {crossover(parent_a, parent_b, rng, alpha=0.5).x.round(3)}")
    print(f"[interpolate]      {interpolate(parent_a, parent_b, 0.25).x.round(3)}")
    print(f"[extrapolate]      {extrapolate(parent_a, parent_b, 1.5).x.round(3)}")
    print(f"[inversion]        {inversion(parent_a).x.round(3)}")
    print(f"[perturbation]     {perturbation(parent_a, rng, scale=0.1, directions=np.eye(3)).x.round(3)}")
    dim_transform = dimensional_transformation(parent_a.clone(x=np.ones(3)), np.eye(3) * 2.0)
    print(f"[dim. transform]   {dim_transform.x.round(3)}")
    topo = topology_transformation(make_room(np.array([1.0, 2.0, 3.0])), [2, 0, 1])
    print(f"[topology]         {topo.x.round(3)}")
    print(f"[equation subst.]  {equation_substitution(parent_a, ['E1', 'E2']).equations}")
    room_with_theta = parent_a.clone(theta={"a": 1.0})
    transformed = parameter_transformation(room_with_theta, lambda t: {**t, "b": 2.0})
    print(f"[param transform]  {transformed.theta}")

    # --- Reality family: a batch of siblings with aggregate statistics ---
    builder = RealityFamilyBuilder(rng=rng)
    family = builder.build(parent_a, count=20, mutation_scale=0.3, second_parent=parent_b)
    best = family.best(score_fn=lambda room: float(np.sum(np.abs(room.x - room.x_star))))

    print("=" * 60)
    print(f"[family]      built {len(family)} children from parent {family.parent_id}")
    print(f"[family]      mean_state={family.mean_state().round(3)}, diversity={family.diversity():.4f}")
    print(f"[family]      best child (closest to x_star)={best.x.round(3)}")

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  modules touched      : reality_operators, room (2 modules)")
    print("  Every operator above is a deterministic numpy array transform "
          "on a Room object -- this is classical genetic-programming-style "
          "search, not physical reality generation.")


if __name__ == "__main__":
    main()
