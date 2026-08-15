"""Phase 2 demo: the complete closed-loop autonomous cycle, running for real.

    INTENT -> GENERATION -> REALITY POPULATION -> EXECUTION -> DIVERGENCE
        -> PERMISSION -> SELECTION -> CONVERGENCE -> KNOWLEDGE EXTRACTION
        -> PATTERN MEMORY -> ADAPTIVE REGENERATION -> new population -> ...

Unlike the two `full_stack_integration*.py` scripts (one measured *pass*
through many modules), this script runs the same intent through several
*generations* without any human re-seeding the population by hand: each
generation's outcome (its dominant room, convergence, entropy) decides
both what gets remembered (`PatternMemory`) and how many children the
next generation gets (`adaptive_branch_count`), and every generation is
checked for structural invariants and recorded to an event log.

Run with:  python examples/closed_loop_cycle.py
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.cycle import AutonomousCycle
from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace


def make_seed_room(dim: int = 3) -> Room:
    return Room(
        x=np.zeros(dim),
        x_star=np.ones(dim) * 0.5,
        lower=-np.ones(dim),
        upper=np.ones(dim),
        activation=np.ones(dim),
    )


def mean_reverting_step(room: Room, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    pull = -0.25 * (room.x - room.x_star)
    noise = rng.normal(0.0, 0.03, size=room.dim)
    return room.x + dt * pull + noise


def main() -> None:
    rng = np.random.default_rng(42)
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES CLOSED-LOOP AUTONOMOUS CYCLE (Phase 2)")
    print("=" * 60)

    space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=lambda room, t, dt: mean_reverting_step(room, t, dt, rng),
        dt=1.0,
    )
    cycle = AutonomousCycle(
        intent="design-refinement",
        space=space,
        generator=RealityGenerator(rng=rng),
        score_fn=lambda room: float(np.sum(np.abs(room.x - room.x_star))),
    )

    seed = make_seed_room()
    cycle.seed(cycle.generator.branch(seed, count=8, scale=0.2))
    print(f"[intent]      '{cycle.intent}' seeded with {len(space.rooms)} rooms "
          f"(no human re-seeding after this point)")

    reports = cycle.run(generations=6)

    print("\ngeneration | active | convergence | entropy | dominant_perm | "
          "children_next | events")
    print("-" * 78)
    for r in reports:
        print(
            f"{r.generation:^10d} | {r.telemetry.active:^6d} | "
            f"{r.telemetry.convergence:^11.3f} | {r.telemetry.entropy:^7.3f} | "
            f"{r.telemetry.dominant_permission:^13.3f} | "
            f"{r.n_children_next:^13d} | {r.events_recorded:^6d}"
        )

    all_passed = all(r.invariants.passed for r in reports)
    total_events = len(cycle.events)
    stored_patterns = cycle.memory.all_patterns(cycle.intent)

    print("=" * 60)
    print(f"[invariants]  every generation passed the global invariant check: {all_passed}")
    print(f"[events]      {total_events} events recorded across "
          f"{len(reports)} generations (SPAWN/EXECUTE/DIVERGE/PERMISSION/SELECT/MERGE)")
    print(f"[memory]      {len(stored_patterns)} pattern(s) on file for "
          f"intent '{cycle.intent}' after retiring dominated entries")
    print(f"[population]  final active rooms: {len(space.active_rooms())} "
          f"(started from {8}, adaptively regenerated each generation)")

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  modules touched      : cycle, space, room, permission, "
          "reality_generator, events, invariants, patterns (8 modules)")
    print("  This loop is a classical fixed-point/evolutionary search over "
          "numpy arrays: no population regenerates itself for free, and "
          "'closed-loop' means no human re-seeds it between generations, "
          "not that computation is bypassed.")


if __name__ == "__main__":
    main()
