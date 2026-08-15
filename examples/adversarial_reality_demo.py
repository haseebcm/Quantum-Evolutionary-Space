"""Phase 16 demo: classical red-team stress testing for candidate realities.

This walkthrough generates synthetic worst-case `Room` candidates, routes them
through the existing QES permission gate, runs a deliberately fragile numeric
evaluator, and summarizes which stress patterns failed most often. It is a
classical adversarial-style test harness for numeric candidate states, not
literal "reality breaking" and not free compute.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.adversarial import AdversarialGenerator, FailureCategory, run_red_team_campaign
from qes.permission import GenesisPermission
from qes.room import Room


def make_template_room() -> Room:
    return Room(
        x=np.zeros(3),
        x_star=np.zeros(3),
        lower=-np.ones(3),
        upper=np.ones(3),
        activation=np.ones(3),
        compute={"compute": 1.0},
    )


def fragile_evaluate(room: Room) -> dict[str, object]:
    """Toy evaluator that intentionally fails on several stress patterns."""
    if room.compute.get("compute", 0.0) > 200_000.0:
        raise MemoryError("synthetic compute budget exhausted")

    if room.equations:
        for equation in room.equations:
            if hasattr(equation, "evaluate"):
                value = float(equation.evaluate(room.x))
                if not np.isfinite(value):
                    raise FloatingPointError("pathological equation produced a non-finite value")

    denominator = float(room.x[0])
    if abs(denominator) < 1e-5:
        raise ZeroDivisionError("first state dimension is too close to zero for the toy solver")

    magnitude = float(np.max(np.abs(room.x)))
    if magnitude > 2.25:
        return {"passed": False, "reason": "state magnitude blew past the toy solver threshold"}

    if "cascade_stages" in room.memory and np.max(room.x - room.upper) > 0.02:
        return {"passed": False, "reason": "cascade markers indicate coupled recovery saturation"}

    score = float(np.sum(np.sin(room.x) / denominator))
    return {"passed": np.isfinite(score) and abs(score) < 20.0, "value": score}


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 16: ADVERSARIAL REALITY GENERATOR")
    print("=" * 60)

    template_room = make_template_room()
    generator = AdversarialGenerator(template_room, seed=16)
    candidates = generator.generate_all(n_per_category=2, seed=16)
    permission_gate = GenesisPermission(theta=1.0, gamma=1.2)
    report = run_red_team_campaign(candidates, fragile_evaluate, permission_gate)

    print(f"[batch]       generated {len(candidates)} classical red-team candidates")
    print(f"[summary]     total failures: {report.total_failures} / {report.total_candidates}")
    print("[per-category] failure rates")
    for category in FailureCategory:
        count = report.category_counts[category]
        failures = report.category_failure_counts[category]
        rate = report.category_failure_rates[category]
        print(f"  - {category.value:22s}: {failures:2d}/{count:2d} failed ({rate:.0%})")

    print("[improvements] suggested next hardening steps")
    if report.suggested_improvements:
        for suggestion in report.suggested_improvements:
            print(f"  - {suggestion}")
    else:
        print("  - No observed failures in this batch.")

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (ordinary CPU/RAM on this machine)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  This is classical synthetic stress-testing of numeric candidate states")
    print("  -- a computational red-team layer, not literal reality breaking or free compute.")


if __name__ == "__main__":
    main()
