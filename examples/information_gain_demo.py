"""Phase 8 demo: information-gain-driven branch selection for QES.

This script shows a classical active-search workflow: start from one current
population of rooms, simulate several candidate branch outcomes, measure
which branch would reduce uncertainty the most, rank those candidates, and
allocate more compute to the highest-information experiments.

All numbers printed below are ordinary CPU/RAM costs of Python/Numpy code on
this machine. "Branches" and "rooms" are in-memory data structures, not
literal parallel universes or quantum hardware.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.information_gain import (
    InformationGainEstimator,
    allocate_by_information_gain,
    population_entropy,
    rank_by_information_gain,
)
from qes.room import Room


def make_room(weight: float, x_value: float) -> Room:
    return Room(
        x=np.array([x_value, x_value + 0.1, x_value + 0.2]),
        x_star=np.array([0.5, 0.5, 0.5]),
        lower=-np.ones(3),
        upper=np.ones(3),
        activation=np.ones(3),
        weight=weight,
    )


def make_population(weights: list[float]) -> list[Room]:
    return [make_room(weight, x_value=0.05 * index) for index, weight in enumerate(weights)]


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES INFORMATION GAIN DEMO (Phase 8)")
    print("=" * 60)

    current_population = make_population([0.55, 0.2, 0.15, 0.1])
    estimator = InformationGainEstimator()

    candidates = {
        "focused_probe": (current_population, make_population([0.82, 0.1, 0.05, 0.03])),
        "moderate_probe": (current_population, make_population([0.65, 0.18, 0.1, 0.07])),
        "steady_probe": (current_population, make_population([0.55, 0.2, 0.15, 0.1])),
        "confusing_probe": (current_population, make_population([0.3, 0.25, 0.23, 0.22])),
    }

    before_entropy = population_entropy(current_population)
    scores = estimator.estimate_expected(candidates)
    ranked = rank_by_information_gain(scores)
    allocations = allocate_by_information_gain(scores, total_budget=100.0, min_floor=5.0)

    print(f"Current population entropy: {before_entropy:.6f}")
    print("\ncandidate        | after entropy | info gain | compute budget")
    print("-" * 60)
    for name, score in ranked:
        after_entropy = population_entropy(candidates[name][1])
        print(
            f"{name:<16} | {after_entropy:>13.6f} | {score:>9.6f} | "
            f"{allocations[name]:>14.4f}"
        )

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (ordinary classical computation)")
    print(f"  wall time            : {elapsed:.6f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  This is classical branch scoring over room-weight distributions: "
          "QES learns which simulated branch would reduce uncertainty most, "
          "then can hand that priority signal to broader compute-allocation logic.")


if __name__ == "__main__":
    main()
