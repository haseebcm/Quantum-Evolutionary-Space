"""Phase 13 demo: heuristic meta-learning for QES search-strategy selection.

This walkthrough does not show literal machine learning or quantum hardware.
It shows a classical controller that inspects cheap problem features, looks at
tracked past outcomes, and recommends search settings accordingly.
"""
from __future__ import annotations

import time
import tracemalloc

from qes.meta_learning import MetaController, ProblemStructure, SearchHistory, record_outcome


def print_recommendation(label: str, recommendation: object) -> None:
    print(f"{label}")
    print(f"  optimizer                : {recommendation.optimizer}")
    print(f"  model class              : {recommendation.model_class}")
    print(f"  mutation rate            : {recommendation.mutation_rate:.4f}")
    print(f"  population size          : {recommendation.population_size}")
    print(f"  branching factor         : {recommendation.branching_factor}")
    print(f"  exploration/exploitation : {recommendation.exploration_exploitation:.4f}")
    print(f"  compute allocation       : {recommendation.compute_allocation:.4f}")
    print(f"  convergence threshold    : {recommendation.convergence_threshold:.6f}")


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES META-LEARNING DEMO (Phase 13)")
    print("=" * 60)

    controller = MetaController()
    history = SearchHistory()

    low_dimensional = ProblemStructure(
        dimensionality=4,
        bounds_width=1.5,
        ruggedness=0.20,
        noise_level=0.05,
        constraint_count=1,
        gradient_available=True,
    )
    wide_exploratory = ProblemStructure(
        dimensionality=18,
        bounds_width=14.0,
        ruggedness=0.60,
        noise_level=0.20,
        constraint_count=2,
        gradient_available=False,
    )
    rugged_noisy = ProblemStructure(
        dimensionality=30,
        bounds_width=25.0,
        ruggedness=0.85,
        noise_level=0.60,
        constraint_count=4,
        gradient_available=False,
    )

    initial_low = controller.recommend(low_dimensional, history)
    initial_wide = controller.recommend(wide_exploratory, history)
    initial_rugged = controller.recommend(rugged_noisy, history)

    print_recommendation("[cold start] low-dimensional tight-bounds problem", initial_low)
    print_recommendation("[cold start] wide exploratory problem", initial_wide)
    print_recommendation("[cold start] rugged noisy problem", initial_rugged)

    record_outcome(history, low_dimensional, initial_low, True, 10, 0.11)
    record_outcome(
        history,
        wide_exploratory,
        controller.recommend(wide_exploratory, SearchHistory()),
        False,
        40,
        3.20,
    )
    record_outcome(
        history,
        wide_exploratory,
        initial_wide.__class__(
            optimizer="evolutionary",
            mutation_rate=0.24,
            population_size=96,
            branching_factor=11,
            exploration_exploitation=0.24,
            compute_allocation=0.92,
            convergence_threshold=0.004,
            model_class="EvolutionaryPopulation",
        ),
        True,
        8,
        0.09,
    )
    record_outcome(
        history,
        wide_exploratory,
        initial_wide.__class__(
            optimizer="evolutionary",
            mutation_rate=0.22,
            population_size=88,
            branching_factor=10,
            exploration_exploitation=0.27,
            compute_allocation=0.88,
            convergence_threshold=0.0045,
            model_class="EvolutionaryPopulation",
        ),
        True,
        9,
        0.12,
    )
    record_outcome(
        history,
        rugged_noisy,
        initial_rugged.__class__(
            optimizer="evolutionary",
            mutation_rate=0.12,
            population_size=40,
            branching_factor=5,
            exploration_exploitation=0.78,
            compute_allocation=0.55,
            convergence_threshold=0.020,
            model_class="EvolutionaryPopulation",
        ),
        False,
        50,
        5.50,
    )

    shifted_wide = controller.recommend(wide_exploratory, history)

    print("-" * 60)
    print("After recording synthetic outcomes, the recommendation shifts:")
    print_recommendation("[history-aware] wide exploratory problem", shifted_wide)

    elapsed = time.perf_counter() - t_start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (ordinary classical computation)")
    print(f"  wall time            : {elapsed:.6f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  recorded outcomes    : 5 synthetic search runs held in Python memory")
    print("  This is a documented heuristic over tracked history and cheap feature")
    print("  signals: it is not literal machine learning, not AGI, and not free compute.")


if __name__ == "__main__":
    main()
