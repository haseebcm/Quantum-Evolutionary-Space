"""Phase 18 demo: universal benchmark suite (qes-bench).

Runs the real QES optimizer against several classical reference baselines on
small synthetic objectives with a modest, identical evaluation budget.
"""
from __future__ import annotations

import time
import tracemalloc

from qes.bench import SCIPY_AVAILABLE, make_rastrigin_problem, make_sphere_problem, run_qes_bench


def main() -> None:
    tracemalloc.start()
    started = time.perf_counter()

    print("QES PHASE 18: UNIVERSAL BENCHMARK SUITE (qes-bench)")
    print("=" * 72)
    print(f"SciPy available for eligible baselines: {SCIPY_AVAILABLE}")
    print("Budget per run: 250 objective evaluations")
    print("Problems      : Sphere-3D, Rastrigin-3D")
    print("Baselines     : qes, differential_evolution, simulated_annealing,")
    print("                genetic_algorithm, gradient_method, cma_es")
    print()

    report = run_qes_bench(
        problems=(make_sphere_problem(dimension=3, seed=11), make_rastrigin_problem(dimension=3, seed=17)),
        baselines=[
            "differential_evolution",
            "simulated_annealing",
            "genetic_algorithm",
            "gradient_method",
            "cma_es",
        ],
        budget=250,
        seeds=(0, 1),
    )

    for problem in report.problems:
        print(f"Problem: {problem.name}")
        print(
            f"{'Baseline':<24}{'Perf':>12}{'Best':>12}{'EvalEff':>12}"
            f"{'Wall(s)':>10}{'Novelty':>12}{'DiscEvals':>12}"
        )
        print("-" * 94)
        for baseline_name in report.baseline_names:
            metrics = report.metrics[problem.name][baseline_name]
            print(
                f"{baseline_name:<24}"
                f"{metrics.performance:>12.6f}"
                f"{metrics.best_performance:>12.6f}"
                f"{metrics.compute_efficiency:>12.6f}"
                f"{metrics.mean_wall_time:>10.4f}"
                f"{metrics.novelty:>12.6f}"
                f"{metrics.evaluations_to_discovery:>12.1f}"
            )
        print()

    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 72)
    print("REAL MEASURED COST (ordinary CPU/RAM, classical optimization only)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print(f"  rows in report table : {len(report.table_rows())}")
    print("  QES used the real qes.intelligence.optimize entry point.")
    print("  Non-QES baselines above are minimal budget-matched references,")
    print("  not SOTA library implementations, except where SciPy is available")
    print("  for differential_evolution and gradient_method.")


if __name__ == "__main__":
    main()
