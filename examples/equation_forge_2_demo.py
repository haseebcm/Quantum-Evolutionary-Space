"""Phase 4 demo: Equation Forge 2.0 running a real structural search.

This script evolves symbolic equation trees rather than only perturbing
scalar coefficients. A population of AST equations is generated,
mutated, crossed over, simplified, checked for dimensional validity,
screened for numerical/stability failures, and then ranked against a
synthetic target signal.

As with the rest of QES, this is ordinary classical numpy/Python
computation measured in real wall time and memory. The "forge" evolves
Python data structures on this CPU; it does not imply literal quantum
computing.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.equation_ast import ASTNode, EquationASTForge


def main() -> None:
    rng = np.random.default_rng(2026)
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES EQUATION FORGE 2.0 DEMO")
    print("=" * 60)

    training_states = rng.uniform(-1.0, 1.0, size=(128, 2))
    targets = training_states[:, 0] ** 2 + 0.5 * training_states[:, 1] - 0.25

    def score_fn(node: ASTNode) -> float:
        predictions = np.asarray([node.evaluate(state) for state in training_states], dtype=float)
        return float(np.mean((predictions - targets) ** 2))

    forge = EquationASTForge(
        population_size=40,
        elite_fraction=0.2,
        mutation_rate=0.9,
        crossover_rate=0.7,
        max_depth=4,
        rng=rng,
    )
    result = forge.evolve(
        dim=2,
        score_fn=score_fn,
        lower=-np.ones(2),
        upper=np.ones(2),
        generations=18,
        variable_names=("x0", "x1"),
        sample_count=96,
        stability_threshold=250.0,
        complexity_penalty=5e-4,
    )

    best = result.best_equation
    final_snapshot = result.history[-1]
    print(f"[best]        expression={best.root}")
    print(f"[best]        objective={best.objective:.6f}, depth={best.depth()}, nodes={best.node_count()}")
    print(f"[population]  attempted={final_snapshot.attempted}, valid={final_snapshot.valid}")
    print(
        f"[evolution]   generation-0 objective={result.history[0].best_objective:.6f}, "
        f"final objective={final_snapshot.best_objective:.6f}"
    )

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  modules touched      : equation_ast (AST, primitives, mutation, crossover, "
          "simplification, validation, fitness)")
    print("  This is a classical symbolic/evolutionary search over numpy-evaluated "
          "expression trees, not literal quantum computation.")


if __name__ == "__main__":
    main()
