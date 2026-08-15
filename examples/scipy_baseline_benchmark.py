"""Example: QES governed search vs. an external optimizer baseline (SciPy).

This is a genuine external-baseline comparison rather than a purely synthetic,
self-referential demo: the objective is `scipy.optimize.rosen`, the standard
N-dimensional Rosenbrock benchmark function shipped with SciPy and widely used
across the optimization literature, and the reference solution comes from
`scipy.optimize.minimize` (a real, independently implemented optimizer), not
from QES itself.

QES explores the same objective through `qes.intelligence.optimize` -- the
framework's adaptive, gradient-informed search strategy (Adam-style per-
dimension gradient steps, Rechenberg 1/5-rule self-adaptive stochastic search,
and stagnation-triggered restarts), running inside the governed room/space
loop (permission gates, reality branching). This demonstrates that QES's
domain-agnostic room/space primitives, combined with real search intelligence
rather than a blind random walk, can competitively attack a real, externally-
defined optimization problem.

Requires the optional `scipy` dependency: `pip install scipy`.
"""
from __future__ import annotations

import time

import numpy as np

from qes.intelligence import AdaptiveSearchConfig, optimize
from qes.room import Room

try:
    from scipy.optimize import minimize, rosen
except ImportError as exc:  # pragma: no cover - exercised only without scipy installed
    raise SystemExit(
        "This example requires the optional 'scipy' dependency. Install it with "
        "`pip install scipy` and re-run."
    ) from exc


DIM = 6


def make_seed_room(dim: int = DIM) -> Room:
    return Room(
        x=np.full(dim, -1.0),
        x_star=np.ones(dim),  # rosen's known global minimum is the all-ones vector
        lower=-np.ones(dim) * 5.0,
        upper=np.ones(dim) * 5.0,
        activation=np.ones(dim),
    )


def run_qes_search(
    seed: Room, rng: np.random.Generator, iterations: int = 300
) -> tuple[np.ndarray, float, int]:
    """QES's adaptive gradient search (`qes.intelligence.optimize`) on the objective;
    returns (best_x, best_value, evals)."""
    config = AdaptiveSearchConfig(gradient_prob=0.85, learning_rate=0.03, stagnation_patience=25)
    result = optimize(
        rosen,
        seed,
        iterations=iterations,
        population=40,
        config=config,
        rng=rng,
    )
    return result.best_x, result.best_value, result.evaluations


def run_scipy_baseline(seed: Room) -> tuple[np.ndarray, float, int]:
    """Independent external baseline: SciPy's Nelder-Mead optimizer on the same objective."""
    result = minimize(rosen, x0=seed.x, method="Nelder-Mead")
    return result.x, float(result.fun), int(result.nfev)


def main() -> None:
    rng = np.random.default_rng(7)
    seed = make_seed_room()

    start = time.perf_counter()
    qes_x, qes_value, qes_evals = run_qes_search(seed, rng)
    qes_time = time.perf_counter() - start

    start = time.perf_counter()
    scipy_x, scipy_value, scipy_evals = run_scipy_baseline(seed)
    scipy_time = time.perf_counter() - start

    print("QES vs. external baseline: Rosenbrock minimization")
    print("=" * 55)
    print(f"Objective   : scipy.optimize.rosen (dim={DIM})")
    print(f"Known optimum: x* = ones({DIM}), f(x*) = 0.0\n")

    print(f"{'Method':<24}{'f(x)':>14}{'evals':>12}{'time (s)':>12}")
    print("-" * 62)
    print(f"{'QES adaptive search':<24}{qes_value:>14.6f}{qes_evals:>12}{qes_time:>12.4f}")
    print(f"{'SciPy Nelder-Mead':<24}{scipy_value:>14.6f}{scipy_evals:>12}{scipy_time:>12.4f}")

    print("\nQES best x   :", np.round(qes_x, 3))
    print("SciPy best x :", np.round(scipy_x, 3))


if __name__ == "__main__":
    main()
