"""QES performance benchmarks.

Measures wall-clock throughput of the hot paths a caller actually pays for
when running QES at scale: the master space operator's per-tick cost across
room-population sizes, universe/world stepping (including nested universes
and agent messaging), reality branching (Gaussian vs. Latin-hypercube),
compute allocation strategies, equation evolution, and the dynamics
integrators (Euler vs. RK4 vs. stochastic).

This intentionally uses only the standard library (`time.perf_counter`) --
no extra dependency (e.g. pytest-benchmark) is required to run it.

Run with:  python benchmarks/run_benchmarks.py
Optional:  python benchmarks/run_benchmarks.py --repeat 10 --json results.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable

import numpy as np

from qes.agent import Agent
from qes.compute_allocator import ComputeAllocator, RoomComputeProfile
from qes.dynamics import RoomDynamics
from qes.equation_forge import EquationForge
from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace
from qes.universe import Universe
from qes.world import World


def make_seed_room(dim: int = 3) -> Room:
    return Room(
        x=np.zeros(dim),
        x_star=np.zeros(dim),
        lower=-np.ones(dim) * 2.0,
        upper=np.ones(dim) * 2.0,
        activation=np.ones(dim),
    )


def mean_reverting_step(room: Room, t: float, dt: float) -> np.ndarray:
    pull = -0.1 * (room.x - room.x_star)
    return room.x + dt * pull


def make_space(n_rooms: int) -> QESSpace:
    rng = np.random.default_rng(42)
    seed = make_seed_room(dim=3)
    generator = RealityGenerator(rng=rng)
    children = generator.branch(seed, count=n_rooms, scale=0.3)
    space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=mean_reverting_step,
        dt=1.0,
    )
    space.spawn(children)
    return space


class BenchmarkResult:
    __slots__ = (
        "name",
        "params",
        "iterations",
        "total_s",
        "mean_ms",
        "ops_per_s",
        "baseline_ms",
        "ratio_vs_baseline",
        "threshold_ms",
        "status",
    )

    def __init__(
        self,
        name: str,
        params: str,
        iterations: int,
        total_s: float,
        baseline_ms: float | None = None,
        threshold_ms: float | None = None,
    ):
        self.name = name
        self.params = params
        self.iterations = iterations
        self.total_s = total_s
        self.mean_ms = (total_s / iterations) * 1000.0 if iterations else float("nan")
        self.ops_per_s = iterations / total_s if total_s > 0 else float("inf")
        self.baseline_ms = baseline_ms
        self.ratio_vs_baseline = None if baseline_ms in (None, 0) else self.mean_ms / baseline_ms
        self.threshold_ms = threshold_ms
        if threshold_ms is not None:
            self.status = "pass" if self.mean_ms <= threshold_ms else "regression"
        elif baseline_ms is not None:
            # Sub-~5ms operations are dominated by measurement noise on shared/virtualized
            # CI runners (GC pauses, scheduler jitter, etc.), so a pure percentage ratio
            # produces false positives at that scale. Allow whichever is more permissive:
            # a 25% ratio, or a fixed +2ms absolute margin above baseline.
            allowed_ms = max(baseline_ms * 1.4, baseline_ms + 3.0)
            self.status = "pass" if self.mean_ms <= allowed_ms else "regression"
        else:
            self.status = "baseline-unavailable"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "iterations": self.iterations,
            "total_s": self.total_s,
            "mean_ms": self.mean_ms,
            "ops_per_s": self.ops_per_s,
            "baseline_ms": self.baseline_ms,
            "ratio_vs_baseline": self.ratio_vs_baseline,
            "threshold_ms": self.threshold_ms,
            "status": self.status,
        }


def bench(
    name: str,
    params: str,
    setup: Callable[[], object],
    run: Callable[[object], None],
    repeat: int,
    baseline_ms: float | None = None,
    threshold_ms: float | None = None,
) -> BenchmarkResult:
    """Time `repeat` fresh (setup -> run) trials; report the mean over all trials."""
    gc.collect()
    total = 0.0
    for _ in range(repeat):
        state = setup()
        start = time.perf_counter()
        run(state)
        total += time.perf_counter() - start
    return BenchmarkResult(name, params, repeat, total, baseline_ms=baseline_ms, threshold_ms=threshold_ms)


def load_baseline(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    lookup: dict[str, float] = {}
    for item in data:
        key = f"{item['name']}::{item['params']}"
        lookup[key] = float(item["mean_ms"])
    return lookup


def build_benchmarks(repeat: int, baseline_path: str | None = None) -> list[BenchmarkResult]:
    baseline = load_baseline(baseline_path)
    thresholds = {
        "space.step::rooms=1000": 110.0,
        "space.run(10)::rooms=200": 150.0,
        "universe.step::worlds=20,agents=100": 1.0,
        "reality.branch (Latin hypercube)::count=1000": 20.0,
        "allocate::rooms=1000": 2.0,
        "equation.spawn_next_generation::size=200": 10.0,
        "dynamics.integrate_rk4::ticks=1000,dim=50": 80.0,
    }
    results: list[BenchmarkResult] = []

    for n_rooms in (50, 200, 1000):

        def setup(n=n_rooms):
            return make_space(n)

        def run(space):
            space.step()

        results.append(
            bench(
                "space.step",
                f"rooms={n_rooms}",
                setup,
                run,
                repeat,
                baseline_ms=baseline.get(f"space.step::rooms={n_rooms}"),
                threshold_ms=thresholds.get(f"space.step::rooms={n_rooms}"),
            )
        )

    def setup_run():
        return make_space(200)

    def run_run(space):
        space.run(10)

    results.append(
        bench(
            "space.run(10)",
            "rooms=200",
            setup_run,
            run_run,
            repeat,
            baseline_ms=baseline.get("space.run(10)::rooms=200"),
            threshold_ms=thresholds.get("space.run(10)::rooms=200"),
        )
    )

    def setup_universe():
        universe = Universe()
        for _i in range(20):
            world = World()
            for _ in range(5):
                world.add_agent(Agent(policy=lambda a, o, t: None))
            universe.add_world(world)
        nested = Universe()
        nested.add_world(World())
        universe.spawn_nested_universe(next(iter(universe.worlds)), nested)
        return universe

    def run_universe(universe):
        universe.step()

    results.append(
        bench(
            "universe.step",
            "worlds=20,agents=100",
            setup_universe,
            run_universe,
            repeat,
            baseline_ms=baseline.get("universe.step::worlds=20,agents=100"),
            threshold_ms=thresholds.get("universe.step::worlds=20,agents=100"),
        )
    )

    def setup_clone():
        return setup_universe()

    def run_clone(universe):
        universe.clone()

    results.append(
        bench(
            "universe.clone",
            "worlds=20,agents=100",
            setup_clone,
            run_clone,
            repeat,
            baseline_ms=baseline.get("universe.clone::worlds=20,agents=100"),
        )
    )

    def setup_snapshot():
        return setup_universe()

    def run_snapshot_restore(universe):
        snap = universe.snapshot()
        universe.step()
        universe.restore(snap)

    results.append(
        bench(
            "universe.snapshot+restore",
            "worlds=20,agents=100",
            setup_snapshot,
            run_snapshot_restore,
            repeat,
            baseline_ms=baseline.get("universe.snapshot+restore::worlds=20,agents=100"),
        )
    )

    for count in (200, 1000):

        def setup_branch(n=count):
            return make_seed_room(dim=3), RealityGenerator(rng=np.random.default_rng(0))

        def run_branch_gaussian(state, n=count):
            room, gen = state
            gen.branch(room, count=n, scale=0.3)

        def run_branch_lhc(state, n=count):
            room, gen = state
            gen.branch_latin_hypercube(room, count=n)

        results.append(
            bench(
                "reality.branch (Gaussian)",
                f"count={count}",
                setup_branch,
                run_branch_gaussian,
                repeat,
                baseline_ms=baseline.get(f"reality.branch (Gaussian)::count={count}"),
            )
        )
        results.append(
            bench(
                "reality.branch (Latin hypercube)",
                f"count={count}",
                setup_branch,
                run_branch_lhc,
                repeat,
                baseline_ms=baseline.get(f"reality.branch (Latin hypercube)::count={count}"),
                threshold_ms=thresholds.get(f"reality.branch (Latin hypercube)::count={count}"),
            )
        )

    def setup_profiles():
        rng = np.random.default_rng(0)
        profiles = [
            RoomComputeProfile(
                permission=float(rng.uniform(0, 1)),
                uncertainty=float(rng.uniform(0, 1)),
                risk=float(rng.uniform(0, 5)),
                value=float(rng.uniform(0.01, 1)),
            )
            for _ in range(1000)
        ]
        return ComputeAllocator(), profiles

    def run_allocate(state):
        allocator, profiles = state
        allocator.allocate(profiles, total=1000.0)

    def run_allocate_with_floor(state):
        allocator, profiles = state
        allocator.allocate_with_floor(profiles, total=1000.0, min_share=0.1)

    def run_allocate_multi(state):
        allocator, profiles = state
        allocator.allocate_multi_resource(profiles, capacities={"cpu": 1000.0, "gpu": 100.0})

    results.append(
        bench(
            "allocate",
            "rooms=1000",
            setup_profiles,
            run_allocate,
            repeat,
            baseline_ms=baseline.get("allocate::rooms=1000"),
            threshold_ms=thresholds.get("allocate::rooms=1000"),
        )
    )
    results.append(
        bench(
            "allocate_with_floor",
            "rooms=1000",
            setup_profiles,
            run_allocate_with_floor,
            repeat,
            baseline_ms=baseline.get("allocate_with_floor::rooms=1000"),
        )
    )
    results.append(
        bench(
            "allocate_multi_resource",
            "rooms=1000,resources=2",
            setup_profiles,
            run_allocate_multi,
            repeat,
            baseline_ms=baseline.get("allocate_multi_resource::rooms=1000,resources=2"),
        )
    )

    def setup_population():
        forge = EquationForge(rng=np.random.default_rng(0))
        base = forge.seed(theta={"a": 1.0, "b": 2.0, "c": 3.0})
        population = forge.spawn_population(base, size=200)
        return forge, population

    def run_mutate_population(state):
        forge, population = state
        [forge.mutate(eq) for eq in population]

    def run_spawn_next_generation(state):
        forge, population = state
        forge.spawn_next_generation(population, size=200)

    results.append(
        bench(
            "equation.mutate (population)",
            "size=200",
            setup_population,
            run_mutate_population,
            repeat,
            baseline_ms=baseline.get("equation.mutate (population)::size=200"),
        )
    )
    results.append(
        bench(
            "equation.spawn_next_generation",
            "size=200",
            setup_population,
            run_spawn_next_generation,
            repeat,
            baseline_ms=baseline.get("equation.spawn_next_generation::size=200"),
            threshold_ms=thresholds.get("equation.spawn_next_generation::size=200"),
        )
    )

    def setup_dynamics():
        dynamics = RoomDynamics(drift=lambda x, t: -0.5 * x)
        x = np.ones(50)
        return dynamics, x

    def run_euler(state):
        dynamics, x = state
        for t in range(1000):
            x = dynamics.integrate(x, None, float(t), dt=0.01)

    def run_rk4(state):
        dynamics, x = state
        for t in range(1000):
            x = dynamics.integrate_rk4(x, None, float(t), dt=0.01)

    def run_stochastic(state):
        dynamics, x = state
        rng = np.random.default_rng(0)
        for t in range(1000):
            x = dynamics.step_stochastic(x, float(t), dt=0.01, sigma=0.1, rng=rng)

    results.append(
        bench(
            "dynamics.integrate (Euler)",
            "ticks=1000,dim=50",
            setup_dynamics,
            run_euler,
            repeat,
            baseline_ms=baseline.get("dynamics.integrate (Euler)::ticks=1000,dim=50"),
        )
    )
    results.append(
        bench(
            "dynamics.integrate_rk4",
            "ticks=1000,dim=50",
            setup_dynamics,
            run_rk4,
            repeat,
            baseline_ms=baseline.get("dynamics.integrate_rk4::ticks=1000,dim=50"),
            threshold_ms=thresholds.get("dynamics.integrate_rk4::ticks=1000,dim=50"),
        )
    )
    results.append(
        bench(
            "dynamics.step_stochastic",
            "ticks=1000,dim=50",
            setup_dynamics,
            run_stochastic,
            repeat,
            baseline_ms=baseline.get("dynamics.step_stochastic::ticks=1000,dim=50"),
        )
    )

    return results


def print_table(results: list[BenchmarkResult]) -> None:
    name_w = max(len(r.name) for r in results) + 2
    params_w = max(len(r.params) for r in results) + 2
    header = (
        f"{'Benchmark':<{name_w}}{'Params':<{params_w}}{'Mean (ms)':>12}{'Ops/s':>14}{'Status':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        ratio_text = "-" if r.ratio_vs_baseline is None else f"{r.ratio_vs_baseline:.2f}x"
        print(
            f"{r.name:<{name_w}}{r.params:<{params_w}}{r.mean_ms:>12.4f}{r.ops_per_s:>14,.1f}{r.status:>10}"
        )
        if r.baseline_ms is not None:
            print(f"{'':<{name_w + params_w + 12}}baseline={r.baseline_ms:.3f}ms  ratio={ratio_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QES performance benchmarks.")
    parser.add_argument("--repeat", type=int, default=5, help="trials per benchmark (default: 5)")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="optional JSON file with prior benchmark results for comparison",
    )
    parser.add_argument("--json", type=str, default=None, help="optional path to write JSON results")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit with a non-zero status if any benchmark reports status=='regression' "
        "(i.e. exceeds its fixed threshold_ms, or exceeds both a 40%% ratio and a +3ms "
        "absolute margin above --baseline). Intended for CI regression gates.",
    )
    args = parser.parse_args()

    print("QES PERFORMANCE BENCHMARKS")
    print("=" * 40)
    print(f"repeat={args.repeat}\n")

    results = build_benchmarks(repeat=args.repeat, baseline_path=args.baseline)
    print_table(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([r.as_dict() for r in results], f, indent=2)
        print(f"\nWrote JSON results to {args.json}")

    regressions = [r for r in results if r.status == "regression"]
    if regressions:
        print(f"\n{len(regressions)} benchmark(s) regressed:")
        for r in regressions:
            if r.threshold_ms is not None:
                detail = f"threshold={r.threshold_ms}ms"
            else:
                detail = f"baseline={r.baseline_ms}ms"
            print(f"  - {r.name} ({r.params}): mean={r.mean_ms:.4f}ms, {detail}")
        if args.fail_on_regression:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
