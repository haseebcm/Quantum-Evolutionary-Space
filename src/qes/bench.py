"""Phase 18 -- universal benchmark suite (``qes-bench``).

This module provides a small, honest benchmark harness for comparing the real
QES optimizer against classical baseline families under identical objective,
seed, bound constraints, and function-evaluation budget conditions.

Every non-QES baseline here is intentionally modest in scope: each runner is a
minimal, budget-matched reference implementation of its named family for fair
comparison purposes, not a literature-SOTA implementation.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from qes.intelligence import AdaptiveSearchConfig, optimize
from qes.room import Room
from qes.selection import pareto_front

try:  # pragma: no cover - exercised in environments without scipy
    from scipy.optimize import differential_evolution as _scipy_differential_evolution
    from scipy.optimize import minimize as _scipy_minimize

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without scipy
    _scipy_differential_evolution = None
    _scipy_minimize = None
    SCIPY_AVAILABLE = False

ObjectiveFn = Callable[[np.ndarray], float]
ConstraintFn = Callable[[np.ndarray], float]
BaselineRunner = Callable[["BenchmarkProblem", int, int], "BaselineResult"]


class _BudgetExhausted(RuntimeError):
    """Internal control-flow signal used to terminate a run exactly at budget."""


def _validate_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    return int(seed)


def _validate_budget(budget: int) -> int:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be an integer > 0")
    return budget


def _mean_pairwise_distance(points: np.ndarray) -> float:
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if points.shape[0] < 2:
        return 0.0
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    upper = np.triu_indices(points.shape[0], k=1)
    return float(np.mean(distances[upper]))


def _mix_seed(problem_seed: int, run_seed: int) -> int:
    mixed = (int(problem_seed) ^ ((int(run_seed) * 0x9E3779B1) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return int(mixed)


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _rastrigin(x: np.ndarray) -> float:
    n = x.shape[0]
    return float(10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def _rosenbrock(x: np.ndarray) -> float:
    if x.shape[0] < 2:  # pragma: no cover - degenerate fallback
        return float(np.sum((1.0 - x) ** 2))
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


@dataclass
class BenchmarkProblem:
    """A bounded benchmark objective.

    The objective is classical NumPy code; there is no literal quantum
    computation involved anywhere in this suite.
    """

    name: str
    objective: ObjectiveFn
    bounds: Sequence[tuple[float, float]]
    dimension: int
    optimum_value: float | None = None
    optimum_x: np.ndarray | None = None
    seed: int = 0
    constraint_fn: ConstraintFn | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not callable(self.objective):
            raise TypeError("objective must be callable")
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool) or self.dimension <= 0:
            raise ValueError("dimension must be an integer > 0")
        self.seed = _validate_seed(self.seed)
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (self.dimension, 2):
            raise ValueError("bounds must have shape (dimension, 2)")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("bounds must be finite")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each lower bound must be strictly less than its upper bound")
        self.bounds = [(float(lower), float(upper)) for lower, upper in bounds]
        if self.optimum_x is not None:
            optimum_x = np.asarray(self.optimum_x, dtype=float)
            if optimum_x.shape != (self.dimension,):
                raise ValueError("optimum_x must have shape (dimension,)")
            if not np.all(np.isfinite(optimum_x)):
                raise ValueError("optimum_x must contain only finite values")
            self.optimum_x = optimum_x
        if self.optimum_value is not None:
            optimum_value = float(self.optimum_value)
            if not np.isfinite(optimum_value):
                raise ValueError("optimum_value must be finite when provided")
            self.optimum_value = optimum_value
        if self.constraint_fn is not None and not callable(self.constraint_fn):
            raise TypeError("constraint_fn must be callable when provided")

    @property
    def lower(self) -> np.ndarray:
        return np.asarray([item[0] for item in self.bounds], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray([item[1] for item in self.bounds], dtype=float)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def span(self) -> np.ndarray:
        return self.upper - self.lower

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float), self.lower, self.upper)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower, self.upper)

    def constraint_violation(self, x: np.ndarray) -> float:
        raw = np.asarray(x, dtype=float)
        lower_violation = np.maximum(self.lower - raw, 0.0)
        upper_violation = np.maximum(raw - self.upper, 0.0)
        total = float(np.sum(lower_violation + upper_violation))
        if self.constraint_fn is not None:
            extra = float(self.constraint_fn(self.clip(raw)))
            if not np.isfinite(extra) or extra < 0.0:
                raise ValueError("constraint_fn must return a finite value >= 0")
            total += extra
        return total


def make_sphere_problem(*, dimension: int = 4, seed: int = 11) -> BenchmarkProblem:
    """Return the classic Sphere minimization problem."""

    return BenchmarkProblem(
        name=f"Sphere-{dimension}D",
        objective=_sphere,
        bounds=[(-5.12, 5.12)] * dimension,
        dimension=dimension,
        optimum_value=0.0,
        optimum_x=np.zeros(dimension, dtype=float),
        seed=seed,
    )


def make_rastrigin_problem(*, dimension: int = 4, seed: int = 17) -> BenchmarkProblem:
    """Return the classic Rastrigin minimization problem."""

    return BenchmarkProblem(
        name=f"Rastrigin-{dimension}D",
        objective=_rastrigin,
        bounds=[(-5.12, 5.12)] * dimension,
        dimension=dimension,
        optimum_value=0.0,
        optimum_x=np.zeros(dimension, dtype=float),
        seed=seed,
    )


def make_rosenbrock_problem(*, dimension: int = 4, seed: int = 23) -> BenchmarkProblem:
    """Return the classic Rosenbrock minimization problem."""

    return BenchmarkProblem(
        name=f"Rosenbrock-{dimension}D",
        objective=_rosenbrock,
        bounds=[(-3.0, 3.0)] * dimension,
        dimension=dimension,
        optimum_value=0.0,
        optimum_x=np.ones(dimension, dtype=float),
        seed=seed,
    )


SPHERE_PROBLEM = make_sphere_problem()
RASTRIGIN_PROBLEM = make_rastrigin_problem()
ROSENBROCK_PROBLEM = make_rosenbrock_problem()
DEFAULT_BENCHMARK_PROBLEMS = (SPHERE_PROBLEM, RASTRIGIN_PROBLEM, ROSENBROCK_PROBLEM)


@dataclass
class BaselineResult:
    """Outcome of one benchmark run."""

    baseline_name: str
    problem_name: str
    seed: int
    solution: np.ndarray
    objective_value: float
    wall_time: float
    evaluations: int
    budget: int
    constraint_violations: int
    constraint_violation_magnitude: float
    visited_points: np.ndarray
    visited_values: np.ndarray
    best_trace: np.ndarray
    time_to_discovery_seconds: float
    evaluations_to_discovery: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.solution = np.asarray(self.solution, dtype=float)
        self.visited_points = np.asarray(self.visited_points, dtype=float)
        self.visited_values = np.asarray(self.visited_values, dtype=float)
        self.best_trace = np.asarray(self.best_trace, dtype=float)
        if self.solution.ndim != 1:
            raise ValueError("solution must be one-dimensional")
        if self.visited_points.ndim != 2:
            raise ValueError("visited_points must be two-dimensional")
        if self.visited_points.shape[0] != self.visited_values.shape[0]:
            raise ValueError("visited_points and visited_values length mismatch")
        if self.best_trace.shape != self.visited_values.shape:
            raise ValueError("best_trace must match visited_values shape")
        if self.evaluations != self.visited_values.shape[0]:
            raise ValueError("evaluations must equal the number of visited values")
        if self.evaluations > self.budget:
            raise ValueError("evaluations cannot exceed budget")


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics across repeated runs of one baseline on one problem."""

    performance: float
    compute_efficiency: float
    constraint_violations: float
    novelty: float
    robustness: float
    solution_diversity: float
    time_to_discovery_seconds: float
    evaluations_to_discovery: float
    mean_wall_time: float
    mean_evaluations: float
    best_performance: float


@dataclass
class BenchReport:
    """Structured output of :func:`run_qes_bench`."""

    problems: tuple[BenchmarkProblem, ...]
    baseline_names: tuple[str, ...]
    budget: int
    seeds: tuple[int, ...]
    runs: dict[str, dict[str, list[BaselineResult]]]
    metrics: dict[str, dict[str, BenchmarkMetrics]]

    def table_rows(self) -> list[dict[str, object]]:
        """Return a flat comparison table suitable for printing."""

        rows: list[dict[str, object]] = []
        for problem in self.problems:
            for baseline_name in self.baseline_names:
                item = self.metrics[problem.name][baseline_name]
                rows.append(
                    {
                        "problem": problem.name,
                        "baseline": baseline_name,
                        "performance": item.performance,
                        "compute_efficiency": item.compute_efficiency,
                        "constraint_violations": item.constraint_violations,
                        "novelty": item.novelty,
                        "robustness": item.robustness,
                        "solution_diversity": item.solution_diversity,
                        "time_to_discovery_seconds": item.time_to_discovery_seconds,
                        "evaluations_to_discovery": item.evaluations_to_discovery,
                        "mean_wall_time": item.mean_wall_time,
                        "mean_evaluations": item.mean_evaluations,
                        "best_performance": item.best_performance,
                    }
                )
        return rows


class _EvaluationTracker:
    def __init__(self, problem: BenchmarkProblem, budget: int, *, discovery_tolerance: float = 1e-6) -> None:
        self.problem = problem
        self.budget = _validate_budget(budget)
        self.discovery_tolerance = float(discovery_tolerance)
        if not np.isfinite(self.discovery_tolerance) or self.discovery_tolerance < 0.0:
            raise ValueError("discovery_tolerance must be finite and >= 0")
        self.start_time = perf_counter()
        self.points: list[np.ndarray] = []
        self.values: list[float] = []
        self.times: list[float] = []
        self.best_x = problem.center.copy()
        self.best_value = float("inf")
        self.constraint_violations = 0
        self.constraint_violation_magnitude = 0.0

    @property
    def evaluations(self) -> int:
        return len(self.values)

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    def should_stop(self) -> bool:
        if self.evaluations >= self.budget:
            return True
        if self.problem.optimum_value is None:
            return False
        return self.best_value <= self.problem.optimum_value + self.discovery_tolerance

    def evaluate(self, x: np.ndarray) -> float:
        if self.evaluations >= self.budget:
            raise _BudgetExhausted("evaluation budget exhausted")
        candidate = np.asarray(x, dtype=float)
        if candidate.shape != (self.problem.dimension,):
            raise ValueError("candidate shape must be (problem.dimension,)")
        violation = self.problem.constraint_violation(candidate)
        feasible = self.problem.clip(candidate)
        value = float(self.problem.objective(feasible))
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        self.points.append(feasible.copy())
        self.values.append(value)
        self.times.append(perf_counter() - self.start_time)
        if violation > 0.0:
            self.constraint_violations += 1
            self.constraint_violation_magnitude += violation
        if value < self.best_value:
            self.best_value = value
            self.best_x = feasible.copy()
        return value

    def points_array(self) -> np.ndarray:
        if not self.points:
            return np.empty((0, self.problem.dimension), dtype=float)
        return np.vstack(self.points)

    def values_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def finalize(
        self,
        baseline_name: str,
        seed: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> BaselineResult:
        if not self.points:
            raise RuntimeError("baseline did not evaluate the objective")
        visited_values = self.values_array()
        best_trace = np.minimum.accumulate(visited_values)
        threshold = self.best_value + self.discovery_tolerance
        discovery_index = int(np.argmax(best_trace <= threshold))
        if not np.any(best_trace <= threshold):  # pragma: no cover - never-approach fallback
            discovery_index = visited_values.shape[0] - 1
        return BaselineResult(
            baseline_name=baseline_name,
            problem_name=self.problem.name,
            seed=seed,
            solution=self.best_x.copy(),
            objective_value=float(self.best_value),
            wall_time=perf_counter() - self.start_time,
            evaluations=self.evaluations,
            budget=self.budget,
            constraint_violations=self.constraint_violations,
            constraint_violation_magnitude=float(self.constraint_violation_magnitude),
            visited_points=self.points_array(),
            visited_values=visited_values,
            best_trace=best_trace,
            time_to_discovery_seconds=float(self.times[discovery_index]),
            evaluations_to_discovery=discovery_index + 1,
            metadata={} if metadata is None else dict(metadata),
        )


def _evaluate_candidates(tracker: _EvaluationTracker, candidates: Iterable[np.ndarray]) -> list[float]:
    values: list[float] = []
    for candidate in candidates:
        if tracker.should_stop():
            break
        values.append(tracker.evaluate(candidate))
    return values


def _tournament_select(
    rng: np.random.Generator,
    population: np.ndarray,
    values: np.ndarray,
    size: int = 3,
) -> np.ndarray:
    indices = rng.integers(0, population.shape[0], size=size)
    best_index = int(indices[np.argmin(values[indices])])
    return population[best_index].copy()


def _polynomial_design_matrix(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")
    features = [np.ones((x.shape[0], 1), dtype=float), x, x**2]
    if x.shape[1] > 1:
        interactions: list[np.ndarray] = []
        for left in range(x.shape[1]):
            for right in range(left + 1, x.shape[1]):
                interactions.append((x[:, left] * x[:, right])[:, None])
                if len(interactions) >= 6:  # pragma: no cover - truncation guard
                    break
            if len(interactions) >= 6:  # pragma: no cover - truncation guard
                break
        if interactions:  # pragma: no cover - feature limit not exercised
            features.extend(interactions)
    return np.hstack(features)


def _surrogate_predict(observed_x: np.ndarray, observed_y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    design = _polynomial_design_matrix(observed_x)
    coeffs, *_ = np.linalg.lstsq(design, observed_y, rcond=None)
    candidate_design = _polynomial_design_matrix(candidates)
    return candidate_design @ coeffs


def _make_pareto_objectives(
    values: np.ndarray,
    compactness: np.ndarray,
) -> Callable[[object], tuple[float, float]]:
    def objectives(candidate_index: object) -> tuple[float, float]:
        assert isinstance(candidate_index, int)
        return float(values[candidate_index]), float(compactness[candidate_index])

    return objectives


def _compute_metrics(results: Sequence[BaselineResult]) -> BenchmarkMetrics:
    if not results:
        raise ValueError("results must contain at least one run")
    performance_values = np.asarray([item.objective_value for item in results], dtype=float)
    efficiency_values = np.asarray(
        [
            max(0.0, float(item.visited_values[0]) - item.objective_value) / max(1, item.evaluations)
            for item in results
        ],
        dtype=float,
    )
    novelty_values = np.asarray(
        [_mean_pairwise_distance(item.visited_points) for item in results],
        dtype=float,
    )
    final_solutions = np.vstack([item.solution for item in results])
    return BenchmarkMetrics(
        performance=float(np.mean(performance_values)),
        compute_efficiency=float(np.mean(efficiency_values)),
        constraint_violations=float(np.mean([item.constraint_violations for item in results])),
        novelty=float(np.mean(novelty_values)),
        robustness=float(np.var(performance_values)),
        solution_diversity=_mean_pairwise_distance(final_solutions),
        time_to_discovery_seconds=float(np.mean([item.time_to_discovery_seconds for item in results])),
        evaluations_to_discovery=float(np.mean([item.evaluations_to_discovery for item in results])),
        mean_wall_time=float(np.mean([item.wall_time for item in results])),
        mean_evaluations=float(np.mean([item.evaluations for item in results])),
        best_performance=float(np.min(performance_values)),
    )


def run_bayesian_optimization_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of Bayesian-optimization-style
    search for fair comparison purposes, not a literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    initial_count = min(max(4, problem.dimension + 1), tracker.budget)
    _evaluate_candidates(tracker, (problem.sample(rng) for _ in range(initial_count)))
    while not tracker.should_stop():
        observed_x = tracker.points_array()
        observed_y = tracker.values_array()
        local_scale = problem.span * max(0.08, 0.35 / np.sqrt(1.0 + tracker.evaluations))
        explore_count = max(8, 3 * problem.dimension)
        exploit_count = max(8, 3 * problem.dimension)
        best = tracker.best_x
        candidates = np.vstack(
            [
                rng.uniform(problem.lower, problem.upper, size=(explore_count, problem.dimension)),
                best + rng.normal(0.0, local_scale, size=(exploit_count, problem.dimension)),
            ]
        )
        candidates = np.clip(candidates, problem.lower, problem.upper)
        if observed_y.shape[0] < 3:  # pragma: no cover - tiny observation fallback
            selected = candidates[0]
        else:
            deltas = candidates[:, None, :] - observed_x[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            length_scale = max(1e-9, float(np.mean(problem.span)))
            weights = np.exp(-(distances**2) / (2.0 * length_scale**2)) + 1e-9
            means = np.sum(weights * observed_y[None, :], axis=1) / np.sum(weights, axis=1)
            variances = np.sum(weights * (observed_y[None, :] - means[:, None]) ** 2, axis=1) / np.sum(
                weights, axis=1
            )
            acquisition = means - (0.6 + 1.0 / np.sqrt(1.0 + tracker.evaluations)) * np.sqrt(variances + 1e-9)
            selected = candidates[int(np.argmin(acquisition))]
        tracker.evaluate(selected)
    return tracker.finalize("bayesian_optimization", seed, metadata={"implementation": "numpy"})


def run_evolutionary_algorithm_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of an evolutionary
    algorithm for fair comparison purposes, not a literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    population_size = min(max(4, 2 * problem.dimension), tracker.budget)
    population = rng.uniform(problem.lower, problem.upper, size=(population_size, problem.dimension))
    values = np.asarray(_evaluate_candidates(tracker, population), dtype=float)
    population = population[: values.shape[0]]
    step_scale = 0.20 * problem.span
    while not tracker.should_stop() and population.shape[0] > 0:
        elite_count = max(1, population.shape[0] // 2)
        elite_indices = np.argsort(values)[:elite_count]
        elites = population[elite_indices]
        offspring = np.empty_like(population)
        for index in range(population.shape[0]):
            parent = elites[index % elite_count]
            noise = rng.normal(0.0, step_scale, size=problem.dimension)
            offspring[index] = np.clip(parent + noise, problem.lower, problem.upper)
        offspring_values = np.asarray(_evaluate_candidates(tracker, offspring), dtype=float)
        offspring = offspring[: offspring_values.shape[0]]
        if offspring.shape[0] == 0:
            break
        combined_population = np.vstack([population, offspring])
        combined_values = np.concatenate([values, offspring_values])
        keep = np.argsort(combined_values)[: population.shape[0]]
        population = combined_population[keep]
        values = combined_values[keep]
        step_scale = np.clip(step_scale * (0.95 + 0.10 * rng.random(problem.dimension)), 1e-3, problem.span)
    return tracker.finalize("evolutionary_algorithm", seed, metadata={"implementation": "numpy"})


def run_genetic_algorithm_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of a genetic algorithm
    for fair comparison purposes, not a literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    population_size = min(max(6, 2 * problem.dimension + 2), tracker.budget)
    population = rng.uniform(problem.lower, problem.upper, size=(population_size, problem.dimension))
    values = np.asarray(_evaluate_candidates(tracker, population), dtype=float)
    population = population[: values.shape[0]]
    mutation_scale = 0.10 * problem.span
    while not tracker.should_stop() and population.shape[0] >= 2:
        children: list[np.ndarray] = []
        for _ in range(population.shape[0]):
            parent_a = _tournament_select(rng, population, values)
            parent_b = _tournament_select(rng, population, values)
            blend = rng.uniform(0.0, 1.0, size=problem.dimension)
            child = blend * parent_a + (1.0 - blend) * parent_b
            mutation_mask = rng.random(problem.dimension) < 0.30
            child = child + mutation_mask * rng.normal(0.0, mutation_scale, size=problem.dimension)
            children.append(np.clip(child, problem.lower, problem.upper))
        child_values = np.asarray(_evaluate_candidates(tracker, children), dtype=float)
        if child_values.shape[0] == 0:
            break
        child_population = np.vstack(children[: child_values.shape[0]])
        elite_count = max(1, population.shape[0] // 5)
        elite_indices = np.argsort(values)[:elite_count]
        survivors = population[elite_indices]
        survivor_values = values[elite_indices]
        merged_population = np.vstack([survivors, child_population])
        merged_values = np.concatenate([survivor_values, child_values])
        keep = np.argsort(merged_values)[: population.shape[0]]
        population = merged_population[keep]
        values = merged_values[keep]
        mutation_scale = np.clip(mutation_scale * 0.98, 1e-3, problem.span)
    return tracker.finalize("genetic_algorithm", seed, metadata={"implementation": "numpy"})


def run_simulated_annealing_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of simulated annealing
    for fair comparison purposes, not a literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    current = problem.sample(rng)
    current_value = tracker.evaluate(current)
    temperature = max(1e-6, float(np.mean(problem.span)))
    cooling = 0.95
    while not tracker.should_stop():
        scale = problem.span * max(0.02, temperature / max(1e-9, float(np.mean(problem.span))))
        candidate = current + rng.normal(0.0, scale, size=problem.dimension)
        candidate = np.clip(candidate, problem.lower, problem.upper)
        candidate_value = tracker.evaluate(candidate)
        delta = candidate_value - current_value
        if delta <= 0.0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
            current = candidate
            current_value = candidate_value
        temperature *= cooling
    return tracker.finalize("simulated_annealing", seed, metadata={"implementation": "numpy"})


def run_cma_es_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of a CMA-ES-style
    isotropic sampler for fair comparison purposes, not a literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    population_size = min(max(4, 4 + int(3 * np.log(problem.dimension + 1))), tracker.budget)
    mean = problem.sample(rng)
    sigma = max(1e-3, float(np.mean(problem.span)) * 0.25)
    best_generation = float("inf")
    while not tracker.should_stop():
        samples = mean + sigma * rng.normal(size=(population_size, problem.dimension))
        samples = np.clip(samples, problem.lower, problem.upper)
        values = np.asarray(_evaluate_candidates(tracker, samples), dtype=float)
        samples = samples[: values.shape[0]]
        if samples.shape[0] == 0:  # pragma: no cover - zero-evaluation short-circuit for budget exhaustion
            break
        elite_count = max(1, samples.shape[0] // 2)
        elite_indices = np.argsort(values)[:elite_count]
        elites = samples[elite_indices]
        mean = np.mean(elites, axis=0)
        current_best = float(np.min(values))
        sigma *= 1.05 if current_best < best_generation else 0.90
        sigma = float(np.clip(sigma, 1e-3, max(1e-3, np.max(problem.span))))
        best_generation = min(best_generation, current_best)
    return tracker.finalize("cma_es", seed, metadata={"implementation": "numpy"})


def _run_numpy_differential_evolution(
    problem: BenchmarkProblem,
    tracker: _EvaluationTracker,
    seed: int,
) -> None:
    rng = np.random.default_rng(_validate_seed(seed))
    population_size = min(max(5, 2 * problem.dimension + 1), tracker.budget)
    population = rng.uniform(problem.lower, problem.upper, size=(population_size, problem.dimension))
    values = np.asarray(_evaluate_candidates(tracker, population), dtype=float)
    population = population[: values.shape[0]]
    differential_weight = 0.8
    crossover_rate = 0.7
    while not tracker.should_stop() and population.shape[0] >= 4:
        for target_index in range(population.shape[0]):
            if tracker.should_stop():
                break
            choices = [idx for idx in range(population.shape[0]) if idx != target_index]
            a_idx, b_idx, c_idx = rng.choice(choices, size=3, replace=False)
            mutant = population[a_idx] + differential_weight * (population[b_idx] - population[c_idx])
            cross_mask = rng.random(problem.dimension) < crossover_rate
            if not np.any(cross_mask):  # pragma: no cover - ensure mutation occurs
                cross_mask[rng.integers(0, problem.dimension)] = True
            trial = np.where(cross_mask, mutant, population[target_index])
            trial = np.clip(trial, problem.lower, problem.upper)
            trial_value = tracker.evaluate(trial)
            if trial_value <= values[target_index]:
                population[target_index] = trial
                values[target_index] = trial_value


def run_differential_evolution_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of differential evolution
    for fair comparison purposes, not a literature-SOTA implementation. When
    SciPy is importable this uses ``scipy.optimize.differential_evolution``;
    otherwise it falls back to a small NumPy reference implementation."""

    tracker = _EvaluationTracker(problem, budget)
    if SCIPY_AVAILABLE and _scipy_differential_evolution is not None:
        bounds = list(problem.bounds)
        popsize = max(2, min(8, budget // max(1, problem.dimension)))
        try:
            _scipy_differential_evolution(
                lambda x: tracker.evaluate(np.asarray(x, dtype=float)),
                bounds=bounds,
                seed=_validate_seed(seed),
                maxiter=max(1, budget),
                popsize=popsize,
                polish=False,
                tol=0.0,
                atol=0.0,
                updating="deferred",
            )
        except _BudgetExhausted:
            pass
        implementation = "scipy"
    else:
        _run_numpy_differential_evolution(problem, tracker, seed)
        implementation = "numpy"
    return tracker.finalize("differential_evolution", seed, metadata={"implementation": implementation})


def _finite_difference_gradient(
    problem: BenchmarkProblem,
    tracker: _EvaluationTracker,
    x: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    grad = np.zeros(problem.dimension, dtype=float)
    for index in range(problem.dimension):
        offset = np.zeros(problem.dimension, dtype=float)
        offset[index] = epsilon
        forward = tracker.evaluate(np.clip(x + offset, problem.lower, problem.upper))
        backward = tracker.evaluate(np.clip(x - offset, problem.lower, problem.upper))
        grad[index] = (forward - backward) / (2.0 * epsilon)
    return grad


def _run_numpy_gradient_method(problem: BenchmarkProblem, tracker: _EvaluationTracker, seed: int) -> None:
    rng = np.random.default_rng(_validate_seed(seed))
    x = problem.sample(rng)
    value = tracker.evaluate(x)
    step = 0.10
    epsilon = 1e-4
    while not tracker.should_stop():
        if tracker.remaining >= 2 * problem.dimension + 1:
            grad = _finite_difference_gradient(problem, tracker, x, epsilon)
            candidate = np.clip(x - step * grad, problem.lower, problem.upper)
        else:
            candidate = np.clip(
                x + rng.normal(0.0, 0.05 * problem.span, size=problem.dimension),
                problem.lower,
                problem.upper,
            )
        candidate_value = tracker.evaluate(candidate)
        if candidate_value <= value:
            x = candidate
            value = candidate_value
            step = min(step * 1.05, 0.5)
        else:  # pragma: no cover - conservative fallback not sampled
            step = max(step * 0.7, 1e-4)


def run_gradient_method_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of a gradient method
    for fair comparison purposes, not a literature-SOTA implementation. When
    SciPy is importable this uses ``scipy.optimize.minimize``; otherwise it
    falls back to a small NumPy finite-difference descent."""

    tracker = _EvaluationTracker(problem, budget)
    if SCIPY_AVAILABLE and _scipy_minimize is not None:
        rng = np.random.default_rng(_validate_seed(seed))
        x0 = problem.sample(rng)
        try:
            _scipy_minimize(
                lambda x: tracker.evaluate(np.asarray(x, dtype=float)),
                x0=x0,
                method="L-BFGS-B",
                bounds=list(problem.bounds),
                options={"maxiter": max(1, budget), "maxfun": budget},
            )
        except _BudgetExhausted:
            pass
        implementation = "scipy"
    else:
        _run_numpy_gradient_method(problem, tracker, seed)
        implementation = "numpy"
    return tracker.finalize("gradient_method", seed, metadata={"implementation": implementation})


def run_mpc_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of a random-shooting MPC
    loop for fair comparison purposes, not a literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    current = problem.sample(rng)
    tracker.evaluate(current)
    horizon = 3
    while not tracker.should_stop():
        rollout_count = min(max(3, problem.dimension + 1), tracker.remaining)
        rollouts: list[np.ndarray] = []
        for _ in range(rollout_count):
            actions = rng.normal(0.0, 0.15 * problem.span, size=(horizon, problem.dimension))
            rollout_state = current.copy()
            for action in actions:
                rollout_state = np.clip(rollout_state + action, problem.lower, problem.upper)
            rollouts.append(rollout_state)
        values = np.asarray(_evaluate_candidates(tracker, rollouts), dtype=float)
        if values.shape[0] == 0:
            break
        current = rollouts[int(np.argmin(values))]
    return tracker.finalize("mpc", seed, metadata={"implementation": "numpy"})


def run_reinforcement_learning_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of a policy-search /
    reinforcement-learning-style optimizer for fair comparison purposes, not a
    literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    policy_mean = problem.center.copy()
    policy_std = np.maximum(0.05 * problem.span, 0.35 * problem.span)
    while not tracker.should_stop():
        batch_size = min(max(4, problem.dimension + 1), tracker.remaining)
        candidates = rng.normal(policy_mean, policy_std, size=(batch_size, problem.dimension))
        candidates = np.clip(candidates, problem.lower, problem.upper)
        values = np.asarray(_evaluate_candidates(tracker, candidates), dtype=float)
        candidates = candidates[: values.shape[0]]
        if candidates.shape[0] == 0:
            break
        elite_count = max(1, candidates.shape[0] // 2)
        elite_indices = np.argsort(values)[:elite_count]
        elites = candidates[elite_indices]
        policy_mean = np.mean(elites, axis=0)
        policy_std = np.maximum(np.std(elites, axis=0, ddof=0), 0.02 * problem.span)
    return tracker.finalize("reinforcement_learning", seed, metadata={"implementation": "numpy"})


def run_symbolic_regression_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of a symbolic-regression-style
    surrogate search for fair comparison purposes, not a literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    initial_count = min(max(6, problem.dimension + 2), tracker.budget)
    _evaluate_candidates(tracker, (problem.sample(rng) for _ in range(initial_count)))
    while not tracker.should_stop():
        observed_x = tracker.points_array()
        observed_y = tracker.values_array()
        pool_size = max(8, 3 * problem.dimension)
        pool = np.vstack(
            [
                rng.uniform(problem.lower, problem.upper, size=(pool_size, problem.dimension)),
                tracker.best_x
                + rng.normal(0.0, 0.12 * problem.span, size=(pool_size, problem.dimension)),
            ]
        )
        pool = np.clip(pool, problem.lower, problem.upper)
        predicted = _surrogate_predict(observed_x, observed_y, pool)
        candidate = pool[int(np.argmin(predicted))]
        tracker.evaluate(candidate)
    return tracker.finalize("symbolic_regression", seed, metadata={"implementation": "numpy"})


def run_multi_objective_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Minimal, budget-matched reference implementation of a multi-objective search
    loop for fair comparison purposes, not a literature-SOTA implementation."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    population_size = min(max(6, 2 * problem.dimension), tracker.budget)
    population = rng.uniform(problem.lower, problem.upper, size=(population_size, problem.dimension))
    values = np.asarray(_evaluate_candidates(tracker, population), dtype=float)
    population = population[: values.shape[0]]
    center = problem.center
    while not tracker.should_stop() and population.shape[0] > 0:
        children = np.clip(
            population + rng.normal(0.0, 0.12 * problem.span, size=population.shape),
            problem.lower,
            problem.upper,
        )
        child_values = np.asarray(_evaluate_candidates(tracker, children), dtype=float)
        children = children[: child_values.shape[0]]
        if children.shape[0] == 0:
            break
        all_points = np.vstack([population, children])
        all_values = np.concatenate([values, child_values])
        compactness = np.sum(((all_points - center) / np.maximum(problem.span, 1e-12)) ** 2, axis=1)
        candidates: list[int] = list(range(all_points.shape[0]))
        front = pareto_front(candidates, _make_pareto_objectives(all_values, compactness))
        ranked_front = sorted(front, key=lambda idx: (all_values[idx], compactness[idx]))
        if len(ranked_front) < population.shape[0]:
            leftovers = [idx for idx in candidates if idx not in set(ranked_front)]
            leftovers.sort(key=lambda idx: (all_values[idx] + 0.05 * compactness[idx], compactness[idx]))
            ranked_front.extend(leftovers[: population.shape[0] - len(ranked_front)])
        keep = ranked_front[: population.shape[0]]
        population = all_points[keep]
        values = all_values[keep]
    return tracker.finalize("multi_objective", seed, metadata={"implementation": "numpy"})


def run_qes_baseline(problem: BenchmarkProblem, budget: int, seed: int) -> BaselineResult:
    """Run the real QES search entry point ``qes.intelligence.optimize`` under an
    exact evaluation budget using the same counted objective wrapper as the
    baselines above."""

    tracker = _EvaluationTracker(problem, budget)
    rng = np.random.default_rng(_validate_seed(seed))
    seed_room = Room(
        x=problem.sample(rng),
        x_star=problem.center if problem.optimum_x is None else problem.optimum_x.copy(),
        lower=problem.lower,
        upper=problem.upper,
        activation=np.ones(problem.dimension, dtype=float),
    )
    population = max(2, min(6, budget // max(2, problem.dimension + 2)))
    config = AdaptiveSearchConfig(
        step_size=0.35,
        learning_rate=0.04,
        gradient_prob=0.55,
        stagnation_patience=max(4, 2 * problem.dimension),
    )
    try:
        optimize(
            tracker.evaluate,
            seed_room,
            iterations=max(1, budget),
            population=max(1, population),
            branch_scale=0.25,
            config=config,
            rng=rng,
        )
    except _BudgetExhausted:
        pass
    return tracker.finalize(
        "qes",
        seed,
        metadata={"implementation": "qes.intelligence.optimize", "population": max(1, population)},
    )


BASELINE_REGISTRY: dict[str, BaselineRunner] = {
    "qes": run_qes_baseline,
    "bayesian_optimization": run_bayesian_optimization_baseline,
    "evolutionary_algorithm": run_evolutionary_algorithm_baseline,
    "genetic_algorithm": run_genetic_algorithm_baseline,
    "simulated_annealing": run_simulated_annealing_baseline,
    "cma_es": run_cma_es_baseline,
    "differential_evolution": run_differential_evolution_baseline,
    "gradient_method": run_gradient_method_baseline,
    "mpc": run_mpc_baseline,
    "reinforcement_learning": run_reinforcement_learning_baseline,
    "symbolic_regression": run_symbolic_regression_baseline,
    "multi_objective": run_multi_objective_baseline,
}


def run_qes_bench(
    problems: Sequence[BenchmarkProblem] | None = None,
    baselines: Sequence[str] | None = None,
    *,
    budget: int = 300,
    seeds: Sequence[int] = (0, 1, 2),
) -> BenchReport:
    """Run QES and the requested baselines across benchmark problems.

    Args:
        problems: benchmark problems to evaluate. Defaults to Sphere, Rastrigin,
            and Rosenbrock.
        baselines: requested non-QES baselines by name. ``qes`` is always added.
        budget: identical function-evaluation budget per run.
        seeds: run identifiers mixed with each problem's fixed seed.

    Returns:
        A :class:`BenchReport` containing per-run results and aggregated metrics.
    """

    budget = _validate_budget(budget)
    selected_problems = tuple(DEFAULT_BENCHMARK_PROBLEMS if problems is None else problems)
    if not selected_problems:
        raise ValueError("problems must contain at least one problem")
    for problem in selected_problems:
        if not isinstance(problem, BenchmarkProblem):
            raise TypeError("problems must contain BenchmarkProblem instances")
    selected_seeds = tuple(_validate_seed(seed) for seed in seeds)
    if not selected_seeds:
        raise ValueError("seeds must contain at least one seed")
    requested = list(BASELINE_REGISTRY.keys())[1:] if baselines is None else list(baselines)
    if not requested:
        raise ValueError("baselines must contain at least one baseline name")
    baseline_names = ["qes"]
    for name in requested:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("baseline names must be non-empty strings")
        normalized = name.strip().lower()
        if normalized not in BASELINE_REGISTRY:
            raise ValueError(f"unknown baseline name: {name}")
        if normalized not in baseline_names:
            baseline_names.append(normalized)

    runs: dict[str, dict[str, list[BaselineResult]]] = {}
    metrics: dict[str, dict[str, BenchmarkMetrics]] = {}
    for problem in selected_problems:
        problem_runs: dict[str, list[BaselineResult]] = {}
        problem_metrics: dict[str, BenchmarkMetrics] = {}
        for baseline_name in baseline_names:
            runner = BASELINE_REGISTRY[baseline_name]
            results = [
                runner(problem, budget, _mix_seed(problem.seed, seed))
                for seed in selected_seeds
            ]
            problem_runs[baseline_name] = results
            problem_metrics[baseline_name] = _compute_metrics(results)
        runs[problem.name] = problem_runs
        metrics[problem.name] = problem_metrics
    return BenchReport(
        problems=selected_problems,
        baseline_names=tuple(baseline_names),
        budget=budget,
        seeds=selected_seeds,
        runs=runs,
        metrics=metrics,
    )


__all__ = [
    "BASELINE_REGISTRY",
    "BenchmarkMetrics",
    "BenchmarkProblem",
    "BaselineResult",
    "BenchReport",
    "DEFAULT_BENCHMARK_PROBLEMS",
    "RASTRIGIN_PROBLEM",
    "ROSENBROCK_PROBLEM",
    "SCIPY_AVAILABLE",
    "SPHERE_PROBLEM",
    "make_rastrigin_problem",
    "make_rosenbrock_problem",
    "make_sphere_problem",
    "run_bayesian_optimization_baseline",
    "run_cma_es_baseline",
    "run_differential_evolution_baseline",
    "run_evolutionary_algorithm_baseline",
    "run_genetic_algorithm_baseline",
    "run_gradient_method_baseline",
    "run_mpc_baseline",
    "run_multi_objective_baseline",
    "run_qes_baseline",
    "run_qes_bench",
    "run_reinforcement_learning_baseline",
    "run_simulated_annealing_baseline",
    "run_symbolic_regression_baseline",
]
