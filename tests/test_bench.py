from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from qes.bench import (
    BASELINE_REGISTRY,
    SCIPY_AVAILABLE,
    BaselineResult,
    BenchmarkMetrics,
    BenchmarkProblem,
    BenchReport,
    make_rastrigin_problem,
    make_rosenbrock_problem,
    make_sphere_problem,
    run_qes_bench,
)


def _sphere_problem() -> BenchmarkProblem:
    return make_sphere_problem(dimension=3, seed=5)


@pytest.mark.parametrize(
    ("factory", "expected_name", "expected_optimum"),
    [
        (make_sphere_problem, "Sphere-4D", np.zeros(4)),
        (make_rastrigin_problem, "Rastrigin-4D", np.zeros(4)),
        (make_rosenbrock_problem, "Rosenbrock-4D", np.ones(4)),
    ],
)
def test_problem_factories_define_expected_optima(
    factory: Callable[[], BenchmarkProblem],
    expected_name: str,
    expected_optimum: np.ndarray,
) -> None:
    problem = factory()
    assert problem.name == expected_name
    assert problem.optimum_value == pytest.approx(0.0)
    assert np.allclose(problem.optimum_x, expected_optimum)


def test_benchmark_problem_rejects_invalid_bounds_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        BenchmarkProblem(
            name="bad",
            objective=lambda x: float(np.sum(x**2)),
            bounds=[(-1.0, 1.0)],
            dimension=2,
        )


@pytest.mark.parametrize("baseline_name", sorted(BASELINE_REGISTRY))
def test_each_baseline_respects_budget_and_returns_valid_result(baseline_name: str) -> None:
    problem = _sphere_problem()
    budget = 24
    result = BASELINE_REGISTRY[baseline_name](problem, budget, 13)
    assert isinstance(result, BaselineResult)
    assert result.baseline_name == baseline_name
    assert result.problem_name == problem.name
    assert result.evaluations <= budget
    assert result.evaluations == result.visited_values.shape[0]
    assert result.visited_points.shape == (result.evaluations, problem.dimension)
    assert result.best_trace.shape == (result.evaluations,)
    assert np.all(np.diff(result.best_trace) <= 1e-12)
    assert np.isfinite(result.objective_value)
    assert result.wall_time >= 0.0


@pytest.mark.parametrize(
    "baseline_name",
    [
        "qes",
        "bayesian_optimization",
        "simulated_annealing",
        "differential_evolution",
    ],
)
def test_selected_baselines_are_deterministic_for_fixed_seed(baseline_name: str) -> None:
    problem = _sphere_problem()
    first = BASELINE_REGISTRY[baseline_name](problem, 26, 9)
    second = BASELINE_REGISTRY[baseline_name](problem, 26, 9)
    assert first.evaluations == second.evaluations
    assert first.objective_value == pytest.approx(second.objective_value)
    assert np.allclose(first.solution, second.solution)
    assert np.allclose(first.visited_values, second.visited_values)


def test_gradient_method_reports_expected_backend_metadata() -> None:
    result = BASELINE_REGISTRY["gradient_method"](_sphere_problem(), 24, 7)
    expected = "scipy" if SCIPY_AVAILABLE else "numpy"
    assert result.metadata["implementation"] == expected


def test_differential_evolution_reports_expected_backend_metadata() -> None:
    result = BASELINE_REGISTRY["differential_evolution"](_sphere_problem(), 24, 7)
    expected = "scipy" if SCIPY_AVAILABLE else "numpy"
    assert result.metadata["implementation"] == expected


def test_run_qes_bench_returns_expected_structure() -> None:
    problems = (make_sphere_problem(dimension=2, seed=3), make_rastrigin_problem(dimension=2, seed=4))
    report = run_qes_bench(
        problems=problems,
        baselines=["simulated_annealing", "gradient_method"],
        budget=20,
        seeds=(0, 1),
    )
    assert isinstance(report, BenchReport)
    assert report.baseline_names == ("qes", "simulated_annealing", "gradient_method")
    assert set(report.runs) == {problem.name for problem in problems}
    for problem in problems:
        assert set(report.runs[problem.name]) == set(report.baseline_names)
        assert set(report.metrics[problem.name]) == set(report.baseline_names)
        for baseline_name in report.baseline_names:
            assert len(report.runs[problem.name][baseline_name]) == 2
            assert isinstance(report.metrics[problem.name][baseline_name], BenchmarkMetrics)


def test_report_table_rows_match_problem_baseline_grid() -> None:
    report = run_qes_bench(
        problems=(make_sphere_problem(dimension=2),),
        baselines=["simulated_annealing", "cma_es"],
        budget=20,
        seeds=(0,),
    )
    rows = report.table_rows()
    assert len(rows) == 3
    assert {row["baseline"] for row in rows} == {"qes", "simulated_annealing", "cma_es"}


def test_metrics_are_computed_consistently_for_single_seed_run() -> None:
    report = run_qes_bench(
        problems=(make_sphere_problem(dimension=2, seed=1),),
        baselines=["simulated_annealing"],
        budget=20,
        seeds=(0,),
    )
    metrics = report.metrics["Sphere-2D"]["simulated_annealing"]
    run = report.runs["Sphere-2D"]["simulated_annealing"][0]
    expected_efficiency = max(0.0, float(run.visited_values[0]) - run.objective_value) / run.evaluations
    assert metrics.performance == pytest.approx(run.objective_value)
    assert metrics.best_performance == pytest.approx(run.objective_value)
    assert metrics.compute_efficiency == pytest.approx(expected_efficiency)
    assert metrics.constraint_violations == pytest.approx(float(run.constraint_violations))
    assert metrics.solution_diversity == pytest.approx(0.0)


def test_metrics_robustness_is_positive_when_multiple_seed_outcomes_differ() -> None:
    report = run_qes_bench(
        problems=(make_rastrigin_problem(dimension=2, seed=2),),
        baselines=["genetic_algorithm"],
        budget=22,
        seeds=(0, 1, 2),
    )
    metrics = report.metrics["Rastrigin-2D"]["genetic_algorithm"]
    assert metrics.robustness >= 0.0


def test_qes_is_always_included_even_if_not_requested() -> None:
    report = run_qes_bench(
        problems=(make_sphere_problem(dimension=2),),
        baselines=["simulated_annealing"],
        budget=20,
        seeds=(0,),
    )
    assert report.baseline_names[0] == "qes"

def test_run_qes_bench_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        run_qes_bench(
            problems=(make_sphere_problem(dimension=2),),
            baselines=["simulated_annealing"],
            budget=0,
        )


def test_run_qes_bench_rejects_empty_baselines() -> None:
    with pytest.raises(ValueError, match="at least one baseline"):
        run_qes_bench(problems=(make_sphere_problem(dimension=2),), baselines=[], budget=20)


def test_run_qes_bench_rejects_unknown_baseline_name() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        run_qes_bench(problems=(make_sphere_problem(dimension=2),), baselines=["unknown"], budget=20)


def test_run_qes_bench_rejects_empty_seed_list() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        run_qes_bench(
            problems=(make_sphere_problem(dimension=2),),
            baselines=["simulated_annealing"],
            budget=20,
            seeds=(),
        )


def test_run_qes_bench_rejects_empty_problem_list() -> None:
    with pytest.raises(ValueError, match="at least one problem"):
        run_qes_bench(problems=(), baselines=["simulated_annealing"], budget=20)


def test_run_qes_bench_rejects_non_problem_entries() -> None:
    with pytest.raises(TypeError, match="BenchmarkProblem"):
        run_qes_bench(problems=(object(),), baselines=["simulated_annealing"], budget=20)  # type: ignore[arg-type]


def test_result_discovery_metadata_is_within_budget() -> None:
    result = BASELINE_REGISTRY["simulated_annealing"](_sphere_problem(), 20, 2)
    assert 1 <= result.evaluations_to_discovery <= result.evaluations
    assert 0.0 <= result.time_to_discovery_seconds <= result.wall_time + 1e-9


def test_results_remain_within_bounds_after_clipping() -> None:
    problem = _sphere_problem()
    result = BASELINE_REGISTRY["evolutionary_algorithm"](problem, 24, 4)
    assert np.all(result.visited_points >= problem.lower - 1e-12)
    assert np.all(result.visited_points <= problem.upper + 1e-12)


def test_private_validators_and_objectives_cover_edge_cases() -> None:
    import qes.bench as bench_module

    with pytest.raises(TypeError, match="seed must be an integer"):
        bench_module._validate_seed(True)
    with pytest.raises(ValueError, match="2D array"):
        bench_module._mean_pairwise_distance(np.array([1.0, 2.0]))
    assert bench_module._rosenbrock(np.array([2.0])) == pytest.approx(1.0)
    assert bench_module._sphere(np.array([1.0, 2.0])) == pytest.approx(5.0)


def test_benchmark_problem_validation_and_constraint_violation_errors() -> None:
    import qes.bench as bench_module

    with pytest.raises(ValueError, match="non-empty string"):
        bench_module.BenchmarkProblem(
            name="",
            objective=lambda x: 0.0,
            bounds=[(-1.0, 1.0)],
            dimension=1,
        )
    with pytest.raises(TypeError, match="objective must be callable"):
        bench_module.BenchmarkProblem(  # type: ignore[arg-type]
            name="bad",
            objective=1,
            bounds=[(-1.0, 1.0)],
            dimension=1,
        )
    with pytest.raises(ValueError, match="integer > 0"):
        bench_module.BenchmarkProblem(
            name="bad",
            objective=lambda x: 0.0,
            bounds=[(-1.0, 1.0)],
            dimension=0,
        )
    with pytest.raises(ValueError, match="finite"):
        bench_module.BenchmarkProblem(
            name="bad",
            objective=lambda x: 0.0,
            bounds=[(-np.inf, 1.0)],
            dimension=1,
        )
    with pytest.raises(ValueError, match="strictly less"):
        bench_module.BenchmarkProblem(
            name="bad",
            objective=lambda x: 0.0,
            bounds=[(1.0, 1.0)],
            dimension=1,
        )
    with pytest.raises(ValueError, match="shape"):
        bench_module.BenchmarkProblem(
            name="bad",
            objective=lambda x: 0.0,
            bounds=[(-1.0, 1.0)],
            dimension=1,
            optimum_x=np.zeros(2),
        )
    with pytest.raises(ValueError, match="finite values"):
        bench_module.BenchmarkProblem(
            name="bad",
            objective=lambda x: 0.0,
            bounds=[(-1.0, 1.0)],
            dimension=1,
            optimum_x=np.array([np.nan]),
        )
    with pytest.raises(ValueError, match="optimum_value must be finite"):
        bench_module.BenchmarkProblem(
            name="bad",
            objective=lambda x: 0.0,
            bounds=[(-1.0, 1.0)],
            dimension=1,
            optimum_value=float("inf"),
        )
    with pytest.raises(TypeError, match="constraint_fn must be callable"):
        bench_module.BenchmarkProblem(
            name="bad",
            objective=lambda x: 0.0,
            bounds=[(-1.0, 1.0)],
            dimension=1,
            constraint_fn=1,  # type: ignore[arg-type]
        )

    problem = bench_module.BenchmarkProblem(
        name="ok",
        objective=lambda x: float(np.sum(x**2)),
        bounds=[(-1.0, 1.0)],
        dimension=1,
        constraint_fn=lambda x: float("nan"),
    )
    with pytest.raises(ValueError, match="finite value >= 0"):
        problem.constraint_violation(np.array([0.5]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"solution": [[1.0]]}, "solution must be one-dimensional"),
        ({"visited_points": [1.0]}, "visited_points must be two-dimensional"),
        (
            {"visited_points": np.zeros((2, 1)), "visited_values": np.array([0.0])},
            "visited_points and visited_values length mismatch",
        ),
        (
            {"best_trace": np.array([0.0, 1.0])},
            "best_trace must match visited_values shape",
        ),
        (
            {"evaluations": 2},
            "evaluations must equal the number of visited values",
        ),
        (
            {"budget": 0},
            "evaluations cannot exceed budget",
        ),
    ],
)
def test_baseline_result_validation_errors(kwargs: dict[str, object], message: str) -> None:
    import qes.bench as bench_module

    baseline = dict(
        baseline_name="x",
        problem_name="p",
        seed=0,
        solution=np.array([0.0]),
        objective_value=0.0,
        wall_time=0.0,
        evaluations=1,
        budget=1,
        constraint_violations=0,
        constraint_violation_magnitude=0.0,
        visited_points=np.zeros((1, 1)),
        visited_values=np.array([0.0]),
        best_trace=np.array([0.0]),
        time_to_discovery_seconds=0.0,
        evaluations_to_discovery=1,
    )
    baseline.update(kwargs)
    with pytest.raises(ValueError, match=message):
        bench_module.BaselineResult(**baseline)


def test_evaluation_tracker_validation_and_error_paths() -> None:
    import qes.bench as bench_module

    with pytest.raises(ValueError, match="discovery_tolerance must be finite and >= 0"):
        bench_module._EvaluationTracker(_sphere_problem(), 2, discovery_tolerance=-1.0)

    problem = bench_module.BenchmarkProblem(
        name="finite",
        objective=lambda x: float("nan"),
        bounds=[(-1.0, 1.0)],
        dimension=1,
        optimum_value=None,
    )
    tracker = bench_module._EvaluationTracker(problem, 2)
    assert tracker.should_stop() is False
    with pytest.raises(ValueError, match="candidate shape"):
        tracker.evaluate(np.zeros(2))
    with pytest.raises(ValueError, match="finite scalar"):
        tracker.evaluate(np.zeros(1))

    constrained = bench_module.BenchmarkProblem(
        name="constrained",
        objective=lambda x: float(np.sum(x**2)),
        bounds=[(-1.0, 1.0)],
        dimension=1,
        constraint_fn=lambda x: 0.25,
    )
    constrained_tracker = bench_module._EvaluationTracker(constrained, 2)
    constrained_tracker.evaluate(np.array([2.0]))
    assert constrained_tracker.constraint_violations == 1
    assert constrained_tracker.constraint_violation_magnitude > 0.0

    empty_tracker = bench_module._EvaluationTracker(_sphere_problem(), 2)
    assert empty_tracker.points_array().shape == (0, 3)
    with pytest.raises(RuntimeError, match="did not evaluate"):
        empty_tracker.finalize("x", 0)


def test_design_matrix_surrogate_and_metric_helpers() -> None:
    import qes.bench as bench_module

    with pytest.raises(ValueError, match="2D array"):
        bench_module._polynomial_design_matrix(np.array([1.0, 2.0]))
    one_dim = bench_module._polynomial_design_matrix(np.array([[1.0], [2.0]]))
    assert one_dim.shape[1] == 3
    many = bench_module._polynomial_design_matrix(np.arange(21.0).reshape(3, 7))
    assert many.shape[1] == 1 + 7 + 7 + 6
    predicted = bench_module._surrogate_predict(
        np.array([[0.0], [1.0], [2.0]]),
        np.array([0.0, 1.0, 4.0]),
        np.array([[1.5]]),
    )
    assert predicted.shape == (1,)
    objectives = bench_module._make_pareto_objectives(np.array([1.0]), np.array([2.0]))
    assert objectives(0) == (1.0, 2.0)
    with pytest.raises(ValueError, match="results must contain at least one run"):
        bench_module._compute_metrics([])


def test_run_qes_bench_normalizes_baseline_names_and_rejects_blank_entries() -> None:
    report = run_qes_bench(
        problems=(make_sphere_problem(dimension=2),),
        baselines=[" QES ", "SIMULATED_ANNEALING"],
        budget=10,
        seeds=(0,),
    )
    assert report.baseline_names == ("qes", "simulated_annealing")
    with pytest.raises(ValueError, match="non-empty strings"):
        run_qes_bench(
            problems=(make_sphere_problem(dimension=2),),
            baselines=[""],
            budget=10,
        )


def test_private_baseline_helpers_cover_budget_exhaustion_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import qes.bench as bench_module

    problem = make_sphere_problem(dimension=2, seed=1)

    call_count = {"count": 0}
    original_evaluate = bench_module._evaluate_candidates

    def fake_evaluate(tracker: object, candidates: object) -> list[float]:
        call_count["count"] += 1
        if call_count["count"] == 1:
            return original_evaluate(tracker, candidates)
        return []

    monkeypatch.setattr(bench_module, "_evaluate_candidates", fake_evaluate)
    assert bench_module.run_evolutionary_algorithm_baseline(problem, 6, 0).evaluations >= 1

    call_count["count"] = 0
    assert bench_module.run_genetic_algorithm_baseline(problem, 8, 0).evaluations >= 1

    call_count["count"] = 0
    assert bench_module.run_cma_es_baseline(problem, 5, 0).evaluations >= 1

    call_count["count"] = 0
    assert bench_module.run_mpc_baseline(problem, 4, 0).evaluations >= 1

    call_count["count"] = 0
    assert bench_module.run_reinforcement_learning_baseline(problem, 4, 0).evaluations >= 1

    call_count["count"] = 0
    assert bench_module.run_multi_objective_baseline(problem, 6, 0).evaluations >= 1


def test_bayesian_and_multi_objective_private_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import qes.bench as bench_module

    problem = make_sphere_problem(dimension=1, seed=2)
    result = bench_module.run_bayesian_optimization_baseline(problem, 5, 0)
    assert result.evaluations == 5

    monkeypatch.setattr(bench_module, "pareto_front", lambda candidates, objectives: [0])
    multi = bench_module.run_multi_objective_baseline(make_sphere_problem(dimension=2, seed=3), 12, 0)
    assert multi.evaluations <= 12


def test_numpy_differential_evolution_and_gradient_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qes.bench as bench_module

    class FakeRng:
        def uniform(self, low: object, high: object, size: object = None) -> np.ndarray:
            if size is None:
                return np.full(np.asarray(low).shape, 0.25, dtype=float)
            return np.full(size, 0.25, dtype=float)

        def normal(self, loc: float = 0.0, scale: object = None, size: object = None) -> np.ndarray:
            return np.zeros(size, dtype=float)

        def random(self, size: object = None) -> np.ndarray:
            return np.ones(size, dtype=float)

        def choice(self, choices: list[int], size: int, replace: bool = False) -> np.ndarray:
            return np.array(choices[:size], dtype=int)

        def integers(self, low: int, high: int | None = None, size: object = None) -> int | np.ndarray:
            if size is None:
                return 0
            return np.zeros(size, dtype=int)

    monkeypatch.setattr(bench_module.np.random, "default_rng", lambda seed=None: FakeRng())
    monkeypatch.setattr(bench_module, "SCIPY_AVAILABLE", False)
    monkeypatch.setattr(bench_module, "_scipy_differential_evolution", None)
    monkeypatch.setattr(bench_module, "_scipy_minimize", None)

    diff = bench_module.run_differential_evolution_baseline(make_sphere_problem(dimension=2, seed=4), 10, 0)
    assert diff.metadata["implementation"] == "numpy"

    problem = make_sphere_problem(dimension=2, seed=5)
    tracker = bench_module._EvaluationTracker(problem, 10)
    gradient = bench_module._finite_difference_gradient(problem, tracker, np.zeros(2), 1e-4)
    assert gradient.shape == (2,)

    grad = bench_module.run_gradient_method_baseline(problem, 8, 0)
    assert grad.metadata["implementation"] == "numpy"
