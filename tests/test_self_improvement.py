from __future__ import annotations

from dataclasses import dataclass

import pytest

import qes.self_improvement as self_improvement_module
from qes.self_improvement import (
    ApprovalPolicy,
    ArchitectureProposal,
    BenchmarkResult,
    PerformanceEvaluation,
    RegressionTestResult,
    SafetyTestResult,
    Sandbox,
    SelfImprovementPipeline,
    default_approval_policy,
    propose_alternatives,
    run_in_sandbox,
    run_regression_tests,
    run_safety_tests,
)


def make_strategy(**overrides: object) -> dict[str, object]:
    strategy: dict[str, object] = {
        "mutation_rate": 0.30,
        "population_size": 40,
        "elite_count": 2,
        "label": "baseline",
    }
    strategy.update(overrides)
    return strategy


def make_performance(score: float = 0.45) -> PerformanceEvaluation:
    return PerformanceEvaluation(score=score, convergence_speed=0.6, resource_cost=1.0)


def benchmark(strategy: dict[str, object]) -> dict[str, object]:
    mutation = float(strategy["mutation_rate"])
    population = float(strategy["population_size"])
    elite = float(strategy["elite_count"])
    score = 1.2 - abs(mutation - 0.18) * 3.0 - abs(population - 72.0) * 0.01 - abs(elite - 3.0) * 0.1
    return {
        "score": score,
        "convergence_speed": max(0.0, 1.0 - abs(mutation - 0.18)),
        "resource_cost": population * 0.02 + elite * 0.05,
        "details": {"mutation_rate": mutation, "population_size": population, "elite_count": elite},
    }


def population_safety_check(proposal: ArchitectureProposal) -> tuple[bool, str]:
    population = float(proposal.strategy["population_size"])
    if population > 120.0:
        return False, "population_size exceeds safe envelope"
    return True, ""


def mutation_safety_check(proposal: ArchitectureProposal) -> str | None:
    mutation = float(proposal.strategy["mutation_rate"])
    if not 0.05 <= mutation <= 0.50:
        return "mutation_rate outside safe operating range"
    return None


class TestDataclassesAndValidation:
    def test_private_validation_helpers_cover_error_paths(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            self_improvement_module._validate_nonempty_string("name", 1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be a mapping"):
            self_improvement_module._validate_mapping("strategy", 1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be an integer"):
            self_improvement_module._validate_positive_int("n", True)
        with pytest.raises(TypeError, match="must be a real number"):
            self_improvement_module._validate_finite_float("score", True)
        assert self_improvement_module._restore_numeric_type(True, 0.0) is True

    def test_proposal_reason_covers_rejection_paths(self) -> None:
        benchmark_result = BenchmarkResult("ARCH-R", score=0.3)
        safety_result = SafetyTestResult("ARCH-R", passed=True)
        regressed = RegressionTestResult(
            "ARCH-R",
            passed=False,
            baseline_score=0.4,
            candidate_score=0.3,
            score_delta=-0.1,
            message="regressed",
        )
        assert "rejected by regression gate" in self_improvement_module._proposal_reason(
            benchmark_result, safety_result, regressed, approved=False
        )

        non_approved = RegressionTestResult(
            "ARCH-R",
            passed=True,
            baseline_score=0.4,
            candidate_score=0.41,
            score_delta=0.01,
            message="ok",
        )
        assert "rejected by approval policy" in self_improvement_module._proposal_reason(
            benchmark_result, safety_result, non_approved, approved=False
        )

    def test_performance_evaluation_rejects_non_finite_score(self) -> None:
        with pytest.raises(ValueError, match="score must be finite"):
            PerformanceEvaluation(score=float("inf"))

    def test_performance_evaluation_rejects_non_dict_details(self) -> None:
        with pytest.raises(TypeError, match="details must be a dict"):
            PerformanceEvaluation(score=1.0, details=["bad"])  # type: ignore[arg-type]

    def test_architecture_proposal_copies_strategy(self) -> None:
        strategy = make_strategy()
        proposal = ArchitectureProposal("ARCH-1", strategy, "test rationale")
        strategy["mutation_rate"] = 0.99
        assert proposal.strategy["mutation_rate"] == 0.30

    def test_architecture_proposal_rejects_non_dict_strategy(self) -> None:
        with pytest.raises(TypeError, match="strategy must be a dict"):
            ArchitectureProposal("ARCH-0", make_strategy().items(), "rationale")  # type: ignore[arg-type]

    def test_benchmark_result_requires_non_empty_id(self) -> None:
        with pytest.raises(ValueError, match="proposal_id"):
            BenchmarkResult(proposal_id="", score=1.0)

    def test_benchmark_and_result_dataclasses_reject_invalid_types(self) -> None:
        with pytest.raises(TypeError, match="details must be a dict"):
            BenchmarkResult(proposal_id="ARCH-X", score=1.0, details=["bad"])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="passed must be a bool"):
            SafetyTestResult("ARCH-X", passed=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="details must be a dict"):
            SafetyTestResult("ARCH-X", passed=True, details=["bad"])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="passed must be a bool"):
            RegressionTestResult(
                "ARCH-X",
                passed=1,  # type: ignore[arg-type]
                baseline_score=0.0,
                candidate_score=0.0,
                score_delta=0.0,
                message="msg",
            )

    def test_proposal_evaluation_and_report_reject_invalid_types(self) -> None:
        proposal = ArchitectureProposal("ARCH-E", make_strategy(), "reason")
        benchmark_result = BenchmarkResult("ARCH-E", score=1.0)
        safety_result = SafetyTestResult("ARCH-E", passed=True)
        regression_result = RegressionTestResult(
            "ARCH-E",
            passed=True,
            baseline_score=0.4,
            candidate_score=1.0,
            score_delta=0.6,
            message="ok",
        )

        with pytest.raises(TypeError, match="proposal must be an ArchitectureProposal"):
            self_improvement_module.ProposalEvaluation(
                proposal="bad",  # type: ignore[arg-type]
                benchmark_result=benchmark_result,
                safety_result=safety_result,
                regression_result=regression_result,
                approved=True,
            )
        with pytest.raises(TypeError, match="benchmark_result must be a BenchmarkResult"):
            self_improvement_module.ProposalEvaluation(
                proposal=proposal,
                benchmark_result="bad",  # type: ignore[arg-type]
                safety_result=safety_result,
                regression_result=regression_result,
                approved=True,
            )
        with pytest.raises(TypeError, match="safety_result must be a SafetyTestResult"):
            self_improvement_module.ProposalEvaluation(
                proposal=proposal,
                benchmark_result=benchmark_result,
                safety_result="bad",  # type: ignore[arg-type]
                regression_result=regression_result,
                approved=True,
            )
        with pytest.raises(TypeError, match="regression_result must be a RegressionTestResult"):
            self_improvement_module.ProposalEvaluation(
                proposal=proposal,
                benchmark_result=benchmark_result,
                safety_result=safety_result,
                regression_result="bad",  # type: ignore[arg-type]
                approved=True,
            )
        with pytest.raises(TypeError, match="approved must be a bool"):
            self_improvement_module.ProposalEvaluation(
                proposal=proposal,
                benchmark_result=benchmark_result,
                safety_result=safety_result,
                regression_result=regression_result,
                approved=1,  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="deployed must be a bool"):
            self_improvement_module.ProposalEvaluation(
                proposal=proposal,
                benchmark_result=benchmark_result,
                safety_result=safety_result,
                regression_result=regression_result,
                approved=True,
                deployed=1,  # type: ignore[arg-type]
            )
        assert self_improvement_module.ProposalEvaluation(
            proposal=proposal,
            benchmark_result=benchmark_result,
            safety_result=safety_result,
            regression_result=regression_result,
            approved=True,
        ).decision_reason == ""

        evaluation = self_improvement_module.ProposalEvaluation(
            proposal=proposal,
            benchmark_result=benchmark_result,
            safety_result=safety_result,
            regression_result=regression_result,
            approved=True,
            decision_reason="approved",
        )
        assert evaluation.decision_reason == "approved"

        with pytest.raises(TypeError, match="current_strategy must be a dict"):
            self_improvement_module.SelfImprovementReport(
                current_strategy="bad",  # type: ignore[arg-type]
                baseline_performance=make_performance(),
                proposals=[proposal],
                evaluations=[evaluation],
                approved_proposal=proposal,
                deployed_strategy=make_strategy(),
                active_strategy=make_strategy(),
                strategy_library_snapshot=[make_strategy()],
                summary="summary",
            )
        with pytest.raises(TypeError, match="baseline_performance must be a PerformanceEvaluation"):
            self_improvement_module.SelfImprovementReport(
                current_strategy=make_strategy(),
                baseline_performance="bad",  # type: ignore[arg-type]
                proposals=[proposal],
                evaluations=[evaluation],
                approved_proposal=proposal,
                deployed_strategy=make_strategy(),
                active_strategy=make_strategy(),
                strategy_library_snapshot=[make_strategy()],
                summary="summary",
            )
        with pytest.raises(TypeError, match="proposals must contain only ArchitectureProposal"):
            self_improvement_module.SelfImprovementReport(
                current_strategy=make_strategy(),
                baseline_performance=make_performance(),
                proposals=["bad"],  # type: ignore[list-item]
                evaluations=[evaluation],
                approved_proposal=proposal,
                deployed_strategy=make_strategy(),
                active_strategy=make_strategy(),
                strategy_library_snapshot=[make_strategy()],
                summary="summary",
            )
        with pytest.raises(TypeError, match="evaluations must contain only ProposalEvaluation"):
            self_improvement_module.SelfImprovementReport(
                current_strategy=make_strategy(),
                baseline_performance=make_performance(),
                proposals=[proposal],
                evaluations=["bad"],  # type: ignore[list-item]
                approved_proposal=proposal,
                deployed_strategy=make_strategy(),
                active_strategy=make_strategy(),
                strategy_library_snapshot=[make_strategy()],
                summary="summary",
            )
        with pytest.raises(TypeError, match="approved_proposal must be an ArchitectureProposal or None"):
            self_improvement_module.SelfImprovementReport(
                current_strategy=make_strategy(),
                baseline_performance=make_performance(),
                proposals=[proposal],
                evaluations=[evaluation],
                approved_proposal="bad",  # type: ignore[arg-type]
                deployed_strategy=make_strategy(),
                active_strategy=make_strategy(),
                strategy_library_snapshot=[make_strategy()],
                summary="summary",
            )
        with pytest.raises(TypeError, match="deployed_strategy must be a dict or None"):
            self_improvement_module.SelfImprovementReport(
                current_strategy=make_strategy(),
                baseline_performance=make_performance(),
                proposals=[proposal],
                evaluations=[evaluation],
                approved_proposal=proposal,
                deployed_strategy="bad",  # type: ignore[arg-type]
                active_strategy=make_strategy(),
                strategy_library_snapshot=[make_strategy()],
                summary="summary",
            )
        with pytest.raises(TypeError, match="active_strategy must be a dict"):
            self_improvement_module.SelfImprovementReport(
                current_strategy=make_strategy(),
                baseline_performance=make_performance(),
                proposals=[proposal],
                evaluations=[evaluation],
                approved_proposal=proposal,
                deployed_strategy=make_strategy(),
                active_strategy="bad",  # type: ignore[arg-type]
                strategy_library_snapshot=[make_strategy()],
                summary="summary",
            )
        with pytest.raises(TypeError, match="strategy_library_snapshot must contain only dict strategies"):
            self_improvement_module.SelfImprovementReport(
                current_strategy=make_strategy(),
                baseline_performance=make_performance(),
                proposals=[proposal],
                evaluations=[evaluation],
                approved_proposal=proposal,
                deployed_strategy=make_strategy(),
                active_strategy=make_strategy(),
                strategy_library_snapshot=["bad"],  # type: ignore[list-item]
                summary="summary",
            )


class TestProposalGeneration:
    def test_propose_alternatives_returns_requested_count(self) -> None:
        proposals = propose_alternatives(make_performance(), make_strategy(), n=5)
        assert len(proposals) == 5
        assert all(proposal.proposal_id.startswith("ARCH-") for proposal in proposals)

    def test_proposals_are_distinct_perturbations(self) -> None:
        proposals = propose_alternatives(make_performance(), make_strategy(), n=4)
        seen = {tuple(sorted(proposal.strategy.items())) for proposal in proposals}
        assert len(seen) == len(proposals)

    def test_propose_alternatives_does_not_mutate_current_strategy(self) -> None:
        current = make_strategy()
        _ = propose_alternatives(make_performance(), current, n=3)
        assert current == make_strategy()

    def test_propose_alternatives_rejects_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be > 0"):
            propose_alternatives(make_performance(), make_strategy(), n=0)

    def test_propose_alternatives_handles_non_numeric_strategies(self) -> None:
        proposals = propose_alternatives(
            make_performance(),
            {"mode": "explore", "label": "baseline"},
            n=2,
        )
        assert proposals[0].strategy["_heuristic_variant"] == 1
        assert proposals[1].strategy["_heuristic_variant"] == 2

    def test_propose_alternatives_covers_zero_numeric_and_positive_int_restore(self) -> None:
        proposals = propose_alternatives(
            make_performance(score=-1.0),
            {"count": 1, "bias": 0.0, "enabled": True, "label": "baseline"},
            n=3,
        )
        assert all(proposal.strategy["count"] >= 1 for proposal in proposals)
        assert any(proposal.strategy["bias"] != 0.0 for proposal in proposals)
        assert all(proposal.strategy["enabled"] is True for proposal in proposals)

    def test_propose_alternatives_rejects_invalid_current_performance_and_bool_n(self) -> None:
        with pytest.raises(TypeError, match="PerformanceEvaluation"):
            propose_alternatives("bad", make_strategy())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be an integer"):
            propose_alternatives(make_performance(), make_strategy(), n=True)  # type: ignore[arg-type]


class TestSandbox:
    def test_sandbox_accepts_numeric_benchmark_output(self) -> None:
        proposal = ArchitectureProposal("ARCH-2", make_strategy(), "test")
        result = Sandbox().run(proposal, lambda _: 0.75)
        assert result.score == pytest.approx(0.75)

    def test_sandbox_accepts_mapping_benchmark_output(self) -> None:
        proposal = ArchitectureProposal("ARCH-3", make_strategy(), "test")
        result = run_in_sandbox(proposal, benchmark)
        assert result.proposal_id == "ARCH-3"
        assert "mutation_rate" in result.details

    def test_sandbox_accepts_performance_and_benchmark_result_outputs(self) -> None:
        proposal = ArchitectureProposal("ARCH-3A", make_strategy(), "test")
        result_from_performance = run_in_sandbox(
            proposal,
            lambda _: PerformanceEvaluation(score=0.8, convergence_speed=0.5, resource_cost=0.1),
        )
        assert result_from_performance.score == pytest.approx(0.8)

        result_from_benchmark = run_in_sandbox(
            proposal,
            lambda _: BenchmarkResult(
                proposal_id="IGNORED",
                score=0.9,
                convergence_speed=0.4,
                resource_cost=0.2,
                details={"k": "v"},
            ),
        )
        assert result_from_benchmark.score == pytest.approx(0.9)
        assert result_from_benchmark.details == {"k": "v"}

    def test_sandbox_isolation_prevents_proposal_mutation(self) -> None:
        proposal = ArchitectureProposal("ARCH-4", make_strategy(), "test")

        def mutating_benchmark(strategy: dict[str, object]) -> float:
            strategy["mutation_rate"] = 99.0
            return 0.50

        _ = run_in_sandbox(proposal, mutating_benchmark)
        assert proposal.strategy["mutation_rate"] == 0.30

    def test_run_in_sandbox_rejects_non_callable_benchmark(self) -> None:
        proposal = ArchitectureProposal("ARCH-5", make_strategy(), "test")
        with pytest.raises(TypeError, match="benchmark_fn must be callable"):
            run_in_sandbox(proposal, "not-callable")  # type: ignore[arg-type]

    def test_sandbox_rejects_invalid_proposal_and_bad_benchmark_outputs(self) -> None:
        with pytest.raises(TypeError, match="ArchitectureProposal"):
            Sandbox().run("bad", benchmark)  # type: ignore[arg-type]
        proposal = ArchitectureProposal("ARCH-5A", make_strategy(), "test")
        with pytest.raises(ValueError, match="contain 'score'"):
            run_in_sandbox(proposal, lambda _: {"details": {}})
        with pytest.raises(TypeError, match="benchmark result must be"):
            run_in_sandbox(proposal, lambda _: True)
        result = run_in_sandbox(proposal, lambda _: {"score": 0.6, "details": "opaque"})
        assert result.details == {"value": "opaque"}


class TestSafetyTests:
    def test_run_safety_tests_passes_with_no_checks(self) -> None:
        proposal = ArchitectureProposal("ARCH-6", make_strategy(), "test")
        result = run_safety_tests(proposal)
        assert result.passed
        assert result.violations == []

    def test_run_safety_tests_collects_multiple_failure_shapes(self) -> None:
        proposal = ArchitectureProposal(
            "ARCH-7",
            make_strategy(mutation_rate=0.70, population_size=150),
            "test",
        )

        def list_check(_: ArchitectureProposal) -> list[str]:
            return ["list-based violation"]

        result = run_safety_tests(
            proposal,
            [population_safety_check, mutation_safety_check, list_check],
        )
        assert not result.passed
        assert "population_size exceeds safe envelope" in result.violations
        assert "mutation_rate outside safe operating range" in result.violations
        assert "list-based violation" in result.violations

    def test_run_safety_tests_supports_invariant_like_objects(self) -> None:
        proposal = ArchitectureProposal("ARCH-8", make_strategy(), "test")

        @dataclass
        class Violation:
            message: str

        @dataclass
        class Report:
            passed: bool
            violations: list[Violation]

        result = run_safety_tests(proposal, [lambda _: Report(False, [Violation("invariant failed")])])
        assert result.passed is False
        assert result.violations == ["invariant failed"]

    def test_run_safety_tests_rejects_non_callable_checks(self) -> None:
        proposal = ArchitectureProposal("ARCH-9", make_strategy(), "test")
        with pytest.raises(TypeError, match="callable"):
            run_safety_tests(proposal, ["not-a-check"])  # type: ignore[list-item]

    def test_run_safety_tests_covers_additional_result_shapes(self) -> None:
        proposal = ArchitectureProposal("ARCH-9A", make_strategy(), "test")

        def false_check(_: ArchitectureProposal) -> bool:
            return False

        def tuple_true(_: ArchitectureProposal) -> tuple[bool, str]:
            return True, "ignored"

        def tuple_iterable(_: ArchitectureProposal) -> tuple[bool, list[str]]:
            return False, ["tuple-1", "tuple-2"]

        def tuple_scalar(_: ArchitectureProposal) -> tuple[bool, int]:
            return False, 7

        @dataclass
        class Report:
            passed: bool
            violations: list[object]

        result = run_safety_tests(
            proposal,
            [
                false_check,
                tuple_true,
                tuple_iterable,
                tuple_scalar,
                lambda _: Report(True, ["message despite pass"]),
                lambda _: Report(False, []),
            ],
        )
        assert result.passed is False
        assert "false_check returned False" in result.violations
        assert "tuple-1" in result.violations
        assert "7" in result.violations
        assert "message despite pass" in result.violations
        assert "<lambda> reported failure" in result.violations

    def test_run_safety_tests_rejects_invalid_proposal_and_result_type(self) -> None:
        with pytest.raises(TypeError, match="ArchitectureProposal"):
            run_safety_tests("bad")  # type: ignore[arg-type]
        proposal = ArchitectureProposal("ARCH-9B", make_strategy(), "test")
        with pytest.raises(TypeError, match="safety check results must be"):
            run_safety_tests(proposal, [lambda _: object()])


class TestRegressionTests:
    def test_regression_test_passes_for_better_proposal(self) -> None:
        proposal = ArchitectureProposal(
            "ARCH-10",
            make_strategy(mutation_rate=0.18, population_size=72, elite_count=3),
            "better",
        )
        result = run_regression_tests(proposal, make_performance(), benchmark)
        assert result.passed
        assert result.score_delta > 0.0

    def test_regression_test_fails_for_worse_proposal(self) -> None:
        proposal = ArchitectureProposal(
            "ARCH-11",
            make_strategy(mutation_rate=0.60, population_size=20, elite_count=1),
            "worse",
        )
        result = run_regression_tests(proposal, make_performance(), benchmark)
        assert not result.passed
        assert "regressed below baseline" in result.message

    def test_regression_test_can_reuse_existing_benchmark_result(self) -> None:
        proposal = ArchitectureProposal(
            "ARCH-12",
            make_strategy(mutation_rate=0.18, population_size=72, elite_count=3),
            "better",
        )
        existing = BenchmarkResult(proposal_id="ARCH-12", score=1.1)

        def failing_benchmark(_: dict[str, object]) -> float:
            raise AssertionError("benchmark should not be called")

        result = run_regression_tests(
            proposal,
            make_performance(),
            failing_benchmark,
            benchmark_result=existing,
        )
        assert result.passed
        assert result.candidate_score == pytest.approx(1.1)

    def test_regression_test_rejects_negative_tolerance(self) -> None:
        proposal = ArchitectureProposal("ARCH-13", make_strategy(), "test")
        with pytest.raises(ValueError, match="tolerance must be >= 0"):
            run_regression_tests(proposal, make_performance(), benchmark, tolerance=-0.01)

    def test_regression_tests_cover_success_message_and_invalid_inputs(self) -> None:
        proposal = ArchitectureProposal(
            "ARCH-13A",
            make_strategy(mutation_rate=0.18, population_size=72, elite_count=3),
            "equal-or-better",
        )
        result = run_regression_tests(
            proposal,
            make_performance(score=1.0),
            benchmark,
            benchmark_result=BenchmarkResult("ARCH-13A", score=1.0),
        )
        assert "matches or exceeds baseline" in result.message
        with pytest.raises(TypeError, match="ArchitectureProposal"):
            run_regression_tests("bad", make_performance(), benchmark)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="PerformanceEvaluation"):
            run_regression_tests(proposal, "bad", benchmark)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="benchmark_fn must be callable"):
            run_regression_tests(proposal, make_performance(), "bad")  # type: ignore[arg-type]


class TestApprovalPolicy:
    def test_approval_policy_approves_safe_improving_proposal(self) -> None:
        policy = ApprovalPolicy(min_improvement=0.1)
        benchmark_result = BenchmarkResult("ARCH-14", score=0.8)
        safety_result = SafetyTestResult("ARCH-14", passed=True)
        regression_result = RegressionTestResult(
            "ARCH-14",
            passed=True,
            baseline_score=0.4,
            candidate_score=0.8,
            score_delta=0.4,
            message="ok",
        )
        assert policy(benchmark_result, safety_result, regression_result)

    def test_approval_policy_rejects_unsafe_proposal(self) -> None:
        policy = ApprovalPolicy()
        benchmark_result = BenchmarkResult("ARCH-15", score=0.8)
        safety_result = SafetyTestResult("ARCH-15", passed=False, violations=["unsafe"])
        regression_result = RegressionTestResult(
            "ARCH-15",
            passed=True,
            baseline_score=0.4,
            candidate_score=0.8,
            score_delta=0.4,
            message="ok",
        )
        assert not policy(benchmark_result, safety_result, regression_result)

    def test_default_approval_policy_matches_default_policy_class(self) -> None:
        benchmark_result = BenchmarkResult("ARCH-16", score=0.8)
        safety_result = SafetyTestResult("ARCH-16", passed=True)
        regression_result = RegressionTestResult(
            "ARCH-16",
            passed=True,
            baseline_score=0.4,
            candidate_score=0.8,
            score_delta=0.4,
            message="ok",
        )
        assert default_approval_policy(benchmark_result, safety_result, regression_result)

    def test_approval_policy_rejects_regression_and_validates_inputs(self) -> None:
        with pytest.raises(TypeError, match="require_safety must be a bool"):
            ApprovalPolicy(require_safety=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="require_regression must be a bool"):
            ApprovalPolicy(require_regression=1)  # type: ignore[arg-type]

        benchmark_result = BenchmarkResult("ARCH-16A", score=0.8)
        safety_result = SafetyTestResult("ARCH-16A", passed=True)
        regression_result = RegressionTestResult(
            "ARCH-16A",
            passed=False,
            baseline_score=0.9,
            candidate_score=0.8,
            score_delta=-0.1,
            message="regressed",
        )
        assert not ApprovalPolicy()(benchmark_result, safety_result, regression_result)
        assert ApprovalPolicy(require_regression=False, min_improvement=-0.2)(
            benchmark_result, safety_result, regression_result
        )

        policy = ApprovalPolicy()
        with pytest.raises(TypeError, match="benchmark_result must be a BenchmarkResult"):
            policy("bad", safety_result, regression_result)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="safety_result must be a SafetyTestResult"):
            policy(benchmark_result, "bad", regression_result)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="regression_result must be a RegressionTestResult"):
            policy(benchmark_result, safety_result, "bad")  # type: ignore[arg-type]


class TestPipeline:
    def test_run_cycle_approves_best_candidate_and_updates_strategy_library(self) -> None:
        better = ArchitectureProposal(
            "BETTER",
            make_strategy(mutation_rate=0.18, population_size=72, elite_count=3),
            "better",
        )
        safe_but_smaller = ArchitectureProposal(
            "SMALLER",
            make_strategy(mutation_rate=0.22, population_size=68, elite_count=3),
            "smaller improvement",
        )
        unsafe = ArchitectureProposal(
            "UNSAFE",
            make_strategy(mutation_rate=0.18, population_size=160, elite_count=3),
            "unsafe",
        )

        def generator(
            current_performance: PerformanceEvaluation,
            current_strategy: dict[str, object],
            n: int,
        ) -> list[ArchitectureProposal]:
            del current_performance, current_strategy, n
            return [safe_but_smaller, unsafe, better]

        pipeline = SelfImprovementPipeline(proposal_generator=generator)
        report = pipeline.run_cycle(
            make_strategy(),
            make_performance(),
            benchmark,
            safety_checks=[population_safety_check, mutation_safety_check],
            approval_policy=ApprovalPolicy(min_improvement=0.05),
        )

        assert report.approved_proposal is not None
        assert report.approved_proposal.proposal_id == "BETTER"
        assert report.deployed_strategy == better.strategy
        assert pipeline.strategy_library == [make_strategy(), better.strategy]
        deployed = [evaluation for evaluation in report.evaluations if evaluation.deployed]
        assert len(deployed) == 1
        assert deployed[0].proposal.proposal_id == "BETTER"

    def test_run_cycle_rejects_all_bad_candidates_and_retains_baseline(self) -> None:
        unsafe = ArchitectureProposal(
            "UNSAFE",
            make_strategy(mutation_rate=0.18, population_size=160, elite_count=3),
            "unsafe",
        )
        regressive = ArchitectureProposal(
            "REGRESS",
            make_strategy(mutation_rate=0.60, population_size=20, elite_count=1),
            "regressive",
        )

        pipeline = SelfImprovementPipeline(
            proposal_generator=lambda _p, _s, _n: [unsafe, regressive]
        )
        report = pipeline.run_cycle(
            make_strategy(),
            make_performance(),
            benchmark,
            safety_checks=[population_safety_check, mutation_safety_check],
            approval_policy=ApprovalPolicy(min_improvement=0.01),
        )

        assert report.approved_proposal is None
        assert report.deployed_strategy is None
        assert report.active_strategy == make_strategy()
        assert pipeline.strategy_library == [make_strategy()]
        assert "retained the existing active strategy" in report.summary

    def test_run_cycle_records_external_baseline_if_library_is_out_of_date(self) -> None:
        pipeline = SelfImprovementPipeline(proposal_generator=lambda _p, _s, _n: [])
        first = make_strategy()
        second = make_strategy(population_size=44)
        pipeline.run_cycle(first, make_performance(), benchmark)
        report = pipeline.run_cycle(second, make_performance(), benchmark)
        assert report.strategy_library_snapshot[-1] == second
        assert pipeline.strategy_library == [first, second]

    def test_run_cycle_rejects_non_callable_approval_policy(self) -> None:
        pipeline = SelfImprovementPipeline(proposal_generator=lambda _p, _s, _n: [])
        with pytest.raises(TypeError, match="approval_policy must be callable"):
            pipeline.run_cycle(
                make_strategy(),
                make_performance(),
                benchmark,
                approval_policy="not-callable",  # type: ignore[arg-type]
            )

    def test_run_cycle_rejects_invalid_generator_output(self) -> None:
        pipeline = SelfImprovementPipeline(proposal_generator=lambda _p, _s, _n: ["bad"])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="ArchitectureProposal"):
            pipeline.run_cycle(make_strategy(), make_performance(), benchmark)

    def test_pipeline_constructor_and_seed_management_validation(self) -> None:
        with pytest.raises(TypeError, match="callable or None"):
            SelfImprovementPipeline(proposal_generator="bad")  # type: ignore[arg-type]
        pipeline = SelfImprovementPipeline(proposal_generator=lambda _p, _s, _n: [])
        pipeline._ensure_library_seed(make_strategy())
        pipeline._ensure_library_seed(make_strategy())
        assert pipeline.strategy_library == [make_strategy()]

    def test_run_cycle_validates_current_performance_and_benchmark_fn(self) -> None:
        pipeline = SelfImprovementPipeline(proposal_generator=lambda _p, _s, _n: [])
        with pytest.raises(TypeError, match="current_performance must be a PerformanceEvaluation"):
            pipeline.run_cycle(make_strategy(), "bad", benchmark)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="benchmark_fn must be callable"):
            pipeline.run_cycle(make_strategy(), make_performance(), "bad")  # type: ignore[arg-type]

    def test_run_cycle_approved_duplicate_strategy_does_not_append_library_entry(self) -> None:
        baseline = make_strategy()
        duplicate = ArchitectureProposal("SAME", baseline, "same strategy")
        pipeline = SelfImprovementPipeline(proposal_generator=lambda _p, _s, _n: [duplicate])
        report = pipeline.run_cycle(
            baseline,
            make_performance(),
            benchmark,
            approval_policy=lambda *_args: True,
        )
        assert report.approved_proposal is not None
        assert pipeline.strategy_library == [baseline]
