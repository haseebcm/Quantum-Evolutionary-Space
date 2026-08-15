"""Phase 14 -- governed self-improvement for QES.

QES can evaluate the measured performance of one search strategy,
heuristically propose alternative parameter configurations, benchmark them in
an isolated sandbox, run safety and regression gates, and record an approved
configuration in a strategy library. This module never rewrites production
source code. "Deployment" here only means recording a caller-supplied
strategy/configuration as the new active entry in `strategy_library`.

Everything below is ordinary classical Python control flow over dict/list
configuration objects and caller-provided benchmark functions. There is no AGI,
no autonomous code mutation, and no unsupervised self-rewrite.
"""
from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

StrategyConfig = dict[str, Any]
BenchmarkFn = Callable[[StrategyConfig], object]
SafetyCheck = Callable[["ArchitectureProposal"], object]

_proposal_id_counter = itertools.count(1)


def _next_proposal_id() -> str:
    return f"ARCH-{next(_proposal_id_counter):06d}"


def _validate_nonempty_string(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _validate_mapping(name: str, value: Mapping[str, Any]) -> StrategyConfig:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return copy.deepcopy(dict(value))


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    metric = float(value)
    if not math.isfinite(metric):
        raise ValueError(f"{name} must be finite")
    return metric


def _clone_strategy(strategy: Mapping[str, Any]) -> StrategyConfig:
    return copy.deepcopy(dict(strategy))


def _is_numeric_tunable(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _restore_numeric_type(original: Any, candidate: float) -> Any:
    if isinstance(original, bool):
        return original
    if isinstance(original, int):
        rounded = int(round(candidate))
        if original > 0 and rounded <= 0:
            return 1
        return rounded
    return float(candidate)


def _coerce_performance(result: object) -> PerformanceEvaluation:
    if isinstance(result, PerformanceEvaluation):
        return result
    if isinstance(result, BenchmarkResult):
        return PerformanceEvaluation(
            score=result.score,
            convergence_speed=result.convergence_speed,
            resource_cost=result.resource_cost,
            details=result.details,
        )
    if isinstance(result, Mapping):
        if "score" not in result:
            raise ValueError("benchmark result mapping must contain 'score'")
        details = result.get("details", {})
        return PerformanceEvaluation(
            score=float(result["score"]),
            convergence_speed=float(result.get("convergence_speed", 0.0)),
            resource_cost=float(result.get("resource_cost", 0.0)),
            details=dict(details) if isinstance(details, Mapping) else {"value": details},
        )
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return PerformanceEvaluation(score=float(result))
    raise TypeError(
        "benchmark result must be a PerformanceEvaluation, BenchmarkResult, "
        "mapping with 'score', or numeric score"
    )


def _normalize_safety_result(result: object, check_name: str) -> tuple[bool, list[str]]:
    if result is None or result is True:
        return True, []
    if result is False:
        return False, [f"{check_name} returned False"]
    if isinstance(result, str):
        return False, [result]
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
        passed = result[0]
        detail = result[1]
        if passed:
            return True, []
        if isinstance(detail, str):
            return False, [detail]
        if isinstance(detail, Iterable):
            return False, [str(item) for item in detail]
        return False, [str(detail)]
    if isinstance(result, Iterable) and not isinstance(result, (bytes, bytearray, str, Mapping)):
        messages = [str(item) for item in result]
        return not messages, messages
    if hasattr(result, "passed"):
        passed = bool(result.passed)
        raw_violations = result.violations if hasattr(result, "violations") else []
        messages = [getattr(item, "message", str(item)) for item in raw_violations]
        if passed and messages:
            return False, messages
        if not passed and not messages:
            return False, [f"{check_name} reported failure"]
        return passed, messages
    raise TypeError(
        "safety check results must be bool, str, iterable[str], "
        "(bool, detail), or an object with passed/violations"
    )


def _proposal_reason(
    benchmark_result: BenchmarkResult,
    safety_result: SafetyTestResult,
    regression_result: RegressionTestResult,
    approved: bool,
) -> str:
    if approved:
        return (
            f"approved by policy: score_delta={regression_result.score_delta:.4f}, "
            f"safety_passed={safety_result.passed}"
        )
    if not safety_result.passed:
        return f"rejected by safety gate: {', '.join(safety_result.violations)}"
    if not regression_result.passed:
        return (
            "rejected by regression gate: "
            f"candidate_score={benchmark_result.score:.4f} < baseline_score="
            f"{regression_result.baseline_score:.4f}"
        )
    return (
        "rejected by approval policy: "
        f"score_delta={regression_result.score_delta:.4f} did not meet policy threshold"
    )


@dataclass
class PerformanceEvaluation:
    """Measured performance of one caller-supplied strategy configuration.

    The fields are numeric summaries of benchmark behavior. This is ordinary
    metric bookkeeping over caller-provided benchmark outputs, not a claim that
    QES is conscious or rewriting itself.
    """

    score: float
    convergence_speed: float = 0.0
    resource_cost: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = _validate_finite_float("score", self.score)
        self.convergence_speed = _validate_finite_float(
            "convergence_speed", self.convergence_speed
        )
        self.resource_cost = _validate_finite_float("resource_cost", self.resource_cost)
        if not isinstance(self.details, dict):
            raise TypeError("details must be a dict")
        self.details = copy.deepcopy(self.details)


@dataclass
class ArchitectureProposal:
    """Candidate strategy/configuration produced by heuristic perturbation.

    `strategy` is data only: a plain configuration dictionary. This class does
    not contain, patch, or rewrite source code.
    """

    proposal_id: str
    strategy: dict[str, Any]
    rationale: str

    def __post_init__(self) -> None:
        self.proposal_id = _validate_nonempty_string("proposal_id", self.proposal_id)
        if not isinstance(self.strategy, dict):
            raise TypeError("strategy must be a dict")
        self.strategy = copy.deepcopy(self.strategy)
        self.rationale = _validate_nonempty_string("rationale", self.rationale)


@dataclass
class BenchmarkResult:
    """Measured sandbox benchmark result for one proposal."""

    proposal_id: str
    score: float
    convergence_speed: float = 0.0
    resource_cost: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.proposal_id = _validate_nonempty_string("proposal_id", self.proposal_id)
        self.score = _validate_finite_float("score", self.score)
        self.convergence_speed = _validate_finite_float(
            "convergence_speed", self.convergence_speed
        )
        self.resource_cost = _validate_finite_float("resource_cost", self.resource_cost)
        if not isinstance(self.details, dict):
            raise TypeError("details must be a dict")
        self.details = copy.deepcopy(self.details)


@dataclass
class SafetyTestResult:
    """Aggregate result of caller-supplied proposal safety checks."""

    proposal_id: str
    passed: bool
    violations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.proposal_id = _validate_nonempty_string("proposal_id", self.proposal_id)
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        self.violations = [str(item) for item in self.violations]
        if not isinstance(self.details, dict):
            raise TypeError("details must be a dict")
        self.details = copy.deepcopy(self.details)


@dataclass
class RegressionTestResult:
    """Regression gate comparing a proposal against the current baseline."""

    proposal_id: str
    passed: bool
    baseline_score: float
    candidate_score: float
    score_delta: float
    message: str

    def __post_init__(self) -> None:
        self.proposal_id = _validate_nonempty_string("proposal_id", self.proposal_id)
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")
        self.baseline_score = _validate_finite_float("baseline_score", self.baseline_score)
        self.candidate_score = _validate_finite_float("candidate_score", self.candidate_score)
        self.score_delta = _validate_finite_float("score_delta", self.score_delta)
        self.message = _validate_nonempty_string("message", self.message)


@dataclass
class ProposalEvaluation:
    """Full governed evaluation trace for one proposal."""

    proposal: ArchitectureProposal
    benchmark_result: BenchmarkResult
    safety_result: SafetyTestResult
    regression_result: RegressionTestResult
    approved: bool
    deployed: bool = False
    decision_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, ArchitectureProposal):
            raise TypeError("proposal must be an ArchitectureProposal")
        if not isinstance(self.benchmark_result, BenchmarkResult):
            raise TypeError("benchmark_result must be a BenchmarkResult")
        if not isinstance(self.safety_result, SafetyTestResult):
            raise TypeError("safety_result must be a SafetyTestResult")
        if not isinstance(self.regression_result, RegressionTestResult):
            raise TypeError("regression_result must be a RegressionTestResult")
        if not isinstance(self.approved, bool):
            raise TypeError("approved must be a bool")
        if not isinstance(self.deployed, bool):
            raise TypeError("deployed must be a bool")
        if self.decision_reason:
            self.decision_reason = _validate_nonempty_string("decision_reason", self.decision_reason)


@dataclass
class SelfImprovementReport:
    """One full pipeline cycle.

    The report records proposals, benchmark/safety/regression outcomes, the
    approved proposal if one exists, and the strategy-library snapshot after
    the cycle. No source-code mutation occurs; the report is only structured
    bookkeeping around candidate strategy configurations.
    """

    current_strategy: dict[str, Any]
    baseline_performance: PerformanceEvaluation
    proposals: list[ArchitectureProposal]
    evaluations: list[ProposalEvaluation]
    approved_proposal: ArchitectureProposal | None
    deployed_strategy: dict[str, Any] | None
    active_strategy: dict[str, Any]
    strategy_library_snapshot: list[dict[str, Any]]
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.current_strategy, dict):
            raise TypeError("current_strategy must be a dict")
        if not isinstance(self.baseline_performance, PerformanceEvaluation):
            raise TypeError("baseline_performance must be a PerformanceEvaluation")
        if any(not isinstance(proposal, ArchitectureProposal) for proposal in self.proposals):
            raise TypeError("proposals must contain only ArchitectureProposal objects")
        if any(not isinstance(item, ProposalEvaluation) for item in self.evaluations):
            raise TypeError("evaluations must contain only ProposalEvaluation objects")
        if self.approved_proposal is not None and not isinstance(
            self.approved_proposal, ArchitectureProposal
        ):
            raise TypeError("approved_proposal must be an ArchitectureProposal or None")
        if self.deployed_strategy is not None and not isinstance(self.deployed_strategy, dict):
            raise TypeError("deployed_strategy must be a dict or None")
        if not isinstance(self.active_strategy, dict):
            raise TypeError("active_strategy must be a dict")
        if any(not isinstance(item, dict) for item in self.strategy_library_snapshot):
            raise TypeError("strategy_library_snapshot must contain only dict strategies")
        self.current_strategy = copy.deepcopy(self.current_strategy)
        self.proposals = copy.deepcopy(self.proposals)
        self.evaluations = copy.deepcopy(self.evaluations)
        self.deployed_strategy = copy.deepcopy(self.deployed_strategy)
        self.active_strategy = copy.deepcopy(self.active_strategy)
        self.strategy_library_snapshot = copy.deepcopy(self.strategy_library_snapshot)
        self.summary = _validate_nonempty_string("summary", self.summary)


def propose_alternatives(
    current_performance: PerformanceEvaluation,
    current_strategy: Mapping[str, Any],
    n: int = 4,
) -> list[ArchitectureProposal]:
    """Generate heuristic alternatives by perturbing numeric strategy fields.

    This is deterministic parameter variation, not literal creativity or AGI.
    It copies the caller's configuration, perturbs numeric tunables, and
    returns data-only proposals for later sandboxing and governance.
    """

    if not isinstance(current_performance, PerformanceEvaluation):
        raise TypeError("current_performance must be a PerformanceEvaluation")
    baseline_strategy = _validate_mapping("current_strategy", current_strategy)
    proposal_count = _validate_positive_int("n", n)

    numeric_keys = [key for key, value in baseline_strategy.items() if _is_numeric_tunable(value)]
    step_scale = max(0.05, min(0.25, 0.05 + max(0.0, 1.0 - current_performance.score) * 0.1))
    proposals: list[ArchitectureProposal] = []

    for index in range(proposal_count):
        candidate = _clone_strategy(baseline_strategy)
        if numeric_keys:
            for key_index, key in enumerate(sorted(numeric_keys)):
                original = candidate[key]
                direction = -1.0 if (index + key_index) % 2 == 0 else 1.0
                magnitude = step_scale * (1.0 + ((index + key_index) % 3) * 0.5)
                if float(original) == 0.0:
                    updated = direction * magnitude
                else:
                    updated = float(original) * (1.0 + direction * magnitude)
                candidate[key] = _restore_numeric_type(original, updated)
        else:
            candidate["_heuristic_variant"] = index + 1

        proposals.append(
            ArchitectureProposal(
                proposal_id=_next_proposal_id(),
                strategy=candidate,
                rationale=(
                    "Heuristic perturbation of caller-supplied numeric tunables; "
                    f"proposal {index + 1}/{proposal_count} derived from score="
                    f"{current_performance.score:.4f}."
                ),
            )
        )
    return proposals


class Sandbox:
    """Isolated benchmark runner for data-only strategy proposals.

    The sandbox clones a proposal's configuration before executing the
    benchmark function so test code cannot mutate the stored proposal or a
    caller's current production configuration by aliasing shared state.
    """

    def run(self, proposal: ArchitectureProposal, benchmark_fn: BenchmarkFn) -> BenchmarkResult:
        """Benchmark `proposal` against a caller-supplied function in isolation."""

        if not isinstance(proposal, ArchitectureProposal):
            raise TypeError("proposal must be an ArchitectureProposal")
        if not callable(benchmark_fn):
            raise TypeError("benchmark_fn must be callable")

        benchmark_input = _clone_strategy(proposal.strategy)
        performance = _coerce_performance(benchmark_fn(benchmark_input))
        return BenchmarkResult(
            proposal_id=proposal.proposal_id,
            score=performance.score,
            convergence_speed=performance.convergence_speed,
            resource_cost=performance.resource_cost,
            details=performance.details,
        )


def run_in_sandbox(proposal: ArchitectureProposal, benchmark_fn: BenchmarkFn) -> BenchmarkResult:
    """Convenience wrapper around `Sandbox.run`."""

    return Sandbox().run(proposal, benchmark_fn)


def run_safety_tests(
    proposal: ArchitectureProposal,
    invariant_checks: Iterable[SafetyCheck] | None = None,
) -> SafetyTestResult:
    """Run caller-supplied proposal safety checks.

    These checks govern candidate strategy data only. They do not mutate source
    code and can stand in for policy, invariant, or operational guardrails.
    """

    if not isinstance(proposal, ArchitectureProposal):
        raise TypeError("proposal must be an ArchitectureProposal")
    checks = list(invariant_checks or ())

    violations: list[str] = []
    details: dict[str, Any] = {}
    for check in checks:
        if not callable(check):
            raise TypeError("each safety check must be callable")
        check_name = getattr(check, "__name__", check.__class__.__name__)
        passed, messages = _normalize_safety_result(check(proposal), check_name)
        details[check_name] = {"passed": passed, "violations": list(messages)}
        if not passed:
            violations.extend(messages)

    return SafetyTestResult(
        proposal_id=proposal.proposal_id,
        passed=not violations,
        violations=violations,
        details=details,
    )


def run_regression_tests(
    proposal: ArchitectureProposal,
    baseline_performance: PerformanceEvaluation,
    benchmark_fn: BenchmarkFn,
    *,
    tolerance: float = 0.0,
    benchmark_result: BenchmarkResult | None = None,
) -> RegressionTestResult:
    """Confirm that a proposal does not score below the measured baseline.

    The same caller-supplied benchmark is used for comparison. Passing an
    existing `benchmark_result` avoids re-running the benchmark inside a
    pipeline that already executed the sandbox stage.
    """

    if not isinstance(proposal, ArchitectureProposal):
        raise TypeError("proposal must be an ArchitectureProposal")
    if not isinstance(baseline_performance, PerformanceEvaluation):
        raise TypeError("baseline_performance must be a PerformanceEvaluation")
    if not callable(benchmark_fn):
        raise TypeError("benchmark_fn must be callable")
    tolerance = _validate_finite_float("tolerance", tolerance)
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")

    result = benchmark_result or run_in_sandbox(proposal, benchmark_fn)
    score_delta = result.score - baseline_performance.score
    passed = result.score >= baseline_performance.score - tolerance
    if passed:
        message = (
            f"proposal matches or exceeds baseline within tolerance {tolerance:.4f}: "
            f"{result.score:.4f} vs {baseline_performance.score:.4f}"
        )
    else:
        message = (
            f"proposal regressed below baseline by {baseline_performance.score - result.score:.4f}: "
            f"{result.score:.4f} vs {baseline_performance.score:.4f}"
        )
    return RegressionTestResult(
        proposal_id=proposal.proposal_id,
        passed=passed,
        baseline_score=baseline_performance.score,
        candidate_score=result.score,
        score_delta=score_delta,
        message=message,
    )


@dataclass
class ApprovalPolicy:
    """Automated approval gate for the final pipeline stage.

    This stands in for a human/policy approval stage in tests and demos. A real
    human callback can be substituted anywhere a callable with the same
    signature is accepted.
    """

    min_improvement: float = 0.0
    require_safety: bool = True
    require_regression: bool = True

    def __post_init__(self) -> None:
        self.min_improvement = _validate_finite_float("min_improvement", self.min_improvement)
        if not isinstance(self.require_safety, bool):
            raise TypeError("require_safety must be a bool")
        if not isinstance(self.require_regression, bool):
            raise TypeError("require_regression must be a bool")

    def __call__(
        self,
        benchmark_result: BenchmarkResult,
        safety_result: SafetyTestResult,
        regression_result: RegressionTestResult,
    ) -> bool:
        """Approve only proposals that satisfy the configured policy."""

        if not isinstance(benchmark_result, BenchmarkResult):
            raise TypeError("benchmark_result must be a BenchmarkResult")
        if not isinstance(safety_result, SafetyTestResult):
            raise TypeError("safety_result must be a SafetyTestResult")
        if not isinstance(regression_result, RegressionTestResult):
            raise TypeError("regression_result must be a RegressionTestResult")
        if self.require_safety and not safety_result.passed:
            return False
        if self.require_regression and not regression_result.passed:
            return False
        return regression_result.score_delta >= self.min_improvement


def default_approval_policy(
    benchmark_result: BenchmarkResult,
    safety_result: SafetyTestResult,
    regression_result: RegressionTestResult,
) -> bool:
    """Default automated policy gate.

    This is not a literal human approval step; it is a deterministic policy
    callable that callers may replace with a human review callback.
    """

    return ApprovalPolicy()(benchmark_result, safety_result, regression_result)


class SelfImprovementPipeline:
    """Governed architecture-selection loop for strategy configurations.

    The pipeline is:
        Architecture Proposal -> Sandbox -> Benchmark -> Safety Tests
        -> Regression Tests -> Approval -> Deployment

    Deployment never rewrites code. It only records an approved, caller-defined
    strategy dictionary in `strategy_library` as the new active configuration.
    """

    def __init__(
        self,
        proposal_generator: Callable[
            [PerformanceEvaluation, Mapping[str, Any], int], list[ArchitectureProposal]
        ]
        | None = None,
    ) -> None:
        if proposal_generator is not None and not callable(proposal_generator):
            raise TypeError("proposal_generator must be callable or None")
        self.proposal_generator = proposal_generator or propose_alternatives
        self.strategy_library: list[dict[str, Any]] = []
        self.reports: list[SelfImprovementReport] = []

    def _ensure_library_seed(self, current_strategy: Mapping[str, Any]) -> None:
        strategy_copy = _clone_strategy(current_strategy)
        if not self.strategy_library or self.strategy_library[-1] != strategy_copy:
            self.strategy_library.append(strategy_copy)

    def run_cycle(
        self,
        current_strategy: Mapping[str, Any],
        current_performance: PerformanceEvaluation,
        benchmark_fn: BenchmarkFn,
        safety_checks: Iterable[SafetyCheck] | None = None,
        approval_policy: Callable[[BenchmarkResult, SafetyTestResult, RegressionTestResult], bool]
        | None = None,
        *,
        n: int = 4,
    ) -> SelfImprovementReport:
        """Run one governed self-improvement cycle over strategy data.

        The current strategy is treated as immutable input. Every candidate is
        benchmarked on a cloned copy, checked by safety and regression gates,
        optionally approved by policy, and only then recorded in
        `strategy_library` if selected. No source file mutation happens.
        """

        baseline_strategy = _validate_mapping("current_strategy", current_strategy)
        if not isinstance(current_performance, PerformanceEvaluation):
            raise TypeError("current_performance must be a PerformanceEvaluation")
        if not callable(benchmark_fn):
            raise TypeError("benchmark_fn must be callable")
        if approval_policy is None:
            resolved_policy: Callable[
                [BenchmarkResult, SafetyTestResult, RegressionTestResult], bool
            ] = default_approval_policy
        else:
            if not callable(approval_policy):
                raise TypeError("approval_policy must be callable")
            resolved_policy = approval_policy

        self._ensure_library_seed(baseline_strategy)
        proposals = self.proposal_generator(current_performance, baseline_strategy, n)
        if any(not isinstance(proposal, ArchitectureProposal) for proposal in proposals):
            raise TypeError("proposal_generator must return ArchitectureProposal objects")

        evaluations: list[ProposalEvaluation] = []
        approved_candidates: list[ProposalEvaluation] = []
        for proposal in proposals:
            benchmark_result = run_in_sandbox(proposal, benchmark_fn)
            safety_result = run_safety_tests(proposal, safety_checks)
            regression_result = run_regression_tests(
                proposal,
                current_performance,
                benchmark_fn,
                benchmark_result=benchmark_result,
            )
            approved = bool(resolved_policy(benchmark_result, safety_result, regression_result))
            evaluation = ProposalEvaluation(
                proposal=proposal,
                benchmark_result=benchmark_result,
                safety_result=safety_result,
                regression_result=regression_result,
                approved=approved,
                decision_reason=_proposal_reason(
                    benchmark_result, safety_result, regression_result, approved
                ),
            )
            evaluations.append(evaluation)
            if approved:
                approved_candidates.append(evaluation)

        approved_proposal: ArchitectureProposal | None = None
        deployed_strategy: dict[str, Any] | None = None
        active_strategy = _clone_strategy(baseline_strategy)
        if approved_candidates:
            chosen = max(
                approved_candidates,
                key=lambda item: (
                    item.benchmark_result.score,
                    item.regression_result.score_delta,
                    -item.benchmark_result.resource_cost,
                ),
            )
            chosen.deployed = True
            approved_proposal = copy.deepcopy(chosen.proposal)
            deployed_strategy = _clone_strategy(chosen.proposal.strategy)
            active_strategy = _clone_strategy(chosen.proposal.strategy)
            if self.strategy_library[-1] != deployed_strategy:
                self.strategy_library.append(_clone_strategy(deployed_strategy))
            summary = (
                f"Approved {chosen.proposal.proposal_id} and recorded it as the new active "
                "strategy configuration."
            )
        else:
            summary = (
                "No proposal satisfied the governed gates; retained the existing active "
                "strategy configuration."
            )

        report = SelfImprovementReport(
            current_strategy=baseline_strategy,
            baseline_performance=current_performance,
            proposals=list(proposals),
            evaluations=evaluations,
            approved_proposal=approved_proposal,
            deployed_strategy=deployed_strategy,
            active_strategy=active_strategy,
            strategy_library_snapshot=copy.deepcopy(self.strategy_library),
            summary=summary,
        )
        self.reports.append(copy.deepcopy(report))
        return report


__all__ = [
    "ApprovalPolicy",
    "ArchitectureProposal",
    "BenchmarkResult",
    "PerformanceEvaluation",
    "ProposalEvaluation",
    "RegressionTestResult",
    "SafetyTestResult",
    "Sandbox",
    "SelfImprovementPipeline",
    "SelfImprovementReport",
    "default_approval_policy",
    "propose_alternatives",
    "run_in_sandbox",
    "run_regression_tests",
    "run_safety_tests",
]
