"""Phase 14 demo: governed self-improvement over strategy configurations.

This walkthrough shows QES evaluating a current search strategy, generating
alternative strategy configurations, benchmarking them in a sandbox, running
safety and regression gates, and approving only safe, better configurations.
No production code is rewritten here. "Deployment" means only that an
approved configuration is recorded as the new active strategy in an in-memory
strategy library.
"""
from __future__ import annotations

import time
import tracemalloc

from qes.self_improvement import (
    ApprovalPolicy,
    ArchitectureProposal,
    PerformanceEvaluation,
    SelfImprovementPipeline,
)


def synthetic_benchmark(strategy: dict[str, object]) -> dict[str, object]:
    """Toy classical benchmark for a search-strategy configuration."""

    mutation = float(strategy["mutation_rate"])
    population = float(strategy["population_size"])
    elite = float(strategy["elite_count"])
    score = 1.2 - abs(mutation - 0.18) * 3.0 - abs(population - 72.0) * 0.01 - abs(elite - 3.0) * 0.1
    convergence_speed = max(0.0, 1.0 - abs(mutation - 0.18))
    resource_cost = population * 0.02 + elite * 0.05
    return {
        "score": score,
        "convergence_speed": convergence_speed,
        "resource_cost": resource_cost,
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


def print_cycle_report(cycle_label: str, report: object) -> None:
    """Print one cycle in the existing examples' simple, labeled style."""

    print(cycle_label)
    print("-" * 60)
    for evaluation in report.evaluations:
        proposal = evaluation.proposal
        benchmark_result = evaluation.benchmark_result
        safety_result = evaluation.safety_result
        regression_result = evaluation.regression_result
        deployment = "deployed" if evaluation.deployed else "not deployed"
        print(f"[Proposal]    {proposal.proposal_id}: {proposal.strategy}")
        print(f"[Sandbox]     isolated copy benchmarked for {proposal.proposal_id}")
        print(
            "[Benchmark]   "
            f"score={benchmark_result.score:.4f}, "
            f"speed={benchmark_result.convergence_speed:.4f}, "
            f"cost={benchmark_result.resource_cost:.4f}"
        )
        print(
            f"[Safety]      passed={safety_result.passed}, "
            f"violations={safety_result.violations or ['none']}"
        )
        print(
            "[Regression]  "
            f"passed={regression_result.passed}, "
            f"delta={regression_result.score_delta:.4f}"
        )
        print(f"[Approval]    approved={evaluation.approved}, reason={evaluation.decision_reason}")
        print(f"[Deployment]  {deployment}")
        print()
    print(f"[Summary]     {report.summary}")
    print(f"[Library]     {report.strategy_library_snapshot}")
    print()


def cycle_one_generator(
    current_performance: PerformanceEvaluation,
    current_strategy: dict[str, object],
    n: int,
) -> list[ArchitectureProposal]:
    """First cycle: one strong candidate, one unsafe, one regressive."""

    del current_performance, current_strategy, n
    return [
        ArchitectureProposal(
            proposal_id="CYCLE1-UNSAFE",
            strategy={"mutation_rate": 0.18, "population_size": 150, "elite_count": 3},
            rationale="Improves score but exceeds the safety envelope on population size.",
        ),
        ArchitectureProposal(
            proposal_id="CYCLE1-BETTER",
            strategy={"mutation_rate": 0.18, "population_size": 72, "elite_count": 3},
            rationale="Moves all tunables toward the synthetic benchmark sweet spot.",
        ),
        ArchitectureProposal(
            proposal_id="CYCLE1-REGRESS",
            strategy={"mutation_rate": 0.60, "population_size": 20, "elite_count": 1},
            rationale="Intentionally poor control candidate to demonstrate regression gating.",
        ),
    ]


def cycle_two_generator(
    current_performance: PerformanceEvaluation,
    current_strategy: dict[str, object],
    n: int,
) -> list[ArchitectureProposal]:
    """Second cycle: proposals are unsafe or regressive, so nothing deploys."""

    del current_performance, current_strategy, n
    return [
        ArchitectureProposal(
            proposal_id="CYCLE2-UNSAFE",
            strategy={"mutation_rate": 0.18, "population_size": 180, "elite_count": 3},
            rationale="Unsafe resource envelope; should fail safety checks.",
        ),
        ArchitectureProposal(
            proposal_id="CYCLE2-REGRESS",
            strategy={"mutation_rate": 0.48, "population_size": 40, "elite_count": 1},
            rationale="Safe enough to test, but worse than the current approved baseline.",
        ),
    ]


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 14: SELF-IMPROVING QES")
    print("=" * 60)

    current_strategy: dict[str, object] = {
        "mutation_rate": 0.32,
        "population_size": 50,
        "elite_count": 2,
    }
    current_performance = PerformanceEvaluation(**synthetic_benchmark(current_strategy))
    approval_policy = ApprovalPolicy(min_improvement=0.05)

    print(f"[Current]     strategy={current_strategy}")
    print(
        "[Current]     "
        f"score={current_performance.score:.4f}, "
        f"speed={current_performance.convergence_speed:.4f}, "
        f"cost={current_performance.resource_cost:.4f}"
    )
    print()

    pipeline = SelfImprovementPipeline(proposal_generator=cycle_one_generator)
    report_one = pipeline.run_cycle(
        current_strategy,
        current_performance,
        synthetic_benchmark,
        safety_checks=[population_safety_check, mutation_safety_check],
        approval_policy=approval_policy,
    )
    print_cycle_report("CYCLE 1 -- EXPECT ONE APPROVAL", report_one)

    current_strategy = report_one.active_strategy
    current_performance = PerformanceEvaluation(**synthetic_benchmark(current_strategy))
    pipeline.proposal_generator = cycle_two_generator
    report_two = pipeline.run_cycle(
        current_strategy,
        current_performance,
        synthetic_benchmark,
        safety_checks=[population_safety_check, mutation_safety_check],
        approval_policy=approval_policy,
    )
    print_cycle_report("CYCLE 2 -- EXPECT REJECTIONS ONLY", report_two)

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print(f"  strategy library     : {pipeline.strategy_library}")
    print(
        "  No production code was rewritten: QES only proposed, tested, and "
        "selected among caller-supplied strategy configurations."
    )
    print(
        "  This is classical in-process benchmarking and policy gating, not "
        "AGI, self-awareness, or literal quantum/free compute."
    )


if __name__ == "__main__":
    main()
