"""Phase 17 demo: multi-agent QSEE evolution with individual vs collective comparison.

This is a classical in-process orchestration demo. The "agents" below are
lightweight specialized scoring heuristics operating over ordinary Python
dicts; they are not autonomous AI agents or LLMs, and there is no literal
quantum computing involved.
"""
from __future__ import annotations

import time
import tracemalloc

from qes.multi_agent_evolution import (
    SPECIALIZATIONS,
    AgentPopulation,
    SpecializedAgent,
    collective_solution,
    compare_individual_vs_collective,
)

try:
    from qes.knowledge_graph import KnowledgeGraph
except ImportError:  # pragma: no cover - defensive only
    KnowledgeGraph = None


def make_candidate(candidate_id: str, **overrides: float) -> dict[str, object]:
    candidate: dict[str, object] = {
        "id": candidate_id,
        "objective_score": 0.5,
        "predicted_score": 0.5,
        "novelty": 0.5,
        "spread": 0.5,
        "mathematical_consistency": 0.5,
        "proof_strength": 0.5,
        "analytical_clarity": 0.5,
        "evidence": 0.5,
        "reproducibility": 0.5,
        "consistency": 0.5,
        "safety_margin": 0.5,
        "robustness": 0.5,
        "forecast_confidence": 0.5,
        "trend_alignment": 0.5,
        "resource_efficiency": 0.5,
        "budget_fit": 0.5,
        "throughput": 0.5,
        "causal_confidence": 0.5,
        "explainability": 0.5,
        "counterfactual_support": 0.5,
        "adversarial_resilience": 0.5,
        "edge_case_coverage": 0.5,
        "synthesis_quality": 0.5,
        "coherence": 0.5,
        "versatility": 0.5,
        "validation_score": 0.5,
        "feasibility": 0.5,
        "simplicity": 0.5,
        "efficiency": 0.5,
        "complexity": 0.2,
        "constraint_violation": 0.0,
        "uncertainty": 0.1,
        "risk": 0.1,
        "resource_cost": 0.2,
        "vulnerability": 0.1,
    }
    candidate.update(overrides)
    return candidate


def objective_fn(candidate: dict[str, object]) -> float:
    return (
        1.6 * float(candidate["objective_score"])
        + 1.1 * float(candidate["validation_score"])
        + 1.0 * float(candidate["safety_margin"])
        + 0.8 * float(candidate["robustness"])
        + 0.6 * float(candidate["resource_efficiency"])
        + 0.5 * float(candidate["predicted_score"])
        - 1.6 * float(candidate["constraint_violation"])
        - 0.9 * float(candidate["risk"])
        - 0.7 * float(candidate["vulnerability"])
        - 0.4 * float(candidate["resource_cost"])
    )


def build_candidates() -> list[dict[str, object]]:
    return [
        make_candidate(
            "risky-optimizer",
            objective_score=0.99,
            efficiency=0.96,
            predicted_score=0.9,
            constraint_violation=0.52,
            risk=0.72,
            vulnerability=0.6,
            validation_score=0.28,
        ),
        make_candidate(
            "balanced-best",
            objective_score=0.86,
            predicted_score=0.82,
            novelty=0.71,
            spread=0.69,
            mathematical_consistency=0.88,
            proof_strength=0.85,
            analytical_clarity=0.86,
            evidence=0.91,
            reproducibility=0.92,
            consistency=0.9,
            safety_margin=0.96,
            robustness=0.94,
            forecast_confidence=0.82,
            trend_alignment=0.79,
            resource_efficiency=0.84,
            budget_fit=0.86,
            throughput=0.77,
            causal_confidence=0.88,
            explainability=0.86,
            counterfactual_support=0.83,
            adversarial_resilience=0.93,
            edge_case_coverage=0.9,
            synthesis_quality=0.88,
            coherence=0.87,
            versatility=0.83,
            validation_score=0.95,
            feasibility=0.92,
            resource_cost=0.24,
            risk=0.05,
            vulnerability=0.05,
        ),
        make_candidate(
            "validator",
            validation_score=0.99,
            evidence=0.99,
            feasibility=0.97,
            reproducibility=0.96,
            objective_score=0.74,
            safety_margin=0.84,
        ),
        make_candidate(
            "explorer",
            novelty=0.99,
            spread=0.97,
            objective_score=0.52,
            safety_margin=0.28,
            validation_score=0.32,
            risk=0.4,
        ),
        make_candidate(
            "resource-saver",
            resource_efficiency=0.97,
            budget_fit=0.98,
            throughput=0.9,
            resource_cost=0.05,
            objective_score=0.68,
            validation_score=0.7,
        ),
    ]


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 17: MULTI-AGENT QSEE EVOLUTION")
    print("=" * 60)

    population = AgentPopulation(
        [
            SpecializedAgent(
                id=f"{specialization}-agent", specialization=specialization
            )
            for specialization in SPECIALIZATIONS
        ]
    )
    candidates = build_candidates()
    knowledge_graph = KnowledgeGraph() if KnowledgeGraph is not None else None

    collective = collective_solution(
        population, candidates, knowledge_graph=knowledge_graph
    )
    comparison = compare_individual_vs_collective(
        population, candidates, objective_fn, knowledge_graph=knowledge_graph
    )

    print("[competition]  per-specialization top picks")
    for agent in population.agents:
        top = collective.competition_rankings[agent.id][0]
        print(f"  - {agent.specialization:20s} -> {top.candidate_id:16s} score={top.score:6.3f}")

    print("[coalitions]   complementary specialist groupings")
    if collective.coalitions:
        for coalition in collective.coalitions:
            print(
                f"  - focus={coalition.focus_candidate_id:16s} "
                f"strength={coalition.strength:5.3f} "
                f"members={coalition.member_ids}"
            )
    else:
        print("  - no coalitions formed")

    print("[collective]   final cooperative ranking")
    for row in collective.collective_ranking:
        print(f"  - rank {row.rank:>2}: {row.candidate_id:16s} aggregate={row.score:6.3f}")

    print("[comparison]   individual intelligence vs collective intelligence")
    print(
        f"  true best candidate  : {comparison.true_best_candidate_id} "
        f"(objective={comparison.true_best_objective:.3f})"
    )
    print(
        f"  collective choice    : {comparison.collective_candidate_id} "
        f"(objective={comparison.collective_objective:.3f})"
    )
    print(f"  average individual   : objective={comparison.average_individual_objective:.3f}")
    print(f"  best individual      : objective={comparison.best_individual_objective:.3f}")
    print(f"  collective - average : {comparison.collective_minus_average:+.3f}")
    print(f"  collective - best    : {comparison.collective_minus_best:+.3f}")
    print(
        f"  outperformed agents  : "
        f"{len(comparison.agents_outperformed_by_collective)}/{len(population.agents)}"
    )

    if knowledge_graph is not None:
        print(
            f"[knowledge]    knowledge-graph observations recorded: "
            f"{len(knowledge_graph.nodes(kind='observation'))}"
        )

    elapsed = time.perf_counter() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print(f"  candidates evaluated : {len(candidates)}")
    print(f"  specialist heuristics: {len(population.agents)}")
    print("  These 'agents' are lightweight specialized scoring/search heuristics,")
    print("  not autonomous AI agents or LLMs, and this demo is ordinary classical")
    print("  computation over Python dicts and numbers.")


if __name__ == "__main__":
    main()
