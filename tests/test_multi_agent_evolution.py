from __future__ import annotations

import pytest

from qes.knowledge_graph import KnowledgeGraph
from qes.multi_agent_evolution import (
    SPECIALIZATIONS,
    AgentPopulation,
    ScoredCandidate,
    SpecializedAgent,
    _ranking_overlap,
    collective_solution,
    compare_individual_vs_collective,
)


def make_candidate(candidate_id: str, **overrides: float) -> dict[str, object]:
    base: dict[str, object] = {
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
    base.update(overrides)
    return base


def make_full_population() -> AgentPopulation:
    return AgentPopulation(
        [
            SpecializedAgent(
                id=f"{specialization}-agent", specialization=specialization
            )
            for specialization in SPECIALIZATIONS
        ]
    )


def ground_truth_objective(candidate: dict[str, object] | object) -> float:
    assert isinstance(candidate, dict)
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


class TestSpecializedAgentScoring:
    def test_specialization_list_covers_all_eleven_roles(self) -> None:
        assert len(SPECIALIZATIONS) == 11
        assert set(SPECIALIZATIONS) == {
            "exploration",
            "mathematics",
            "optimization",
            "verification",
            "safety",
            "prediction",
            "resource_allocation",
            "causal_analysis",
            "adversarial_testing",
            "synthesis",
            "validation",
        }

    def test_invalid_specialization_rejected(self) -> None:
        with pytest.raises(ValueError, match="specialization"):
            SpecializedAgent(id="bad", specialization="generalist")

    def test_agent_constructor_validates_id_and_weight(self) -> None:
        with pytest.raises(ValueError, match="id must be a non-empty string"):
            SpecializedAgent(id="", specialization="exploration")
        with pytest.raises(TypeError, match="weight must be a finite number"):
            SpecializedAgent(id="bad-weight", specialization="exploration", weight=True)
        with pytest.raises(ValueError, match="weight must be finite"):
            SpecializedAgent(
                id="bad-weight",
                specialization="exploration",
                weight=float("inf"),
            )
        with pytest.raises(ValueError, match="weight must be > 0"):
            SpecializedAgent(id="bad-weight", specialization="exploration", weight=0.0)

    @pytest.mark.parametrize(
        ("specialization", "preferred", "other"),
        [
            (
                "exploration",
                make_candidate("a", novelty=0.95, spread=0.9),
                make_candidate("b", novelty=0.2, spread=0.2),
            ),
            (
                "mathematics",
                make_candidate(
                    "a",
                    mathematical_consistency=0.95,
                    proof_strength=0.9,
                    analytical_clarity=0.9,
                ),
                make_candidate(
                    "b",
                    mathematical_consistency=0.2,
                    proof_strength=0.2,
                    analytical_clarity=0.2,
                ),
            ),
            (
                "optimization",
                make_candidate("a", objective_score=0.98, efficiency=0.9),
                make_candidate("b", objective_score=0.2, efficiency=0.2),
            ),
            (
                "verification",
                make_candidate(
                    "a", evidence=0.95, reproducibility=0.95, consistency=0.9
                ),
                make_candidate("b", evidence=0.2, reproducibility=0.2, consistency=0.2),
            ),
            (
                "safety",
                make_candidate("a", safety_margin=0.96, robustness=0.94),
                make_candidate("b", safety_margin=0.2, robustness=0.2, risk=0.8),
            ),
            (
                "prediction",
                make_candidate(
                    "a",
                    predicted_score=0.95,
                    forecast_confidence=0.9,
                    trend_alignment=0.9,
                ),
                make_candidate(
                    "b",
                    predicted_score=0.2,
                    forecast_confidence=0.2,
                    trend_alignment=0.2,
                ),
            ),
            (
                "resource_allocation",
                make_candidate(
                    "a",
                    resource_efficiency=0.95,
                    budget_fit=0.9,
                    throughput=0.9,
                    resource_cost=0.1,
                ),
                make_candidate(
                    "b",
                    resource_efficiency=0.2,
                    budget_fit=0.2,
                    throughput=0.2,
                    resource_cost=0.7,
                ),
            ),
            (
                "causal_analysis",
                make_candidate(
                    "a",
                    causal_confidence=0.95,
                    explainability=0.9,
                    counterfactual_support=0.92,
                ),
                make_candidate(
                    "b",
                    causal_confidence=0.2,
                    explainability=0.2,
                    counterfactual_support=0.2,
                ),
            ),
            (
                "adversarial_testing",
                make_candidate(
                    "a",
                    adversarial_resilience=0.95,
                    edge_case_coverage=0.9,
                    vulnerability=0.05,
                ),
                make_candidate(
                    "b",
                    adversarial_resilience=0.2,
                    edge_case_coverage=0.2,
                    vulnerability=0.9,
                ),
            ),
            (
                "synthesis",
                make_candidate(
                    "a",
                    synthesis_quality=0.95,
                    coherence=0.92,
                    versatility=0.9,
                    complexity=0.1,
                ),
                make_candidate(
                    "b",
                    synthesis_quality=0.2,
                    coherence=0.2,
                    versatility=0.2,
                    complexity=0.8,
                ),
            ),
            (
                "validation",
                make_candidate(
                    "a",
                    validation_score=0.95,
                    feasibility=0.92,
                    reproducibility=0.9,
                    evidence=0.9,
                ),
                make_candidate(
                    "b",
                    validation_score=0.2,
                    feasibility=0.2,
                    reproducibility=0.2,
                    evidence=0.2,
                ),
            ),
        ],
    )
    def test_each_specialization_prefers_relevant_signal(
        self,
        specialization: str,
        preferred: dict[str, object],
        other: dict[str, object],
    ) -> None:
        agent = SpecializedAgent(id=f"{specialization}-agent", specialization=specialization)
        assert agent.score_candidate(preferred) > agent.score_candidate(other)

    def test_safety_penalizes_constraint_violations_more_than_optimization(
        self,
    ) -> None:
        risky = make_candidate(
            "risky",
            objective_score=0.95,
            efficiency=0.9,
            constraint_violation=0.6,
            risk=0.7,
        )
        safe = make_candidate(
            "safe",
            objective_score=0.8,
            efficiency=0.75,
            constraint_violation=0.0,
            risk=0.0,
            safety_margin=0.95,
        )
        safety_agent = SpecializedAgent(id="s1", specialization="safety")
        optimization_agent = SpecializedAgent(id="o1", specialization="optimization")
        assert safety_agent.score_candidate(safe) > safety_agent.score_candidate(risky)
        assert (
            optimization_agent.score_candidate(risky)
            > safety_agent.score_candidate(risky)
        )

    def test_rank_candidates_records_performance_history(self) -> None:
        agent = SpecializedAgent(id="hist", specialization="optimization")
        ranking = agent.rank_candidates(
            [
                make_candidate("a", objective_score=0.9),
                make_candidate("b", objective_score=0.2),
            ]
        )
        assert [row.candidate_id for row in ranking] == ["a", "b"]
        assert len(agent.performance_history) == 1
        record = agent.performance_history[0]
        assert record.best_candidate_id == "a"
        assert record.candidate_count == 2

    def test_rank_candidates_caps_performance_history_at_sixty_four_entries(self) -> None:
        agent = SpecializedAgent(id="hist", specialization="optimization")
        for idx in range(65):
            agent.rank_candidates(
                [
                    make_candidate(f"a-{idx}", objective_score=0.9),
                    make_candidate(f"b-{idx}", objective_score=0.2),
                ]
            )
        assert len(agent.performance_history) == 64

    def test_non_mapping_candidate_rejected(self) -> None:
        agent = SpecializedAgent(id="x", specialization="exploration")
        with pytest.raises(TypeError):
            agent.score_candidate(["not", "a", "mapping"])  # type: ignore[arg-type]


class TestAgentPopulation:
    def test_population_rejects_duplicate_agent_ids(self) -> None:
        with pytest.raises(ValueError, match="unique ids"):
            AgentPopulation(
                [
                    SpecializedAgent(id="dup", specialization="exploration"),
                    SpecializedAgent(id="dup", specialization="safety"),
                ]
            )

    def test_competition_returns_rankings_for_each_agent(self) -> None:
        population = AgentPopulation(
            [
                SpecializedAgent(id="exp", specialization="exploration"),
                SpecializedAgent(id="opt", specialization="optimization"),
            ]
        )
        results = population.competition(
            [
                make_candidate("wide", novelty=0.95, spread=0.9),
                make_candidate("strong", objective_score=0.95, efficiency=0.9),
            ]
        )
        assert set(results) == {"exp", "opt"}
        assert results["exp"][0].candidate_id == "wide"
        assert results["opt"][0].candidate_id == "strong"

    def test_competition_requires_nonempty_population(self) -> None:
        population = AgentPopulation([])
        with pytest.raises(ValueError, match="at least one agent"):
            population.competition([make_candidate("only")])

    def test_competition_requires_candidates(self) -> None:
        population = AgentPopulation([SpecializedAgent(id="exp", specialization="exploration")])
        with pytest.raises(ValueError, match="at least one candidate"):
            population.competition([])

    def test_competition_rejects_non_mapping_candidates(self) -> None:
        population = AgentPopulation([SpecializedAgent(id="exp", specialization="exploration")])
        with pytest.raises(TypeError, match="mapping objects"):
            population.competition([object()])  # type: ignore[list-item]

    def test_cooperation_aggregates_rankings_by_weighted_borda(self) -> None:
        population = AgentPopulation(
            [
                SpecializedAgent(id="a", specialization="exploration", weight=2.0),
                SpecializedAgent(id="b", specialization="optimization", weight=1.0),
            ]
        )
        rankings = {
            "a": [
                population.agents[0].rank_candidates(
                    [
                        make_candidate("x", novelty=0.9),
                        make_candidate("y", novelty=0.1),
                    ]
                )[0],
                population.agents[0].rank_candidates(
                    [
                        make_candidate("x", novelty=0.9),
                        make_candidate("y", novelty=0.1),
                    ]
                )[1],
            ],
            "b": [
                population.agents[1].rank_candidates(
                    [
                        make_candidate("y", objective_score=0.95),
                        make_candidate("x", objective_score=0.2),
                    ]
                )[0],
                population.agents[1].rank_candidates(
                    [
                        make_candidate("y", objective_score=0.95),
                        make_candidate("x", objective_score=0.2),
                    ]
                )[1],
            ],
        }
        aggregate = population.cooperation(rankings)
        assert aggregate[0].candidate_id == "x"
        assert {row.candidate_id for row in aggregate} == {"x", "y"}

    def test_cooperation_rejects_empty_results(self) -> None:
        population = AgentPopulation([SpecializedAgent(id="a", specialization="exploration")])
        with pytest.raises(ValueError, match="non-empty"):
            population.cooperation({})

    def test_cooperation_skips_agents_without_rankings_and_rejects_all_empty(self) -> None:
        population = AgentPopulation(
            [
                SpecializedAgent(id="a", specialization="exploration"),
                SpecializedAgent(id="b", specialization="optimization"),
            ]
        )
        ranking = population.agents[0].rank_candidates([make_candidate("x", novelty=0.9)])
        aggregate = population.cooperation({"a": ranking, "b": []})
        assert [row.candidate_id for row in aggregate] == ["x"]
        with pytest.raises(ValueError, match="did not contain any rankings"):
            population.cooperation({"a": [], "b": []})

    def test_knowledge_exchange_returns_insights_without_graph(self) -> None:
        population = AgentPopulation([SpecializedAgent(id="a", specialization="exploration")])
        results = population.competition([make_candidate("x", novelty=0.9), make_candidate("y", novelty=0.1)])
        insights = population.knowledge_exchange(results)
        assert len(insights) == 1
        assert insights[0].candidate_id == "x"
        assert insights[0].graph_node_id is None

    def test_knowledge_exchange_records_nodes_when_graph_available(self) -> None:
        graph = KnowledgeGraph()
        population = AgentPopulation(
            [
                SpecializedAgent(id="a", specialization="exploration"),
                SpecializedAgent(id="b", specialization="validation"),
            ]
        )
        results = population.competition(
            [
                make_candidate("x", novelty=0.9),
                make_candidate("y", validation_score=0.9),
            ]
        )
        insights = population.knowledge_exchange(results, knowledge_graph=graph)
        assert len(insights) == 2
        assert all(insight.graph_node_id is not None for insight in insights)
        assert len(graph.nodes(kind="observation")) == 2

    def test_knowledge_exchange_tolerates_graph_failures(self) -> None:
        class BrokenGraph:
            def add_node(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("boom")

        population = AgentPopulation([SpecializedAgent(id="a", specialization="exploration")])
        results = population.competition([make_candidate("x", novelty=0.9), make_candidate("y", novelty=0.1)])
        insights = population.knowledge_exchange(results, knowledge_graph=BrokenGraph())
        assert len(insights) == 1
        assert insights[0].graph_node_id is None

    def test_knowledge_exchange_rejects_invalid_top_n(self) -> None:
        population = AgentPopulation([SpecializedAgent(id="a", specialization="exploration")])
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            population.knowledge_exchange({"a": []}, top_n=0)

    def test_form_coalitions_groups_complementary_specialists(self) -> None:
        population = AgentPopulation(
            [
                SpecializedAgent(id="exp", specialization="exploration"),
                SpecializedAgent(id="safe", specialization="safety"),
                SpecializedAgent(id="val", specialization="validation"),
            ]
        )
        candidates = [
            make_candidate(
                "balanced",
                novelty=0.8,
                spread=0.75,
                safety_margin=0.95,
                validation_score=0.95,
                evidence=0.9,
            ),
            make_candidate(
                "extreme",
                novelty=0.95,
                spread=0.95,
                safety_margin=0.1,
                validation_score=0.2,
            ),
        ]
        results = population.competition(candidates)
        coalitions = population.form_coalitions(results)
        assert len(coalitions) == 1
        assert set(coalitions[0].member_ids) == {"exp", "safe", "val"}
        assert coalitions[0].focus_candidate_id == "balanced"

    def test_form_coalitions_can_return_empty_when_agents_disagree(self) -> None:
        population = AgentPopulation(
            [
                SpecializedAgent(id="exp", specialization="exploration"),
                SpecializedAgent(id="opt", specialization="optimization"),
            ]
        )
        candidates = [
            make_candidate("novel", novelty=0.95, spread=0.9, objective_score=0.2),
            make_candidate("profit", novelty=0.1, spread=0.1, objective_score=0.98, efficiency=0.95),
        ]
        results = population.competition(candidates)
        assert population.form_coalitions(results, top_n=1) == []

    def test_form_coalitions_rejects_invalid_size(self) -> None:
        population = AgentPopulation([SpecializedAgent(id="a", specialization="exploration")])
        with pytest.raises(ValueError, match="max_size"):
            population.form_coalitions({"a": []}, max_size=1)

    def test_form_coalitions_rejects_invalid_top_n_and_skips_empty_rankings(self) -> None:
        population = AgentPopulation(
            [
                SpecializedAgent(id="a", specialization="exploration"),
                SpecializedAgent(id="b", specialization="optimization"),
            ]
        )
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            population.form_coalitions({"a": []}, top_n=0)
        assert population.form_coalitions({"a": [], "b": []}) == []
        ranking = population.agents[0].rank_candidates([make_candidate("x", novelty=0.9)])
        assert population.form_coalitions({"a": ranking, "b": []}) == []


class TestCollectivePipeline:
    def test_collective_solution_end_to_end_returns_valid_result(self) -> None:
        population = make_full_population()
        candidates = [
            make_candidate(
                "risky-optimizer",
                objective_score=0.98,
                efficiency=0.95,
                predicted_score=0.9,
                constraint_violation=0.45,
                risk=0.6,
                vulnerability=0.55,
            ),
            make_candidate(
                "balanced-best",
                objective_score=0.84,
                predicted_score=0.8,
                novelty=0.7,
                spread=0.68,
                mathematical_consistency=0.86,
                proof_strength=0.84,
                evidence=0.9,
                reproducibility=0.9,
                consistency=0.88,
                safety_margin=0.95,
                robustness=0.93,
                forecast_confidence=0.8,
                trend_alignment=0.78,
                resource_efficiency=0.82,
                budget_fit=0.84,
                throughput=0.76,
                causal_confidence=0.86,
                explainability=0.85,
                counterfactual_support=0.82,
                adversarial_resilience=0.92,
                edge_case_coverage=0.9,
                synthesis_quality=0.87,
                coherence=0.86,
                versatility=0.82,
                validation_score=0.94,
                feasibility=0.91,
                resource_cost=0.25,
                risk=0.05,
                vulnerability=0.06,
            ),
            make_candidate(
                "validator",
                validation_score=0.98,
                evidence=0.98,
                feasibility=0.96,
                reproducibility=0.97,
                safety_margin=0.82,
                objective_score=0.72,
            ),
            make_candidate(
                "explorer",
                novelty=0.98,
                spread=0.96,
                objective_score=0.55,
                safety_margin=0.3,
                validation_score=0.35,
            ),
        ]
        result = collective_solution(population, candidates, knowledge_graph=KnowledgeGraph())
        assert result.selected_candidate_id == "balanced-best"
        assert result.collective_ranking[0].candidate_id == "balanced-best"
        assert result.coalitions
        assert len(result.shared_insights) == len(SPECIALIZATIONS)
        assert set(result.aggregate_scores) == {candidate["id"] for candidate in candidates}

    def test_collective_solution_handles_single_candidate(self) -> None:
        population = make_full_population()
        result = collective_solution(population, [make_candidate("only")])
        assert result.selected_candidate_id == "only"
        assert len(result.collective_ranking) == 1

    def test_collective_solution_rejects_duplicate_candidate_ids(self) -> None:
        population = make_full_population()
        with pytest.raises(ValueError, match="duplicate candidate id"):
            collective_solution(population, [make_candidate("dup"), make_candidate("dup")])

    def test_compare_individual_vs_collective_reports_meaningful_metrics(
        self,
    ) -> None:
        population = make_full_population()
        candidates = [
            make_candidate(
                "risky-optimizer",
                objective_score=0.99,
                efficiency=0.96,
                predicted_score=0.9,
                constraint_violation=0.5,
                risk=0.7,
                vulnerability=0.6,
                validation_score=0.3,
            ),
            make_candidate(
                "balanced-best",
                objective_score=0.85,
                predicted_score=0.81,
                novelty=0.7,
                spread=0.68,
                mathematical_consistency=0.88,
                proof_strength=0.84,
                analytical_clarity=0.86,
                evidence=0.9,
                reproducibility=0.91,
                consistency=0.89,
                safety_margin=0.96,
                robustness=0.94,
                forecast_confidence=0.82,
                trend_alignment=0.78,
                resource_efficiency=0.84,
                budget_fit=0.86,
                throughput=0.77,
                causal_confidence=0.88,
                explainability=0.86,
                counterfactual_support=0.82,
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
        ]
        comparison = compare_individual_vs_collective(
            population, candidates, ground_truth_objective
        )
        assert comparison.true_best_candidate_id == "balanced-best"
        assert comparison.collective_candidate_id == "balanced-best"
        assert comparison.collective_objective >= comparison.average_individual_objective
        assert len(comparison.agents_outperformed_by_collective) >= 8

    def test_compare_individual_vs_collective_requires_callable_objective(self) -> None:
        population = make_full_population()
        with pytest.raises(TypeError, match="objective_fn"):
            compare_individual_vs_collective(population, [make_candidate("x")], "not-callable")  # type: ignore[arg-type]

    def test_custom_candidate_ids_are_preserved(self) -> None:
        population = AgentPopulation([SpecializedAgent(id="exp", specialization="exploration")])
        results = population.competition([{"id": "custom-id", "novelty": 1.0, "spread": 1.0}])
        assert results["exp"][0].candidate_id == "custom-id"

    def test_candidate_id_must_be_nonempty_string(self) -> None:
        population = AgentPopulation([SpecializedAgent(id="exp", specialization="exploration")])
        with pytest.raises(ValueError, match="candidate id"):
            population.competition([{"id": "", "novelty": 1.0}])


def test_ranking_overlap_returns_zero_for_two_empty_rankings() -> None:
    assert _ranking_overlap([], [], top_n=3) == pytest.approx(0.0)


def test_scored_candidate_supports_empty_rankings_for_overlap_helpers() -> None:
    row = ScoredCandidate(
        candidate_id="x",
        score=1.0,
        rank=1,
        candidate=make_candidate("x"),
    )
    assert _ranking_overlap([row], [], top_n=1) == pytest.approx(0.0)
