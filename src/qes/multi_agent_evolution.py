"""Phase 17 -- Multi-Agent QSEE evolution.

This module takes the repo's existing QSEE-11L idea ("eleven independent
streams") and extends it into a *classical* population of eleven lightweight
specialists that score the same candidate pool from different angles, then
compete, cooperate, exchange findings, form coalitions, and settle on one
collective solution.

These "agents" are intentionally heuristic scorers, not autonomous AI agents
or LLMs. The class name here is therefore `SpecializedAgent` rather than
plain `Agent`, because `qes.agent.Agent` already exists elsewhere in the
codebase with very different semantics (world-inhabiting entities with
memory, messaging, and policies).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Literal

SpecializationName = Literal[
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
]

SPECIALIZATIONS: tuple[SpecializationName, ...] = (
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
)

_SPECIALIZATION_FAMILY: dict[str, str] = {
    "exploration": "frontier",
    "prediction": "frontier",
    "adversarial_testing": "frontier",
    "mathematics": "rigor",
    "verification": "rigor",
    "causal_analysis": "rigor",
    "validation": "rigor",
    "optimization": "execution",
    "resource_allocation": "execution",
    "safety": "execution",
    "synthesis": "execution",
}


def _validate_specialization(value: str) -> SpecializationName:
    if value not in SPECIALIZATIONS:
        raise ValueError(f"specialization must be one of {list(SPECIALIZATIONS)}, got {value!r}")
    return value


def _validate_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return number


def _candidate_id(candidate: Mapping[str, object], index: int) -> str:
    raw_id = candidate.get("id", f"candidate-{index + 1}")
    if not isinstance(raw_id, str) or not raw_id:
        raise ValueError("candidate id must be a non-empty string")
    return raw_id


def _normalize_candidates(candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError("candidates must contain mapping objects")
        normalized_candidate = dict(candidate)
        candidate_id = _candidate_id(normalized_candidate, index)
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate id: {candidate_id!r}")
        normalized_candidate["id"] = candidate_id
        normalized.append(normalized_candidate)
        seen_ids.add(candidate_id)
    if not normalized:
        raise ValueError("at least one candidate is required")
    return normalized


def _metric(candidate: Mapping[str, object], name: str, default: float = 0.0) -> float:
    if name not in candidate:
        return default
    return _validate_finite_number(candidate[name], name=f"candidate[{name!r}]")


def _mean_metric(candidate: Mapping[str, object], names: Sequence[str], default: float = 0.0) -> float:
    values = [_metric(candidate, name) for name in names if name in candidate]
    return mean(values) if values else default


def _stable_sort_key(row: tuple[float, str]) -> tuple[float, str]:
    score, candidate_id = row
    return (-score, candidate_id)


def _ranking_overlap(left: Sequence[ScoredCandidate], right: Sequence[ScoredCandidate], top_n: int) -> float:
    left_ids = {row.candidate_id for row in left[:top_n]}
    right_ids = {row.candidate_id for row in right[:top_n]}
    if not left_ids and not right_ids:
        return 0.0
    universe = left_ids | right_ids
    return len(left_ids & right_ids) / max(1, len(universe))


@dataclass(frozen=True)
class ScoredCandidate:
    """One candidate plus a specialization- or population-level score."""

    candidate_id: str
    score: float
    rank: int
    candidate: dict[str, object]


@dataclass(frozen=True)
class AgentPerformanceRecord:
    """Compact history entry for one competition round."""

    best_candidate_id: str
    best_score: float
    mean_score: float
    candidate_count: int


@dataclass(frozen=True)
class SharedInsight:
    """Best-findings record emitted during knowledge exchange."""

    agent_id: str
    specialization: str
    candidate_id: str
    score: float
    summary: str
    graph_node_id: str | None = None


@dataclass(frozen=True)
class Coalition:
    """A small alliance of complementary specialists focused on one candidate."""

    member_ids: list[str]
    specializations: list[str]
    focus_candidate_id: str
    strength: float


@dataclass(frozen=True)
class CollectiveSolutionResult:
    """End-to-end population decision result."""

    selected_candidate_id: str
    selected_candidate: dict[str, object]
    cooperative_ranking: list[ScoredCandidate]
    collective_ranking: list[ScoredCandidate]
    competition_rankings: dict[str, list[ScoredCandidate]]
    shared_insights: list[SharedInsight]
    coalitions: list[Coalition]
    aggregate_scores: dict[str, float]


@dataclass(frozen=True)
class ComparisonResult:
    """Comparison between individual picks and the collective pick."""

    true_best_candidate_id: str
    true_best_objective: float
    collective_candidate_id: str
    collective_objective: float
    individual_best_picks: dict[str, str]
    individual_objective_scores: dict[str, float]
    average_individual_objective: float
    best_individual_objective: float
    collective_minus_average: float
    collective_minus_best: float
    agents_outperformed_by_collective: list[str]


@dataclass
class SpecializedAgent:
    """One lightweight specialist in the Phase 17 population.

    Each specialization uses a documented weighted heuristic over candidate
    features. The heuristics are intentionally simple and classical: they are
    not claims of domain mastery, only role-specific scoring rules that let a
    population diversify search pressure.
    """

    id: str
    specialization: str
    weight: float = 1.0
    performance_history: list[AgentPerformanceRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")
        self.specialization = _validate_specialization(self.specialization)
        self.weight = _validate_finite_number(self.weight, name="weight")
        if self.weight <= 0.0:
            raise ValueError("weight must be > 0")

    def score_candidate(self, candidate: Mapping[str, object]) -> float:
        """Return this agent's heuristic score for one candidate."""
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate must be a mapping")

        objective = _metric(candidate, "objective_score")
        predicted = _metric(candidate, "predicted_score")
        novelty = _mean_metric(candidate, ("novelty", "diversity", "unexplored_potential"))
        spread = _mean_metric(candidate, ("spread", "breadth", "coverage"))
        math_quality = _mean_metric(
            candidate, ("mathematical_consistency", "proof_strength", "analytical_clarity")
        )
        evidence = _mean_metric(candidate, ("evidence", "evidence_strength"))
        verification = _mean_metric(candidate, ("reproducibility", "consistency"))
        safety_margin = _mean_metric(candidate, ("safety_margin", "robustness"))
        forecast = _mean_metric(candidate, ("forecast_confidence", "trend_alignment"))
        resource_efficiency = _mean_metric(candidate, ("resource_efficiency", "budget_fit", "throughput"))
        causal = _mean_metric(candidate, ("causal_confidence", "explainability", "counterfactual_support"))
        adversarial = _mean_metric(candidate, ("adversarial_resilience", "edge_case_coverage", "robustness"))
        synthesis = _mean_metric(candidate, ("synthesis_quality", "coherence", "versatility"))
        validation = _mean_metric(candidate, ("validation_score", "feasibility", "reproducibility"))
        simplicity = _metric(
            candidate,
            "simplicity",
            default=max(0.0, 1.0 - _metric(candidate, "complexity")),
        )
        efficiency = _metric(candidate, "efficiency")
        complexity = _metric(candidate, "complexity")
        constraint_violation = _metric(candidate, "constraint_violation")
        uncertainty = _metric(candidate, "uncertainty")
        risk = _metric(candidate, "risk")
        resource_cost = _metric(candidate, "resource_cost")
        vulnerability = _metric(candidate, "vulnerability")

        if self.specialization == "exploration":
            score = 1.4 * novelty + 1.0 * spread + 0.4 * predicted - 0.3 * constraint_violation - 0.2 * risk
        elif self.specialization == "mathematics":
            score = 1.3 * math_quality + 0.6 * simplicity + 0.4 * evidence - 0.5 * uncertainty
        elif self.specialization == "optimization":
            score = (
                1.8 * objective
                + 0.8 * efficiency
                + 0.5 * predicted
                - 0.4 * resource_cost
                - 0.6 * constraint_violation
            )
        elif self.specialization == "verification":
            score = (
                1.2 * evidence
                + 1.0 * verification
                + 0.5 * math_quality
                - 0.8 * constraint_violation
                - 0.6 * uncertainty
            )
        elif self.specialization == "safety":
            score = 1.4 * safety_margin + 0.4 * validation - 2.0 * constraint_violation - 1.0 * risk
        elif self.specialization == "prediction":
            score = 1.3 * predicted + 1.0 * forecast + 0.3 * novelty - 0.8 * uncertainty
        elif self.specialization == "resource_allocation":
            score = 1.2 * resource_efficiency + 0.4 * efficiency - 1.1 * resource_cost - 0.2 * complexity
        elif self.specialization == "causal_analysis":
            score = 1.3 * causal + 0.4 * evidence - 0.7 * uncertainty
        elif self.specialization == "adversarial_testing":
            score = 1.2 * adversarial + 0.3 * safety_margin - 1.4 * vulnerability - 0.5 * constraint_violation
        elif self.specialization == "synthesis":
            score = 1.2 * synthesis + 0.3 * objective + 0.2 * novelty - 0.5 * complexity
        else:  # validation
            score = (
                1.2 * validation
                + 0.9 * evidence
                + 0.4 * safety_margin
                - 0.6 * uncertainty
                - 0.4 * constraint_violation
            )

        return _validate_finite_number(score, name="score")

    def rank_candidates(self, candidates: Sequence[Mapping[str, object]]) -> list[ScoredCandidate]:
        """Score and rank a candidate pool from best to worst."""
        normalized = _normalize_candidates(candidates)
        scored_rows = [
            (self.score_candidate(candidate), str(candidate["id"]), candidate)
            for candidate in normalized
        ]
        scored_rows.sort(key=lambda row: _stable_sort_key((row[0], row[1])))
        ranking = [
            ScoredCandidate(
                candidate_id=candidate_id,
                score=score,
                rank=rank,
                candidate=dict(candidate),
            )
            for rank, (score, candidate_id, candidate) in enumerate(scored_rows, start=1)
        ]
        mean_score = mean(row.score for row in ranking)
        self.performance_history.append(
            AgentPerformanceRecord(
                best_candidate_id=ranking[0].candidate_id,
                best_score=ranking[0].score,
                mean_score=mean_score,
                candidate_count=len(ranking),
            )
        )
        if len(self.performance_history) > 64:
            del self.performance_history[0]
        return ranking


class AgentPopulation:
    """Population manager for competition, cooperation, exchange, and coalitions."""

    def __init__(self, agents: Sequence[SpecializedAgent]):
        validated_agents = list(agents)
        ids = [agent.id for agent in validated_agents]
        if len(ids) != len(set(ids)):
            raise ValueError("agents must have unique ids")
        self.agents = validated_agents

    def competition(self, candidates: Sequence[Mapping[str, object]]) -> dict[str, list[ScoredCandidate]]:
        """Have every specialist independently rank the same candidate pool."""
        if not self.agents:
            raise ValueError("competition requires at least one agent")
        normalized = _normalize_candidates(candidates)
        return {agent.id: agent.rank_candidates(normalized) for agent in self.agents}

    def cooperation(
        self,
        competition_results: Mapping[str, Sequence[ScoredCandidate]],
        *,
        top_k: int | None = None,
    ) -> list[ScoredCandidate]:
        """Aggregate rankings with a weighted Borda-style rank sum."""
        if not competition_results:
            raise ValueError("competition_results must be non-empty")

        aggregate_scores: dict[str, float] = {}
        candidate_lookup: dict[str, dict[str, object]] = {}
        for agent in self.agents:
            ranking = list(competition_results.get(agent.id, ()))
            if not ranking:
                continue
            pool_size = len(ranking)
            for row in ranking:
                candidate_lookup[row.candidate_id] = dict(row.candidate)
                borda_points = (pool_size - row.rank + 1) * agent.weight
                tie_break = max(0.0, row.score) * 0.01 * agent.weight
                aggregate_scores[row.candidate_id] = (
                    aggregate_scores.get(row.candidate_id, 0.0)
                    + borda_points
                    + tie_break
                )

        if not aggregate_scores:
            raise ValueError("competition_results did not contain any rankings")

        ordered = sorted(
            ((score, candidate_id) for candidate_id, score in aggregate_scores.items()),
            key=_stable_sort_key,
        )
        ranking = [
            ScoredCandidate(
                candidate_id=candidate_id,
                score=score,
                rank=rank,
                candidate=dict(candidate_lookup[candidate_id]),
            )
            for rank, (score, candidate_id) in enumerate(ordered, start=1)
        ]
        return ranking if top_k is None else ranking[:top_k]

    def knowledge_exchange(
        self,
        competition_results: Mapping[str, Sequence[ScoredCandidate]],
        *,
        knowledge_graph: object | None = None,
        top_n: int = 1,
    ) -> list[SharedInsight]:
        """Share each specialist's best findings, optionally logging them."""
        if top_n < 1:
            raise ValueError("top_n must be >= 1")

        insights: list[SharedInsight] = []
        add_node = getattr(knowledge_graph, "add_node", None)
        for agent in self.agents:
            for row in list(competition_results.get(agent.id, ()))[:top_n]:
                summary = (
                    f"{agent.specialization} specialist {agent.id} prefers "
                    f"{row.candidate_id} with heuristic score {row.score:.3f}"
                )
                graph_node_id: str | None = None
                if callable(add_node):
                    try:
                        node = add_node(
                            kind="observation",
                            payload={
                                "agent_id": agent.id,
                                "specialization": agent.specialization,
                                "candidate_id": row.candidate_id,
                                "score": row.score,
                                "summary": summary,
                            },
                        )
                        graph_node_id = getattr(node, "id", None)
                    except Exception:
                        graph_node_id = None
                insights.append(
                    SharedInsight(
                        agent_id=agent.id,
                        specialization=agent.specialization,
                        candidate_id=row.candidate_id,
                        score=row.score,
                        summary=summary,
                        graph_node_id=graph_node_id,
                    )
                )
        return insights

    def form_coalitions(
        self,
        competition_results: Mapping[str, Sequence[ScoredCandidate]],
        *,
        max_size: int = 3,
        top_n: int = 3,
    ) -> list[Coalition]:
        """Form small coalitions from ranking overlap plus family complementarity."""
        if max_size < 2:
            raise ValueError("max_size must be >= 2")
        if top_n < 1:
            raise ValueError("top_n must be >= 1")

        available = [agent.id for agent in self.agents if agent.id in competition_results]
        coalitions: list[Coalition] = []
        agent_lookup = {agent.id: agent for agent in self.agents}

        while available:
            seed_id = available.pop(0)
            seed_agent = agent_lookup[seed_id]
            seed_ranking = list(competition_results[seed_id])
            if not seed_ranking:
                continue

            partner_scores: list[tuple[float, str]] = []
            for other_id in available:
                other_agent = agent_lookup[other_id]
                other_ranking = list(competition_results[other_id])
                if not other_ranking:
                    continue
                overlap = _ranking_overlap(seed_ranking, other_ranking, top_n)
                family_bonus = (
                    0.35
                    if _SPECIALIZATION_FAMILY[seed_agent.specialization]
                    != _SPECIALIZATION_FAMILY[other_agent.specialization]
                    else 0.1
                )
                shared_focus = (
                    0.35
                    if seed_ranking[0].candidate_id == other_ranking[0].candidate_id
                    else 0.0
                )
                partner_score = overlap + family_bonus + shared_focus
                if partner_score >= 0.55:
                    partner_scores.append((partner_score, other_id))

            partner_scores.sort(key=lambda row: (-row[0], row[1]))
            chosen_partner_ids = [other_id for _, other_id in partner_scores[: max_size - 1]]
            if not chosen_partner_ids:
                continue

            members = [seed_id, *chosen_partner_ids]
            available = [agent_id for agent_id in available if agent_id not in chosen_partner_ids]
            focus_votes = Counter(
                candidate_id
                for member_id in members
                for candidate_id in [row.candidate_id for row in list(competition_results[member_id])[:2]]
            )
            focus_candidate_id = sorted(focus_votes.items(), key=lambda row: (-row[1], row[0]))[0][0]
            strengths = [score for score, other_id in partner_scores if other_id in chosen_partner_ids]
            coalitions.append(
                Coalition(
                    member_ids=members,
                    specializations=[agent_lookup[member_id].specialization for member_id in members],
                    focus_candidate_id=focus_candidate_id,
                    strength=mean(strengths),
                )
            )

        return coalitions


def collective_solution(
    population: AgentPopulation,
    candidates: Sequence[Mapping[str, object]],
    *,
    knowledge_graph: object | None = None,
) -> CollectiveSolutionResult:
    """Run competition -> cooperation -> knowledge exchange -> coalitions -> collective choice."""
    competition_rankings = population.competition(candidates)
    cooperative_ranking = population.cooperation(competition_rankings)
    shared_insights = population.knowledge_exchange(competition_rankings, knowledge_graph=knowledge_graph)
    coalitions = population.form_coalitions(competition_rankings)

    candidate_lookup = {row.candidate_id: dict(row.candidate) for row in cooperative_ranking}
    aggregate_scores = {row.candidate_id: row.score for row in cooperative_ranking}
    for coalition in coalitions:
        aggregate_scores[coalition.focus_candidate_id] = (
            aggregate_scores.get(coalition.focus_candidate_id, 0.0) + coalition.strength
        )

    ordered = sorted(
        ((score, candidate_id) for candidate_id, score in aggregate_scores.items()),
        key=_stable_sort_key,
    )
    collective_ranking = [
        ScoredCandidate(
            candidate_id=candidate_id,
            score=score,
            rank=rank,
            candidate=dict(candidate_lookup[candidate_id]),
        )
        for rank, (score, candidate_id) in enumerate(ordered, start=1)
    ]
    winner = collective_ranking[0]
    return CollectiveSolutionResult(
        selected_candidate_id=winner.candidate_id,
        selected_candidate=dict(candidate_lookup[winner.candidate_id]),
        cooperative_ranking=cooperative_ranking,
        collective_ranking=collective_ranking,
        competition_rankings=competition_rankings,
        shared_insights=shared_insights,
        coalitions=coalitions,
        aggregate_scores=aggregate_scores,
    )


def compare_individual_vs_collective(
    population: AgentPopulation,
    candidates: Sequence[Mapping[str, object]],
    objective_fn: Callable[[Mapping[str, object]], float],
    *,
    knowledge_graph: object | None = None,
) -> ComparisonResult:
    """Compare every individual's best pick against the population's collective pick."""
    if not callable(objective_fn):
        raise TypeError("objective_fn must be callable")

    normalized = _normalize_candidates(candidates)
    collective = collective_solution(population, normalized, knowledge_graph=knowledge_graph)
    objective_scores = {
        str(candidate["id"]): _validate_finite_number(objective_fn(candidate), name="objective score")
        for candidate in normalized
    }
    true_best_candidate_id, true_best_objective = max(
        objective_scores.items(), key=lambda row: (row[1], row[0])
    )

    individual_best_picks = {
        agent_id: ranking[0].candidate_id
        for agent_id, ranking in collective.competition_rankings.items()
        if ranking
    }
    individual_objective_scores = {
        agent_id: objective_scores[candidate_id]
        for agent_id, candidate_id in individual_best_picks.items()
    }
    average_individual_objective = mean(individual_objective_scores.values())
    best_individual_objective = max(individual_objective_scores.values())
    collective_objective = objective_scores[collective.selected_candidate_id]
    agents_outperformed_by_collective = sorted(
        [agent_id for agent_id, score in individual_objective_scores.items() if collective_objective >= score]
    )

    return ComparisonResult(
        true_best_candidate_id=true_best_candidate_id,
        true_best_objective=true_best_objective,
        collective_candidate_id=collective.selected_candidate_id,
        collective_objective=collective_objective,
        individual_best_picks=individual_best_picks,
        individual_objective_scores=individual_objective_scores,
        average_individual_objective=average_individual_objective,
        best_individual_objective=best_individual_objective,
        collective_minus_average=collective_objective - average_individual_objective,
        collective_minus_best=collective_objective - best_individual_objective,
        agents_outperformed_by_collective=agents_outperformed_by_collective,
    )
