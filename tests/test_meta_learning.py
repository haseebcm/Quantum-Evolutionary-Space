from __future__ import annotations

import math

import numpy as np
import pytest

from qes.meta_learning import (
    MetaController,
    ProblemStructure,
    SearchHistory,
    SearchOutcome,
    SearchStrategy,
    record_outcome,
)
from qes.patterns import Pattern, PatternMemory


def make_problem(**overrides: object) -> ProblemStructure:
    data: dict[str, object] = {
        "dimensionality": 6,
        "bounds_width": 2.0,
        "ruggedness": 0.35,
        "noise_level": 0.10,
        "constraint_count": 1,
        "gradient_available": True,
    }
    data.update(overrides)
    return ProblemStructure(**data)


def make_strategy(**overrides: object) -> SearchStrategy:
    data: dict[str, object] = {
        "optimizer": "adaptive_gradient",
        "mutation_rate": 0.08,
        "population_size": 32,
        "branching_factor": 4,
        "exploration_exploitation": 0.35,
        "compute_allocation": 0.7,
        "convergence_threshold": 0.001,
        "model_class": "AdaptiveGradientSearch",
    }
    data.update(overrides)
    return SearchStrategy(**data)


def make_history_with_good_and_bad_records(problem: ProblemStructure) -> SearchHistory:
    history = SearchHistory()
    good_strategy = make_strategy(
        mutation_rate=0.24,
        population_size=92,
        exploration_exploitation=0.22,
        compute_allocation=0.88,
        convergence_threshold=0.003,
    )
    bad_strategy = make_strategy(
        optimizer="evolutionary",
        mutation_rate=0.02,
        population_size=16,
        branching_factor=2,
        exploration_exploitation=0.72,
        compute_allocation=0.35,
        convergence_threshold=0.015,
        model_class="EvolutionaryPopulation",
    )
    record_outcome(history, problem, good_strategy, True, 5, 0.04)
    record_outcome(history, problem, good_strategy, True, 7, 0.05)
    record_outcome(history, problem, bad_strategy, False, 40, 3.5)
    record_outcome(history, problem, bad_strategy, False, 35, 3.1)
    return history


def test_problem_structure_validates_and_normalizes_feature_vector():
    problem = make_problem(dimensionality=12, bounds_width=4.0, constraint_count=3, gradient_available=False)

    vec = problem.feature_vector()

    assert vec.shape == (6,)
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)
    assert vec[-1] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("dimensionality", 0, ValueError),
        ("dimensionality", True, TypeError),
        ("bounds_width", 0.0, ValueError),
        ("ruggedness", 1.5, ValueError),
        ("noise_level", -0.1, ValueError),
        ("constraint_count", -1, ValueError),
        ("gradient_available", 1, TypeError),
    ],
)
def test_problem_structure_rejects_invalid_inputs(field: str, value: object, error_type: type[Exception]):
    kwargs = {
        "dimensionality": 6,
        "bounds_width": 2.0,
        "ruggedness": 0.3,
        "noise_level": 0.1,
        "constraint_count": 0,
        "gradient_available": True,
    }
    kwargs[field] = value
    with pytest.raises(error_type):
        ProblemStructure(**kwargs)


def test_search_strategy_serializes_round_trip():
    strategy = make_strategy()

    restored = SearchStrategy.from_mapping(strategy.to_mapping())

    assert restored == strategy


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("optimizer", "", ValueError),
        ("mutation_rate", 1.1, ValueError),
        ("population_size", 0, ValueError),
        ("branching_factor", 0, ValueError),
        ("exploration_exploitation", -0.1, ValueError),
        ("compute_allocation", 0.0, ValueError),
        ("convergence_threshold", 0.0, ValueError),
        ("model_class", "", ValueError),
    ],
)
def test_search_strategy_rejects_invalid_inputs(field: str, value: object, error_type: type[Exception]):
    kwargs = make_strategy().to_mapping()
    kwargs[field] = value
    with pytest.raises(error_type):
        SearchStrategy(**kwargs)


def test_search_outcome_round_trip_from_mapping():
    outcome = SearchOutcome(
        problem=make_problem(),
        strategy=make_strategy(),
        converged=True,
        generations_to_converge=8,
        final_score=0.25,
    )

    restored = SearchOutcome.from_mapping(outcome.to_mapping())

    assert restored.problem == outcome.problem
    assert restored.strategy == outcome.strategy
    assert restored.converged is True
    assert restored.generations_to_converge == 8
    assert restored.final_score == pytest.approx(0.25)


def test_search_outcome_rejects_non_finite_score():
    with pytest.raises(ValueError):
        SearchOutcome(
            problem=make_problem(),
            strategy=make_strategy(),
            converged=True,
            generations_to_converge=2,
            final_score=math.inf,
        )


def test_search_history_accepts_plain_dict_records():
    outcome = SearchOutcome(
        problem=make_problem(),
        strategy=make_strategy(),
        converged=True,
        generations_to_converge=6,
        final_score=0.12,
    )
    history = SearchHistory(raw_records=[outcome.to_mapping()])

    records = history.all_records()

    assert len(records) == 1
    assert isinstance(records[0], SearchOutcome)
    assert records[0].final_score == pytest.approx(0.12)


def test_search_history_reads_records_from_pattern_memory():
    pattern_memory = PatternMemory()
    history = SearchHistory(pattern_memory=pattern_memory)
    outcome = SearchOutcome(
        problem=make_problem(),
        strategy=make_strategy(),
        converged=True,
        generations_to_converge=4,
        final_score=0.08,
    )
    pattern_memory.store(
        Pattern(
            intent=history.intent,
            context={"source": "test"},
            payload=outcome.to_mapping(),
            phi=0.0,
            cci=0.08,
            margin=0.2,
            id=outcome.record_id,
        )
    )

    records = history.all_records()

    assert len(records) == 1
    assert records[0].record_id == outcome.record_id


def test_search_history_ignores_unparseable_pattern_payloads():
    pattern_memory = PatternMemory()
    history = SearchHistory(pattern_memory=pattern_memory)
    pattern_memory.store(Pattern(intent=history.intent, context={}, payload={"bad": "payload"}))

    assert history.all_records() == []


def test_search_history_rejects_invalid_record_type():
    with pytest.raises(TypeError):
        SearchHistory(raw_records=[object()])  # type: ignore[list-item]


def test_record_outcome_appends_to_history_and_returns_outcome():
    history = SearchHistory()
    problem = make_problem()
    strategy = make_strategy()

    outcome = record_outcome(history, problem, strategy, True, 9, 0.33)

    assert outcome in history.records
    assert history.all_records()[-1].final_score == pytest.approx(0.33)


def test_record_outcome_also_persists_into_pattern_memory():
    pattern_memory = PatternMemory()
    history = SearchHistory(pattern_memory=pattern_memory)

    outcome = record_outcome(history, make_problem(), make_strategy(), True, 3, 0.07)

    patterns = pattern_memory.all_patterns(history.intent)
    assert len(patterns) == 1
    assert patterns[0].id == outcome.record_id
    assert patterns[0].payload["record_id"] == outcome.record_id


def test_record_outcome_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        record_outcome("not-history", make_problem(), make_strategy(), True, 1, 0.1)  # type: ignore[arg-type]


def test_meta_controller_rejects_invalid_min_similarity():
    with pytest.raises(ValueError):
        MetaController(min_similarity=1.2)


def test_recommend_returns_sane_defaults_with_empty_history():
    controller = MetaController()
    recommendation = controller.recommend(make_problem(), SearchHistory())

    assert isinstance(recommendation, SearchStrategy)
    assert 0.0 <= recommendation.mutation_rate <= 1.0
    assert recommendation.population_size >= 8
    assert recommendation.branching_factor >= 1
    assert 0.0 <= recommendation.exploration_exploitation <= 1.0
    assert recommendation.compute_allocation > 0.0
    assert recommendation.convergence_threshold > 0.0


def test_recommend_prefers_gradient_search_for_smooth_problem():
    controller = MetaController()
    problem = make_problem(ruggedness=0.15, noise_level=0.05, gradient_available=True)

    recommendation = controller.recommend(problem, SearchHistory())

    assert recommendation.optimizer == "adaptive_gradient"
    assert recommendation.model_class == "AdaptiveGradientSearch"


def test_recommend_prefers_evolutionary_search_for_rugged_noisy_problem():
    controller = MetaController()
    problem = make_problem(ruggedness=0.9, noise_level=0.7, gradient_available=False)

    recommendation = controller.recommend(problem, SearchHistory())

    assert recommendation.optimizer == "evolutionary"
    assert recommendation.model_class == "EvolutionaryPopulation"


def test_recommend_uses_larger_population_for_harder_problem():
    controller = MetaController()
    easy = make_problem(dimensionality=3, bounds_width=1.0, ruggedness=0.2)
    hard = make_problem(dimensionality=40, bounds_width=20.0, ruggedness=0.8, gradient_available=False)

    easy_recommendation = controller.recommend(easy, SearchHistory())
    hard_recommendation = controller.recommend(hard, SearchHistory())

    assert hard_recommendation.population_size > easy_recommendation.population_size
    assert hard_recommendation.mutation_rate > easy_recommendation.mutation_rate


def test_recommend_with_empty_history_keeps_exploration_higher_than_rich_history():
    controller = MetaController()
    problem = make_problem()
    empty_recommendation = controller.recommend(problem, SearchHistory())
    rich_history = make_history_with_good_and_bad_records(problem)

    historical_recommendation = controller.recommend(problem, rich_history)

    assert empty_recommendation.exploration_exploitation > historical_recommendation.exploration_exploitation


def test_recommend_biases_mutation_rate_toward_successful_similar_history():
    controller = MetaController()
    problem = make_problem()
    default_mutation = controller.recommend(problem, SearchHistory()).mutation_rate
    history = make_history_with_good_and_bad_records(problem)

    recommendation = controller.recommend(problem, history)

    assert abs(recommendation.mutation_rate - 0.24) < abs(default_mutation - 0.24)
    assert recommendation.mutation_rate > default_mutation


def test_recommend_biases_population_toward_successful_similar_history():
    controller = MetaController()
    problem = make_problem()
    default_population = controller.recommend(problem, SearchHistory()).population_size
    history = make_history_with_good_and_bad_records(problem)

    recommendation = controller.recommend(problem, history)

    assert abs(recommendation.population_size - 92) < abs(default_population - 92)
    assert recommendation.population_size > default_population


def test_recommend_lowers_exploration_when_track_record_is_strong():
    controller = MetaController()
    problem = make_problem()
    history = SearchHistory()
    for _ in range(5):
        record_outcome(
            history,
            problem,
            make_strategy(exploration_exploitation=0.18, mutation_rate=0.22, population_size=80),
            True,
            6,
            0.05,
        )

    recommendation = controller.recommend(problem, history)

    assert recommendation.exploration_exploitation < 0.4


def test_recommend_keeps_exploration_higher_when_only_one_sparse_record_exists():
    controller = MetaController()
    problem = make_problem()
    sparse_history = SearchHistory()
    record_outcome(sparse_history, problem, make_strategy(exploration_exploitation=0.2), True, 6, 0.05)

    sparse_recommendation = controller.recommend(problem, sparse_history)
    dense_history = SearchHistory()
    for _ in range(4):
        record_outcome(dense_history, problem, make_strategy(exploration_exploitation=0.2), True, 6, 0.05)
    dense_recommendation = controller.recommend(problem, dense_history)

    assert sparse_recommendation.exploration_exploitation > dense_recommendation.exploration_exploitation


def test_dissimilar_history_does_not_override_default_recommendation():
    controller = MetaController()
    target_problem = make_problem(dimensionality=4, bounds_width=1.2, ruggedness=0.15, noise_level=0.05)
    default_recommendation = controller.recommend(target_problem, SearchHistory())

    very_different_problem = make_problem(
        dimensionality=96,
        bounds_width=80.0,
        ruggedness=0.95,
        noise_level=0.9,
        constraint_count=8,
        gradient_available=False,
    )
    history = SearchHistory()
    for _ in range(3):
        record_outcome(
            history,
            very_different_problem,
            make_strategy(
                optimizer="evolutionary",
                mutation_rate=0.9,
                population_size=180,
                branching_factor=18,
                exploration_exploitation=0.9,
                compute_allocation=0.95,
                convergence_threshold=0.02,
                model_class="EvolutionaryPopulation",
            ),
            True,
            12,
            0.01,
        )

    recommendation = controller.recommend(target_problem, history)

    assert abs(recommendation.mutation_rate - default_recommendation.mutation_rate) < 0.05
    assert abs(recommendation.population_size - default_recommendation.population_size) < 10


def test_recommend_can_select_history_preferred_optimizer():
    controller = MetaController()
    problem = make_problem(gradient_available=False, ruggedness=0.7, noise_level=0.5)
    history = SearchHistory()
    for _ in range(3):
        record_outcome(
            history,
            problem,
            make_strategy(
                optimizer="evolutionary",
                mutation_rate=0.26,
                population_size=88,
                branching_factor=10,
                exploration_exploitation=0.28,
                compute_allocation=0.85,
                convergence_threshold=0.004,
                model_class="EvolutionaryPopulation",
            ),
            True,
            8,
            0.04,
        )

    recommendation = controller.recommend(problem, history)

    assert recommendation.optimizer == "evolutionary"
    assert recommendation.model_class == "EvolutionaryPopulation"


def test_recommend_uses_adaptive_branch_count(monkeypatch: pytest.MonkeyPatch):
    controller = MetaController()
    problem = make_problem()
    captured: dict[str, object] = {}

    def fake_adaptive_branch_count(**kwargs: object) -> int:
        captured.update(kwargs)
        return 17

    monkeypatch.setattr("qes.meta_learning.adaptive_branch_count", fake_adaptive_branch_count)

    recommendation = controller.recommend(problem, SearchHistory())

    assert recommendation.branching_factor == 17
    assert captured["base"] == max(2, round(recommendation.population_size / 12))
    assert 0.0 <= float(captured["uncertainty"]) <= 1.0
    assert 0.0 <= float(captured["risk"]) <= 1.0
    assert 0.0 <= float(captured["novelty"]) <= 1.0


def test_recommend_rejects_invalid_inputs():
    controller = MetaController()
    with pytest.raises(TypeError):
        controller.recommend("not-a-problem", SearchHistory())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        controller.recommend(make_problem(), "not-history")  # type: ignore[arg-type]


def test_problem_structure_and_mapping_helpers_reject_invalid_serialized_values():
    with pytest.raises(KeyError):
        ProblemStructure.from_mapping({"bounds_width": 2.0})
    with pytest.raises(TypeError, match="gradient_available must be a boolean"):
        ProblemStructure.from_mapping(
            {
                "dimensionality": 3,
                "bounds_width": 1.0,
                "gradient_available": "yes",
            }
        )
    invalid_problem = make_problem().to_mapping()
    invalid_problem["constraint_count"] = True
    with pytest.raises(TypeError, match="constraint_count must be an integer"):
        ProblemStructure(**invalid_problem)
    with pytest.raises(TypeError, match="constraint_count must be an integer"):
        ProblemStructure.from_mapping(
            {
                "dimensionality": 3,
                "bounds_width": 1.0,
                "constraint_count": True,
            }
        )
    with pytest.raises(TypeError, match="constraint_count must be int-compatible"):
        ProblemStructure.from_mapping(
            {
                "dimensionality": 3,
                "bounds_width": 1.0,
                "constraint_count": [],
            }
        )
    with pytest.raises(KeyError):
        ProblemStructure.from_mapping({"dimensionality": 3})
    with pytest.raises(TypeError, match="bounds_width must be float-compatible, not a boolean"):
        ProblemStructure.from_mapping(
            {
                "dimensionality": 3,
                "bounds_width": True,
            }
        )
    with pytest.raises(TypeError, match="bounds_width must be float-compatible"):
        ProblemStructure.from_mapping(
            {
                "dimensionality": 3,
                "bounds_width": [],
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "error_type", "match"),
    [
        ("optimizer", 1, TypeError, "optimizer must be a string"),
        ("optimizer", "   ", ValueError, "optimizer must be non-empty"),
        ("model_class", 1, TypeError, "model_class must be a string"),
        ("model_class", "", ValueError, "model_class must be non-empty"),
    ],
)
def test_search_strategy_rejects_invalid_string_fields(
    field: str, value: object, error_type: type[Exception], match: str
):
    kwargs = make_strategy().to_mapping()
    kwargs[field] = value
    with pytest.raises(error_type, match=match):
        SearchStrategy(**kwargs)


def test_search_outcome_rejects_invalid_core_types_and_record_ids():
    with pytest.raises(TypeError, match="problem must be a ProblemStructure"):
        SearchOutcome(
            problem="bad",  # type: ignore[arg-type]
            strategy=make_strategy(),
            converged=True,
            generations_to_converge=1,
            final_score=0.1,
        )
    with pytest.raises(TypeError, match="strategy must be a SearchStrategy"):
        SearchOutcome(
            problem=make_problem(),
            strategy="bad",  # type: ignore[arg-type]
            converged=True,
            generations_to_converge=1,
            final_score=0.1,
        )
    with pytest.raises(TypeError, match="converged must be a boolean"):
        SearchOutcome(
            problem=make_problem(),
            strategy=make_strategy(),
            converged="yes",  # type: ignore[arg-type]
            generations_to_converge=1,
            final_score=0.1,
        )
    with pytest.raises(TypeError, match="record_id must be a string"):
        SearchOutcome(
            problem=make_problem(),
            strategy=make_strategy(),
            converged=True,
            generations_to_converge=1,
            final_score=0.1,
            record_id=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="record_id must be non-empty"):
        SearchOutcome(
            problem=make_problem(),
            strategy=make_strategy(),
            converged=True,
            generations_to_converge=1,
            final_score=0.1,
            record_id="",
        )


def test_search_outcome_from_mapping_validates_nested_mappings():
    payload = {
        "problem": [],
        "strategy": make_strategy().to_mapping(),
        "converged": True,
        "generations_to_converge": 1,
        "final_score": 0.1,
    }
    with pytest.raises(TypeError, match="problem must be a mapping"):
        SearchOutcome.from_mapping(payload)


def test_search_history_rejects_invalid_pattern_memory_and_intent():
    with pytest.raises(TypeError, match="pattern_memory must be a PatternMemory"):
        SearchHistory(pattern_memory="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="intent must be a string"):
        SearchHistory(intent=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="intent must be non-empty"):
        SearchHistory(intent="")


def test_meta_controller_private_history_bias_helpers_handle_zero_weight_cases():
    controller = MetaController()
    outcome = SearchOutcome(
        problem=make_problem(),
        strategy=make_strategy(),
        converged=False,
        generations_to_converge=5,
        final_score=0.5,
    )
    base = make_strategy()
    similar = [(outcome, 0.0, 0.0)]

    assert controller._historical_success(similar) == pytest.approx(0.5)
    assert controller._apply_history_bias(base, similar) == base
    assert controller._weighted_average("mutation_rate", 0.25, []) == pytest.approx(0.25)
    assert controller._weighted_average("mutation_rate", 0.25, [(outcome, 0.0)]) == pytest.approx(
        0.25
    )
