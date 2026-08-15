"""Tests for `qes.causal_engine` (Phase 5 -- Causal Reality Engine)."""
from __future__ import annotations

import numpy as np
import pytest

from qes.causal_engine import (
    CausalEdge,
    CausalGraph,
    HiddenVariableHypothesis,
    analyze_causality,
    counterfactual,
    hidden_variable_hypotheses,
    intervene,
    perturbation_sensitivity,
)
from qes.events import EventLog
from qes.room import Room


def make_graph() -> CausalGraph:
    graph = CausalGraph()
    for name in ("effort", "quality", "adoption", "cost"):
        graph.add_variable(name)
    graph.add_edge("effort", "quality", strength=2.0)
    graph.add_edge("quality", "adoption", strength=0.5)
    graph.add_edge("effort", "cost", strength=-1.0)
    return graph


def make_room(x: list[float] | np.ndarray, *, use_mapping: bool = False) -> Room:
    x_array = np.asarray(x, dtype=float)
    room = Room(
        x=x_array,
        x_star=np.zeros_like(x_array),
        lower=-np.ones_like(x_array) * 10.0,
        upper=np.ones_like(x_array) * 10.0,
        activation=np.ones_like(x_array),
    )
    if use_mapping:
        room.memory["causal_variables"] = {
            "effort": 0,
            "quality": 1,
            "adoption": 2,
            "cost": 3,
        }
    return room


class TestCausalEdge:
    def test_roundtrip(self) -> None:
        edge = CausalEdge("effort", "quality", strength=1.25)
        restored = CausalEdge.from_dict(edge.to_dict())
        assert restored == edge

    def test_rejects_self_edge(self) -> None:
        with pytest.raises(ValueError, match="different variables"):
            CausalEdge("effort", "effort")


class TestCausalGraph:
    def test_add_variable_rejects_duplicate(self) -> None:
        graph = CausalGraph()
        graph.add_variable("effort")
        with pytest.raises(ValueError, match="duplicate variable"):
            graph.add_variable("effort")

    def test_add_edge_requires_known_variables(self) -> None:
        graph = CausalGraph()
        graph.add_variable("effort")
        with pytest.raises(KeyError, match="unknown effect"):
            graph.add_edge("effort", "quality")

    def test_add_edge_rejects_duplicate_edge(self) -> None:
        graph = make_graph()
        with pytest.raises(ValueError, match="duplicate edge"):
            graph.add_edge("effort", "quality", strength=3.0)

    def test_cycle_rejection(self) -> None:
        graph = make_graph()
        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("adoption", "effort")

    def test_parents_children_ancestors_descendants(self) -> None:
        graph = make_graph()
        assert graph.parents("quality") == ["effort"]
        assert set(graph.children("effort")) == {"quality", "cost"}
        assert graph.ancestors("adoption") == ["quality", "effort"]
        assert graph.descendants("effort") == ["quality", "cost", "adoption"]

    def test_topological_order_respects_dependencies(self) -> None:
        graph = make_graph()
        order = graph.topological_order()
        assert order.index("effort") < order.index("quality") < order.index("adoption")
        assert order.index("effort") < order.index("cost")

    def test_edge_strength_lookup(self) -> None:
        graph = make_graph()
        assert graph.edge_strength("effort", "quality") == pytest.approx(2.0)
        with pytest.raises(KeyError, match="unknown edge"):
            graph.edge_strength("quality", "effort")

    def test_to_dict_from_dict_roundtrip(self) -> None:
        graph = make_graph()
        restored = CausalGraph.from_dict(graph.to_dict())
        assert restored.variables() == graph.variables()
        assert restored.parents("quality") == ["effort"]
        assert restored.edge_strength("effort", "cost") == pytest.approx(-1.0)

    def test_from_dict_rejects_non_list_sections(self) -> None:
        with pytest.raises(TypeError, match="variables must be a list"):
            CausalGraph.from_dict({"variables": "effort", "edges": []})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="edges must be a list"):
            CausalGraph.from_dict({"variables": [], "edges": {}})  # type: ignore[arg-type]

    def test_unknown_variable_queries_raise(self) -> None:
        graph = make_graph()
        with pytest.raises(KeyError):
            graph.parents("missing")
        with pytest.raises(KeyError):
            graph.children("missing")
        with pytest.raises(KeyError):
            graph.ancestors("missing")
        with pytest.raises(KeyError):
            graph.descendants("missing")


class TestInterventionsAndCounterfactuals:
    def test_intervene_propagates_deltas_through_dag_for_mapping(self) -> None:
        graph = make_graph()
        state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
        result = intervene(graph, "effort", 2.0, state, lambda s: s["adoption"])  # type: ignore[index]
        assert result.baseline_value == pytest.approx(1.0)
        assert result.intervened_state["quality"] == pytest.approx(5.0)
        assert result.intervened_state["adoption"] == pytest.approx(5.0)
        assert result.intervened_state["cost"] == pytest.approx(1.0)
        assert result.outcome_delta == pytest.approx(1.0)

    def test_intervene_on_room_uses_default_variable_order(self) -> None:
        graph = make_graph()
        room = make_room([1.0, 3.0, 4.0, 2.0])
        result = intervene(graph, "effort", 2.0, room, lambda candidate: float(candidate.x[2]))
        assert result.intervened_state["adoption"] == pytest.approx(5.0)
        assert result.intervened_outcome == pytest.approx(5.0)

    def test_intervene_on_room_uses_explicit_variable_mapping(self) -> None:
        graph = make_graph()
        room = make_room([1.0, 3.0, 4.0, 2.0], use_mapping=True)
        result = intervene(graph, "effort", 0.0, room, lambda candidate: float(candidate.x[3]))
        assert result.intervened_state["cost"] == pytest.approx(3.0)
        assert result.intervened_outcome == pytest.approx(3.0)

    def test_intervene_rejects_missing_state_values(self) -> None:
        graph = make_graph()
        with pytest.raises(KeyError, match="missing variable"):
            intervene(
                graph,
                "effort",
                2.0,
                {"effort": 1.0, "quality": 3.0, "adoption": 4.0},
                lambda s: s["adoption"],  # type: ignore[index]
            )

    def test_intervene_rejects_room_without_mapping_when_dimensions_do_not_match(self) -> None:
        graph = make_graph()
        room = make_room([1.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="graph variable count must equal room.dim"):
            intervene(graph, "effort", 2.0, room, lambda candidate: float(candidate.x[0]))

    def test_counterfactual_compares_actual_and_counterfactual_worlds(self) -> None:
        graph = make_graph()
        state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
        result = counterfactual(graph, "quality", 1.0, state, lambda s: s["adoption"])  # type: ignore[index]
        assert result.actual_value == pytest.approx(3.0)
        assert result.counterfactual_state["adoption"] == pytest.approx(3.0)
        assert result.outcome_delta == pytest.approx(-1.0)

    def test_counterfactual_rejects_actual_value_mismatch(self) -> None:
        graph = make_graph()
        state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
        with pytest.raises(ValueError, match="does not match observed state value"):
            counterfactual(
                graph,
                "quality",
                1.0,
                state,
                lambda s: s["adoption"],  # type: ignore[index]
                actual_value=99.0,
            )

    def test_evaluate_fn_must_return_finite_value(self) -> None:
        graph = make_graph()
        state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
        with pytest.raises(ValueError, match="evaluate_fn result must be finite"):
            intervene(graph, "effort", 2.0, state, lambda s: float("nan"))


class TestPerturbationSensitivity:
    def test_mapping_sensitivity_ranks_variables(self) -> None:
        graph = make_graph()
        state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
        results = perturbation_sensitivity(
            graph,
            state,
            lambda s: 10.0 * s["effort"] + 0.5 * s["quality"] + 0.1 * s["adoption"],  # type: ignore[index]
            scale=0.1,
            samples_per_variable=64,
            rng=np.random.default_rng(0),
        )
        assert [result.variable for result in results[:2]] == ["effort", "quality"]
        assert results[0].mean_absolute_outcome_delta > results[1].mean_absolute_outcome_delta

    def test_room_sensitivity_uses_reality_operator_perturbation(self) -> None:
        graph = make_graph()
        room = make_room([1.0, 3.0, 4.0, 2.0], use_mapping=True)
        results = perturbation_sensitivity(
            graph,
            room,
            lambda candidate: float(5.0 * candidate.x[0] + 0.1 * candidate.x[2]),
            scale=0.1,
            samples_per_variable=48,
            rng=np.random.default_rng(1),
        )
        assert results[0].variable == "effort"
        assert results[0].mean_absolute_state_delta > 0.0

    def test_sensitivity_rejects_negative_scale(self) -> None:
        graph = make_graph()
        state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
        with pytest.raises(ValueError, match="scale must be >= 0"):
            perturbation_sensitivity(graph, state, lambda s: s["adoption"], scale=-0.1)  # type: ignore[index]

    def test_sensitivity_rejects_non_positive_sample_count(self) -> None:
        graph = make_graph()
        state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
        with pytest.raises(ValueError, match="samples_per_variable must be > 0"):
            perturbation_sensitivity(
                graph,
                state,
                lambda s: s["adoption"],  # type: ignore[index]
                samples_per_variable=0,
            )


class TestHiddenVariableHypotheses:
    def test_roundtrip(self) -> None:
        hypothesis = HiddenVariableHypothesis(
            variable_a="demand",
            variable_b="supply",
            observed_correlation=-0.92,
            candidate_name="latent_market_pressure",
            reason="High inverse correlation with no known path in the DAG.",
        )
        restored = HiddenVariableHypothesis.from_dict(hypothesis.to_dict())
        assert restored == hypothesis

    def test_detects_unexplained_correlations(self) -> None:
        graph = CausalGraph()
        for name in ("x", "y", "z"):
            graph.add_variable(name)
        latent = np.linspace(-1.0, 1.0, 10)
        samples = [
            {"x": float(value), "y": float(1.5 * value + 0.01), "z": float((-1.0) ** i)}
            for i, value in enumerate(latent)
        ]
        hypotheses = hidden_variable_hypotheses(graph, samples, correlation_threshold=0.9)
        assert len(hypotheses) == 1
        assert {hypotheses[0].variable_a, hypotheses[0].variable_b} == {"x", "y"}

    def test_ignores_pairs_explained_by_graph_structure(self) -> None:
        graph = CausalGraph()
        for name in ("shared", "x", "y"):
            graph.add_variable(name)
        graph.add_edge("shared", "x", strength=1.0)
        graph.add_edge("shared", "y", strength=1.0)
        values = np.linspace(-1.0, 1.0, 10)
        samples = [
            {"shared": float(v), "x": float(v), "y": float(v)}
            for v in values
        ]
        assert hidden_variable_hypotheses(graph, samples, correlation_threshold=0.9) == []

    def test_rejects_threshold_above_one(self) -> None:
        graph = make_graph()
        with pytest.raises(ValueError, match="correlation_threshold must be in \\[0, 1\\]"):
            hidden_variable_hypotheses(graph, [], correlation_threshold=1.5)

    def test_returns_empty_list_for_insufficient_samples(self) -> None:
        graph = make_graph()
        samples = [{"effort": 1.0, "quality": 2.0, "adoption": 3.0, "cost": 4.0}]
        assert hidden_variable_hypotheses(graph, samples) == []


class TestBundledAnalysis:
    def test_analyze_causality_bundles_outputs_and_logs_events(self) -> None:
        graph = make_graph()
        state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
        samples = [
            {"effort": 0.0, "quality": 0.0, "adoption": 0.0, "cost": 3.0},
            {"effort": 1.0, "quality": 2.0, "adoption": 3.0, "cost": 2.0},
            {"effort": 2.0, "quality": 4.0, "adoption": 5.0, "cost": 1.0},
        ]
        event_log = EventLog()
        result = analyze_causality(
            graph,
            state,
            lambda s: s["adoption"] - 0.1 * s["cost"],  # type: ignore[index]
            interventions=[("effort", 2.0)],
            counterfactual_queries=[("quality", 1.0)],
            sensitivity_scale=0.1,
            sensitivity_samples=8,
            correlation_samples=samples,
            correlation_threshold=0.95,
            rng=np.random.default_rng(2),
            event_log=event_log,
        )
        assert len(result.interventions) == 1
        assert len(result.counterfactuals) == 1
        assert len(result.sensitivities) == 4
        assert len(event_log.filter("EXECUTE")) == 1
        assert len(event_log.filter("DIVERGE")) == 1
        assert len(event_log.filter("MUTATE")) == 1
        assert len(event_log.filter("MERGE")) == 1


def test_internal_validators_and_hidden_variable_hypothesis_validation() -> None:
    import qes.causal_engine as causal_engine_module

    with pytest.raises(ValueError, match="non-empty string"):
        causal_engine_module._validate_name("", "name")
    with pytest.raises(TypeError, match="samples must be an integer"):
        causal_engine_module._validate_positive_int(True, "samples")
    with pytest.raises(ValueError, match="must be different"):
        HiddenVariableHypothesis(
            variable_a="same",
            variable_b="same",
            observed_correlation=0.5,
            candidate_name="latent",
            reason="bad",
        )


def test_graph_validation_and_deduplication_paths() -> None:
    graph = CausalGraph()
    graph.add_variable("a")
    graph.add_variable("b")
    with pytest.raises(KeyError, match="unknown cause"):
        graph.add_edge("missing", "a")
    with pytest.raises(ValueError, match="different variables"):
        graph.add_edge("a", "a")

    graph = CausalGraph()
    for name in ("a", "b", "c", "d"):
        graph.add_variable(name)
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    graph.add_edge("b", "d")
    graph.add_edge("c", "d")
    assert graph.ancestors("d") == ["b", "c", "a"]
    assert graph.descendants("a") == ["b", "c", "d"]
    assert graph._has_path("a", "missing") is False
    assert graph.topological_order() == ["a", "b", "c", "d"]


def test_topological_order_detects_cycle_in_corrupted_graph() -> None:
    graph = CausalGraph()
    graph.add_variable("a")
    graph.add_variable("b")
    graph._parents["a"] = ["b"]
    graph._parents["b"] = ["a"]
    graph._children["a"] = ["b"]
    graph._children["b"] = ["a"]
    with pytest.raises(ValueError, match="contains a cycle"):
        graph.topological_order()


def test_room_variable_mapping_validation_and_state_extraction_errors() -> None:
    import qes.causal_engine as causal_engine_module

    graph = make_graph()
    room = make_room([1.0, 2.0, 3.0, 4.0], use_mapping=True)
    room.memory["causal_variables"] = {"effort": 0, "quality": 1, "adoption": 2}
    with pytest.raises(KeyError, match="missing 'cost'"):
        causal_engine_module._room_variable_indices(graph, room)

    room.memory["causal_variables"] = {"effort": 0, "quality": 1, "adoption": 2, "cost": "3"}
    with pytest.raises(TypeError, match="must be integers"):
        causal_engine_module._room_variable_indices(graph, room)

    room.memory["causal_variables"] = {"effort": 0, "quality": 1, "adoption": 2, "cost": 4}
    with pytest.raises(ValueError, match="out of bounds"):
        causal_engine_module._room_variable_indices(graph, room)

    room.memory["causal_variables"] = ["effort", "quality"]
    with pytest.raises(ValueError, match="length must match"):
        causal_engine_module._room_variable_indices(graph, room)

    room.memory["causal_variables"] = ["effort", "quality", "adoption", "other"]
    with pytest.raises(KeyError, match="missing 'cost'"):
        causal_engine_module._room_variable_indices(graph, room)

    room.memory["causal_variables"] = 123
    with pytest.raises(TypeError, match="mapping or sequence"):
        causal_engine_module._room_variable_indices(graph, room)

    with pytest.raises(TypeError, match="Room or mapping"):
        causal_engine_module._extract_state(graph, 123)  # type: ignore[arg-type]


def test_propagation_and_query_functions_reject_unknown_variables() -> None:
    import qes.causal_engine as causal_engine_module

    graph = make_graph()
    baseline = {"effort": 1.0, "quality": 2.0, "adoption": 3.0, "cost": 4.0}
    with pytest.raises(KeyError, match="unknown variable"):
        causal_engine_module._propagate_state(graph, baseline, {"missing": 1.0})
    with pytest.raises(KeyError, match="unknown variable"):
        intervene(graph, "missing", 1.0, baseline, lambda s: s["adoption"])  # type: ignore[index]
    with pytest.raises(KeyError, match="unknown variable"):
        counterfactual(graph, "missing", 1.0, baseline, lambda s: s["adoption"])  # type: ignore[index]


def test_counterfactual_and_analysis_without_event_log_cover_default_paths() -> None:
    graph = make_graph()
    state = {"effort": 1.0, "quality": 3.0, "adoption": 4.0, "cost": 2.0}
    result = analyze_causality(
        graph,
        state,
        lambda s: s["adoption"],  # type: ignore[index]
        interventions=[("effort", 1.5)],
        counterfactual_queries=[("quality", 2.5)],
        correlation_samples=[],
    )
    assert len(result.interventions) == 1
    assert len(result.counterfactuals) == 1


def test_hidden_variable_hypotheses_event_logging_and_sequence_mapping() -> None:
    graph = make_graph()
    room = make_room([0.0, 0.0, 0.0, 0.0])
    room.memory["causal_variables"] = ["effort", "quality", "adoption", "cost"]
    extracted = intervene(graph, "effort", 1.0, room, lambda candidate: float(candidate.x[0]))
    assert extracted.intervened_value == pytest.approx(1.0)

    simple = CausalGraph()
    for name in ("x", "y", "z"):
        simple.add_variable(name)
    samples = [
        {"x": -1.0, "y": -1.0, "z": 0.0},
        {"x": 0.0, "y": 0.0, "z": 1.0},
        {"x": 1.0, "y": 1.0, "z": 0.0},
    ]
    event_log = EventLog()
    hypotheses = hidden_variable_hypotheses(simple, samples, correlation_threshold=0.8, event_log=event_log)
    assert hypotheses
    assert len(event_log.filter("SELECT")) == 1
