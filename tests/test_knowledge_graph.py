"""Tests for `qes.knowledge_graph` (Phase 12 -- Persistent Knowledge Graph)."""
from __future__ import annotations

import pytest

from qes.knowledge_graph import (
    NODE_KINDS,
    CausalHypothesis,
    KnowledgeGraph,
    KnowledgeNode,
)


class TestKnowledgeNode:
    def test_auto_id_and_kind_validation(self) -> None:
        node = KnowledgeNode(kind="reality", payload={"x": 1})
        assert node.kind == "reality"
        assert node.id.startswith("REALITY-")
        assert node.payload == {"x": 1}
        assert node.success is None

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            KnowledgeNode(kind="not-a-kind")

    def test_payload_must_be_dict(self) -> None:
        with pytest.raises(TypeError):
            KnowledgeNode(kind="reality", payload="nope")  # type: ignore[arg-type]

    def test_success_must_be_bool_or_none(self) -> None:
        with pytest.raises(TypeError):
            KnowledgeNode(kind="reality", success="yes")  # type: ignore[arg-type]

    def test_to_dict_from_dict_roundtrip(self) -> None:
        node = KnowledgeNode(kind="result", payload={"score": 0.9}, success=True)
        restored = KnowledgeNode.from_dict(node.to_dict())
        assert restored.id == node.id
        assert restored.kind == node.kind
        assert restored.payload == node.payload
        assert restored.success is True

    def test_all_node_kinds_valid(self) -> None:
        for kind in NODE_KINDS:
            node = KnowledgeNode(kind=kind)
            assert node.kind == kind

    def test_id_must_be_a_nonempty_string(self) -> None:
        with pytest.raises(ValueError):
            KnowledgeNode(kind="reality", id=123)  # type: ignore[arg-type]


class TestCausalHypothesis:
    def test_defaults_and_validation(self) -> None:
        h = CausalHypothesis(cause_id="A", effect_id="B")
        assert h.confidence == 0.5
        assert h.evidence_count == 1

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            CausalHypothesis(cause_id="A", effect_id="B", confidence=1.5)

    def test_empty_ids_rejected(self) -> None:
        with pytest.raises(ValueError):
            CausalHypothesis(cause_id="", effect_id="B")
        with pytest.raises(ValueError):
            CausalHypothesis(cause_id="A", effect_id="")

    def test_negative_evidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="evidence_count"):
            CausalHypothesis(cause_id="A", effect_id="B", evidence_count=-1)

    def test_evidence_count_must_be_an_integer(self) -> None:
        with pytest.raises(TypeError, match="evidence_count"):
            CausalHypothesis(cause_id="A", effect_id="B", evidence_count=1.5)  # type: ignore[arg-type]

    def test_roundtrip(self) -> None:
        h = CausalHypothesis(cause_id="A", effect_id="B", confidence=0.7, evidence_count=3)
        restored = CausalHypothesis.from_dict(h.to_dict())
        assert restored == h


class TestKnowledgeGraphNodesAndProvenance:
    def test_add_and_get_node(self) -> None:
        graph = KnowledgeGraph()
        node = graph.add_node(kind="reality", payload={"seed": 1})
        assert graph.get(node.id) is node
        assert len(graph) == 1

    def test_get_unknown_node_raises(self) -> None:
        graph = KnowledgeGraph()
        with pytest.raises(KeyError):
            graph.get("missing")

    def test_add_node_with_unknown_parent_raises(self) -> None:
        graph = KnowledgeGraph()
        with pytest.raises(KeyError, match="unknown parent"):
            graph.add_node(kind="observation", parents=["missing"])

    def test_duplicate_node_id_raises(self) -> None:
        graph = KnowledgeGraph()
        node = graph.add_node(kind="reality", node_id="FIXED-1")
        with pytest.raises(ValueError, match="duplicate node id"):
            graph.add_node(kind="reality", node_id=node.id)

    def test_nodes_filtered_by_kind(self) -> None:
        graph = KnowledgeGraph()
        graph.add_node(kind="reality")
        graph.add_node(kind="observation")
        graph.add_node(kind="reality")
        assert len(graph.nodes()) == 3
        assert len(graph.nodes(kind="reality")) == 2
        with pytest.raises(ValueError):
            graph.nodes(kind="bogus")

    def test_ancestors_and_descendants(self) -> None:
        graph = KnowledgeGraph()
        r = graph.add_node(kind="reality")
        obs = graph.add_node(kind="observation", parents=[r.id])
        eq = graph.add_node(kind="equation", parents=[obs.id])
        result = graph.add_node(kind="result", parents=[eq.id])

        assert graph.ancestors(result.id) == [eq.id, obs.id, r.id]
        assert graph.descendants(r.id) == [obs.id, eq.id, result.id]
        assert graph.ancestors(r.id) == []
        assert graph.descendants(result.id) == []

    def test_ancestors_unknown_node_raises(self) -> None:
        graph = KnowledgeGraph()
        with pytest.raises(KeyError):
            graph.ancestors("missing")
        with pytest.raises(KeyError):
            graph.descendants("missing")

    def test_ancestors_and_descendants_deduplicate_shared_paths(self) -> None:
        graph = KnowledgeGraph()
        root = graph.add_node(kind="reality")
        left = graph.add_node(kind="observation", parents=[root.id])
        right = graph.add_node(kind="equation", parents=[root.id])
        leaf = graph.add_node(kind="result", parents=[left.id, right.id])

        assert graph.ancestors(leaf.id) == [left.id, right.id, root.id]
        assert graph.descendants(root.id) == [left.id, right.id, leaf.id]


class TestExperimentRecording:
    def test_record_experiment_links_equation_and_pattern(self) -> None:
        graph = KnowledgeGraph()
        equation = graph.add_node(kind="equation")
        pattern = graph.add_node(kind="pattern")
        result = graph.record_experiment(
            equation_id=equation.id,
            result_payload={"fitness": 0.8},
            success=True,
            pattern_id=pattern.id,
        )
        assert result.kind == "result"
        assert result.success is True
        assert set(graph._parents[result.id]) == {equation.id, pattern.id}

    def test_successful_and_failed_configurations(self) -> None:
        graph = KnowledgeGraph()
        graph.add_node(kind="pattern", success=True)
        graph.add_node(kind="pattern", success=False)
        graph.add_node(kind="pattern", success=None)
        assert len(graph.successful_configurations("pattern")) == 1
        assert len(graph.failed_configurations("pattern")) == 1


class TestCausalHypothesesOnGraph:
    def test_record_and_strengthen_hypothesis(self) -> None:
        graph = KnowledgeGraph()
        a = graph.add_node(kind="equation")
        b = graph.add_node(kind="result")
        graph.record_causal_hypothesis(a.id, b.id, confidence=0.5)
        updated = graph.strengthen_hypothesis(a.id, b.id, delta_confidence=0.3)
        assert updated.confidence == pytest.approx(0.8)
        assert updated.evidence_count == 2

    def test_strengthen_clamps_to_bounds(self) -> None:
        graph = KnowledgeGraph()
        a = graph.add_node(kind="equation")
        b = graph.add_node(kind="result")
        graph.record_causal_hypothesis(a.id, b.id, confidence=0.9)
        updated = graph.strengthen_hypothesis(a.id, b.id, delta_confidence=0.5)
        assert updated.confidence == 1.0

    def test_hypothesis_for_unknown_nodes_raises(self) -> None:
        graph = KnowledgeGraph()
        a = graph.add_node(kind="equation")
        with pytest.raises(KeyError):
            graph.record_causal_hypothesis("missing", a.id)
        with pytest.raises(KeyError):
            graph.record_causal_hypothesis(a.id, "missing")

    def test_strengthen_missing_hypothesis_raises(self) -> None:
        graph = KnowledgeGraph()
        a = graph.add_node(kind="equation")
        b = graph.add_node(kind="result")
        with pytest.raises(KeyError, match="no hypothesis"):
            graph.strengthen_hypothesis(a.id, b.id, delta_confidence=0.1)

    def test_causal_hypotheses_for_node(self) -> None:
        graph = KnowledgeGraph()
        a = graph.add_node(kind="equation")
        b = graph.add_node(kind="result")
        c = graph.add_node(kind="result")
        graph.record_causal_hypothesis(a.id, b.id)
        graph.record_causal_hypothesis(a.id, c.id)
        assert len(graph.causal_hypotheses_for(a.id)) == 2
        assert len(graph.causal_hypotheses_for(b.id)) == 1

    def test_strengthen_hypothesis_skips_non_matching_entries_before_updating_match(self) -> None:
        graph = KnowledgeGraph()
        a = graph.add_node(kind="equation")
        b = graph.add_node(kind="result")
        c = graph.add_node(kind="equation")
        d = graph.add_node(kind="result")
        graph.record_causal_hypothesis(a.id, b.id, confidence=0.2)
        graph.record_causal_hypothesis(c.id, d.id, confidence=0.4)

        updated = graph.strengthen_hypothesis(c.id, d.id, delta_confidence=0.1)

        assert updated.confidence == pytest.approx(0.5)


class TestCounterexamplesAndDomainRelationships:
    def test_add_and_query_counterexamples(self) -> None:
        graph = KnowledgeGraph()
        node = graph.add_node(kind="result")
        graph.add_counterexample("linearity", node.id)
        results = graph.counterexamples("linearity")
        assert results == [node]
        assert graph.counterexamples("unknown-concept") == []

    def test_counterexample_unknown_node_raises(self) -> None:
        graph = KnowledgeGraph()
        with pytest.raises(KeyError):
            graph.add_counterexample("concept", "missing")

    def test_counterexample_empty_concept_raises(self) -> None:
        graph = KnowledgeGraph()
        node = graph.add_node(kind="result")
        with pytest.raises(ValueError):
            graph.add_counterexample("", node.id)

    def test_domain_relationships(self) -> None:
        graph = KnowledgeGraph()
        graph.add_domain_relationship("finance", "engineering", "shares_bounds")
        graph.add_domain_relationship("finance", "physics", "conflicts_with")
        assert len(graph.domain_relationships()) == 2
        finance_rels = graph.domain_relationships("finance")
        assert len(finance_rels) == 2
        engineering_rels = graph.domain_relationships("engineering")
        assert len(engineering_rels) == 1

    def test_domain_relationship_requires_nonempty_fields(self) -> None:
        graph = KnowledgeGraph()
        with pytest.raises(ValueError):
            graph.add_domain_relationship("", "engineering", "shares_bounds")


class TestPersistence:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        graph = KnowledgeGraph()
        r = graph.add_node(kind="reality", payload={"seed": 7})
        obs = graph.add_node(kind="observation", parents=[r.id])
        result = graph.record_experiment(obs.id, {"score": 1.0}, success=True)
        graph.record_causal_hypothesis(obs.id, result.id, confidence=0.6)
        graph.add_counterexample("concept", result.id)
        graph.add_domain_relationship("a", "b", "relates_to")

        restored = KnowledgeGraph.from_dict(graph.to_dict())
        assert len(restored) == len(graph)
        assert restored.ancestors(result.id) == graph.ancestors(result.id)
        assert len(restored.causal_hypotheses_for(obs.id)) == 1
        assert restored.counterexamples("concept")[0].id == result.id
        assert restored.domain_relationships() == graph.domain_relationships()

    def test_save_and_load(self, tmp_path) -> None:
        graph = KnowledgeGraph()
        node = graph.add_node(kind="reality", payload={"seed": 42})
        path = tmp_path / "graph.json"
        graph.save(path)

        loaded = KnowledgeGraph.load(path)
        assert len(loaded) == 1
        assert loaded.get(node.id).payload == {"seed": 42}

    def test_from_dict_rejects_cyclic_parent_data(self) -> None:
        data = {
            "nodes": {
                "A": {"id": "A", "kind": "reality", "payload": {}, "success": None},
                "B": {"id": "B", "kind": "reality", "payload": {}, "success": None},
            },
            "parents": {"A": ["B"], "B": ["A"]},
        }
        with pytest.raises(ValueError, match="cycle"):
            KnowledgeGraph.from_dict(data)


class TestLen:
    def test_len_reflects_node_count(self) -> None:
        graph = KnowledgeGraph()
        assert len(graph) == 0
        graph.add_node(kind="reality")
        graph.add_node(kind="reality")
        assert len(graph) == 2
