"""Phase 12 -- Persistent Knowledge Graph.

    Reality -> Observation -> Equation -> Pattern -> Result
        -> Causal relation -> Knowledge graph

`qes.patterns.PatternMemory` already stores reusable configurations keyed
by intent. This module builds the larger substrate the roadmap calls
for: a typed, persistent graph of knowledge nodes -- realities,
observations, equations, patterns, results, and causal-relation
hypotheses -- with provenance edges recording how one produced another,
plus explicit tracking of successful configurations, failed
configurations, and counterexamples, so a future QES run can query "what
happened last time" instead of starting from zero.

This is an ordinary in-memory graph (dict-of-dicts) with JSON
`save()`/`load()` for persistence across process runs. It is not a
database and not literal cognition -- it is a typed provenance/outcome
record a caller can query, and every operation here is plain Python
dict/list bookkeeping, no different in kind from `qes.events.LineageGraph`
(which this module complements: `LineageGraph` tracks raw event
provenance, `KnowledgeGraph` adds typed nodes, success/failure labels,
and causal-hypothesis confidence on top).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NODE_KINDS = {"reality", "observation", "equation", "pattern", "result", "causal_relation"}

_node_id_counter = itertools.count(1)


def _next_node_id(kind: str) -> str:
    return f"{kind.upper()}-{next(_node_id_counter):06d}"


def _validate_confidence(value: float) -> float:
    conf = float(value)
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {conf!r}")
    return conf


@dataclass
class KnowledgeNode:
    """One typed node in the knowledge graph.

    Attributes:
        id: unique node identifier.
        kind: one of `NODE_KINDS`.
        payload: arbitrary JSON-serializable metadata for this node.
        success: `True`/`False` if this node represents a labeled
            successful/failed configuration or result, `None` if not
            applicable (e.g. a plain observation).
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    success: bool | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"kind must be one of {sorted(NODE_KINDS)}, got {self.kind!r}")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dict")
        self.payload = dict(self.payload)
        if self.success is not None and not isinstance(self.success, bool):
            raise TypeError("success must be a bool or None")
        if not self.id:
            self.id = _next_node_id(self.kind)
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeNode:
        return cls(
            kind=data["kind"],
            payload=dict(data.get("payload", {})),
            success=data.get("success"),
            id=data["id"],
        )


@dataclass
class CausalHypothesis:
    """A hypothesized `cause -> effect` relationship between two nodes,
    strengthened or weakened as more evidence accumulates."""

    cause_id: str
    effect_id: str
    confidence: float = 0.5
    evidence_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.cause_id, str) or not self.cause_id:
            raise ValueError("cause_id must be a non-empty string")
        if not isinstance(self.effect_id, str) or not self.effect_id:
            raise ValueError("effect_id must be a non-empty string")
        self.confidence = _validate_confidence(self.confidence)
        if not isinstance(self.evidence_count, int) or isinstance(self.evidence_count, bool):
            raise TypeError("evidence_count must be an integer")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause_id": self.cause_id,
            "effect_id": self.effect_id,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalHypothesis:
        return cls(
            cause_id=data["cause_id"],
            effect_id=data["effect_id"],
            confidence=data["confidence"],
            evidence_count=data["evidence_count"],
        )


class KnowledgeGraph:
    """Persistent, typed provenance/outcome graph (Phase 12).

    Nodes are typed (`reality`/`observation`/`equation`/`pattern`/
    `result`/`causal_relation`); edges record parent -> child provenance
    (e.g. an equation produced a result, a pattern was derived from a
    result); `record_causal_hypothesis`/`strengthen_hypothesis` track
    cause/effect confidence separately from plain provenance; and
    `add_counterexample` tags nodes that disprove a named concept.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KnowledgeNode] = {}
        self._parents: dict[str, list[str]] = {}
        self._children: dict[str, list[str]] = {}
        self._hypotheses: list[CausalHypothesis] = []
        self._counterexamples: dict[str, list[str]] = {}
        self._domain_relationships: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Nodes and provenance
    # ------------------------------------------------------------------
    def add_node(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        parents: list[str] | tuple[str, ...] = (),
        success: bool | None = None,
        node_id: str | None = None,
    ) -> KnowledgeNode:
        """Register a new typed node, linked to any existing `parents`."""
        for parent_id in parents:
            if parent_id not in self._nodes:
                raise KeyError(f"unknown parent node: {parent_id!r}")
        node = KnowledgeNode(kind=kind, payload=dict(payload or {}), success=success, id=node_id or "")
        if node.id in self._nodes:
            raise ValueError(f"duplicate node id: {node.id!r}")
        self._nodes[node.id] = node
        self._parents[node.id] = list(parents)
        self._children.setdefault(node.id, [])
        for parent_id in parents:
            self._children.setdefault(parent_id, []).append(node.id)
        return node

    def get(self, node_id: str) -> KnowledgeNode:
        if node_id not in self._nodes:
            raise KeyError(f"unknown node: {node_id!r}")
        return self._nodes[node_id]

    def nodes(self, kind: str | None = None) -> list[KnowledgeNode]:
        """All nodes, optionally filtered by `kind`."""
        if kind is not None and kind not in NODE_KINDS:
            raise ValueError(f"kind must be one of {sorted(NODE_KINDS)}, got {kind!r}")
        values = list(self._nodes.values())
        return [n for n in values if n.kind == kind] if kind is not None else values

    def ancestors(self, node_id: str) -> list[str]:
        """All ancestors of `node_id`, nearest first, deduplicated."""
        if node_id not in self._nodes:
            raise KeyError(f"unknown node: {node_id!r}")
        seen: list[str] = []
        frontier = list(self._parents.get(node_id, []))
        while frontier:
            parent_id = frontier.pop(0)
            if parent_id in seen:
                continue
            seen.append(parent_id)
            frontier.extend(self._parents.get(parent_id, []))
        return seen

    def descendants(self, node_id: str) -> list[str]:
        """All descendants of `node_id`, nearest first, deduplicated."""
        if node_id not in self._nodes:
            raise KeyError(f"unknown node: {node_id!r}")
        seen: list[str] = []
        frontier = list(self._children.get(node_id, []))
        while frontier:
            child_id = frontier.pop(0)
            if child_id in seen:
                continue
            seen.append(child_id)
            frontier.extend(self._children.get(child_id, []))
        return seen

    # ------------------------------------------------------------------
    # Experiment recording (Reality -> Observation -> Equation -> Pattern -> Result)
    # ------------------------------------------------------------------
    def record_experiment(
        self,
        equation_id: str,
        result_payload: dict[str, Any],
        success: bool,
        pattern_id: str | None = None,
    ) -> KnowledgeNode:
        """Record one experiment's `result`, linked to the `equation` (and
        optionally the `pattern`) that produced it -- the historical
        experiment trail the roadmap calls for."""
        parents = [equation_id] if pattern_id is None else [equation_id, pattern_id]
        return self.add_node(kind="result", payload=result_payload, parents=parents, success=success)

    def successful_configurations(self, kind: str = "pattern") -> list[KnowledgeNode]:
        """All `kind` nodes explicitly labeled successful (`success=True`)."""
        return [n for n in self.nodes(kind) if n.success is True]

    def failed_configurations(self, kind: str = "pattern") -> list[KnowledgeNode]:
        """All `kind` nodes explicitly labeled failed (`success=False`)."""
        return [n for n in self.nodes(kind) if n.success is False]

    # ------------------------------------------------------------------
    # Causal hypotheses
    # ------------------------------------------------------------------
    def record_causal_hypothesis(
        self, cause_id: str, effect_id: str, confidence: float = 0.5, evidence_count: int = 1
    ) -> CausalHypothesis:
        """Register a new `cause -> effect` hypothesis between two known nodes."""
        if cause_id not in self._nodes:
            raise KeyError(f"unknown cause node: {cause_id!r}")
        if effect_id not in self._nodes:
            raise KeyError(f"unknown effect node: {effect_id!r}")
        hypothesis = CausalHypothesis(
            cause_id=cause_id, effect_id=effect_id, confidence=confidence, evidence_count=evidence_count
        )
        self._hypotheses.append(hypothesis)
        return hypothesis

    def strengthen_hypothesis(
        self, cause_id: str, effect_id: str, delta_confidence: float, extra_evidence: int = 1
    ) -> CausalHypothesis:
        """Update an existing hypothesis' confidence (clamped to [0, 1]) and
        evidence count as new supporting/contradicting evidence arrives."""
        for hypothesis in self._hypotheses:
            if hypothesis.cause_id == cause_id and hypothesis.effect_id == effect_id:
                hypothesis.confidence = _validate_confidence(
                    max(0.0, min(1.0, hypothesis.confidence + delta_confidence))
                )
                hypothesis.evidence_count += extra_evidence
                return hypothesis
        raise KeyError(f"no hypothesis for {cause_id!r} -> {effect_id!r}")

    def causal_hypotheses_for(self, node_id: str) -> list[CausalHypothesis]:
        """All hypotheses where `node_id` is either the cause or the effect."""
        return [h for h in self._hypotheses if h.cause_id == node_id or h.effect_id == node_id]

    # ------------------------------------------------------------------
    # Counterexamples and domain relationships
    # ------------------------------------------------------------------
    def add_counterexample(self, concept: str, node_id: str) -> None:
        """Tag `node_id` as a counterexample disproving/limiting `concept`."""
        if not isinstance(concept, str) or not concept:
            raise ValueError("concept must be a non-empty string")
        if node_id not in self._nodes:
            raise KeyError(f"unknown node: {node_id!r}")
        self._counterexamples.setdefault(concept, []).append(node_id)

    def counterexamples(self, concept: str) -> list[KnowledgeNode]:
        """All nodes tagged as counterexamples for `concept`."""
        return [self._nodes[node_id] for node_id in self._counterexamples.get(concept, [])]

    def add_domain_relationship(self, domain_a: str, domain_b: str, relation: str) -> None:
        """Record a named relationship (e.g. "shares_bounds", "conflicts_with")
        between two domains -- plain metadata, not tied to specific nodes."""
        if not domain_a or not domain_b or not relation:
            raise ValueError("domain_a, domain_b, and relation must all be non-empty strings")
        self._domain_relationships.append(
            {"domain_a": domain_a, "domain_b": domain_b, "relation": relation}
        )

    def domain_relationships(self, domain: str | None = None) -> list[dict[str, Any]]:
        """All recorded domain relationships, optionally filtered to those
        involving `domain`."""
        if domain is None:
            return list(self._domain_relationships)
        return [
            r for r in self._domain_relationships if domain in (r["domain_a"], r["domain_b"])
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) representation of the whole graph."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "parents": {k: list(v) for k, v in self._parents.items()},
            "hypotheses": [h.to_dict() for h in self._hypotheses],
            "counterexamples": {k: list(v) for k, v in self._counterexamples.items()},
            "domain_relationships": [dict(r) for r in self._domain_relationships],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        """Reconstruct a `KnowledgeGraph` previously serialized via `to_dict()`."""
        graph = cls()
        nodes_data = data.get("nodes", {})
        parents_data = data.get("parents", {})

        remaining = dict(parents_data)
        while remaining:
            ready = [
                node_id
                for node_id, node_parents in remaining.items()
                if all(p in graph._nodes for p in node_parents)
            ]
            if not ready:
                raise ValueError("knowledge graph data contains a cycle or unresolved parent reference")
            for node_id in ready:
                node_data = nodes_data[node_id]
                graph.add_node(
                    kind=node_data["kind"],
                    payload=dict(node_data.get("payload", {})),
                    parents=remaining[node_id],
                    success=node_data.get("success"),
                    node_id=node_id,
                )
                del remaining[node_id]

        graph._hypotheses = [CausalHypothesis.from_dict(h) for h in data.get("hypotheses", [])]
        graph._counterexamples = {
            k: list(v) for k, v in data.get("counterexamples", {}).items()
        }
        graph._domain_relationships = [dict(r) for r in data.get("domain_relationships", [])]
        return graph

    def save(self, path: str | Path) -> None:
        """Persist this graph to a JSON file at `path`."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        """Load a `KnowledgeGraph` previously persisted via `save()`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self._nodes)
