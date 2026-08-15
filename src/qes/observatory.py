"""Phase 20 -- QES Observatory.

Plain-text monitoring helpers for the roadmap's "observatory" view. The
dashboard is intentionally dependency-free: it renders aligned metrics, an
ASCII/Unicode reality tree, and optional summaries of related QES modules using
only the standard library (plus whatever upstream objects the caller passes in).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from math import isfinite
from types import ModuleType
from typing import Any

import numpy as np

_adversarial: ModuleType | None = None
try:
    import qes.adversarial as _adversarial
except ImportError:  # pragma: no cover - depends on workspace state
    pass

_digital_twin_loop: ModuleType | None = None
try:
    import qes.digital_twin_loop as _digital_twin_loop
except ImportError:  # pragma: no cover - depends on workspace state
    pass

_divergence: ModuleType | None = None
try:
    import qes.divergence as _divergence
except ImportError:  # pragma: no cover - depends on workspace state
    pass

_knowledge_graph: ModuleType | None = None
try:
    import qes.knowledge_graph as _knowledge_graph
except ImportError:  # pragma: no cover - depends on workspace state
    pass

_marketplace: ModuleType | None = None
try:
    import qes.marketplace as _marketplace
except ImportError:  # pragma: no cover - depends on workspace state
    pass

_multi_agent_evolution: ModuleType | None = None
try:
    import qes.multi_agent_evolution as _multi_agent_evolution
except ImportError:  # pragma: no cover - depends on workspace state
    pass

_permission: ModuleType | None = None
try:
    import qes.permission as _permission
except ImportError:  # pragma: no cover - depends on workspace state
    pass

_OPTIONAL_MODULES = (
    _adversarial,
    _digital_twin_loop,
    _divergence,
    _knowledge_graph,
    _marketplace,
    _multi_agent_evolution,
    _permission,
)

_BANNER = "H¹¹ QES OBSERVATORY"


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_finite_float(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return numeric


def _validate_reality_tree(tree: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    if not isinstance(tree, Mapping):
        raise TypeError("reality_tree must be a mapping of parent -> children")
    if not tree:
        raise ValueError("reality_tree must not be empty")

    normalized: dict[str, list[str]] = {}
    for parent, children in tree.items():
        if not isinstance(parent, str) or not parent:
            raise ValueError("reality_tree keys must be non-empty strings")
        if isinstance(children, (str, bytes)) or not isinstance(children, Sequence):
            raise TypeError(
                f"reality_tree[{parent!r}] must be a sequence of child node names"
            )
        normalized_children: list[str] = []
        for child in children:
            if not isinstance(child, str) or not child:
                raise ValueError(
                    f"reality_tree[{parent!r}] must contain only non-empty string child names"
                )
            normalized_children.append(child)
        normalized[parent] = normalized_children
    return normalized


def _infer_root(tree: Mapping[str, Sequence[str]]) -> str:
    children = {child for values in tree.values() for child in values}
    roots = [node for node in tree if node not in children]
    if roots:
        return sorted(roots)[0]
    return sorted(tree)[0]


def _to_plain_data(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain_data(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


def _extract_knowledge_graph_sections(knowledge_graph: object) -> tuple[object | None, object | None]:
    equations: object | None = None
    patterns: object | None = None

    nodes = getattr(knowledge_graph, "nodes", None)
    if callable(nodes):
        try:
            equation_nodes = nodes("equation")
        except Exception:
            equation_nodes = None
        try:
            pattern_nodes = nodes("pattern")
        except Exception:
            pattern_nodes = None
        if equation_nodes:
            equations = [
                {"id": getattr(node, "id", None), "payload": _to_plain_data(getattr(node, "payload", {}))}
                for node in equation_nodes
            ]
        if pattern_nodes:
            patterns = [
                {
                    "id": getattr(node, "id", None),
                    "payload": _to_plain_data(getattr(node, "payload", {})),
                    "success": getattr(node, "success", None),
                }
                for node in pattern_nodes
            ]

    hypotheses_for = getattr(knowledge_graph, "causal_hypotheses_for", None)
    all_nodes = getattr(knowledge_graph, "nodes", None)
    if callable(hypotheses_for) and callable(all_nodes):
        try:
            node_ids = [getattr(node, "id", None) for node in all_nodes()]
            hypotheses = []
            seen: set[tuple[str, str, float, int]] = set()
            for node_id in node_ids:
                if not isinstance(node_id, str):
                    continue
                for hypothesis in hypotheses_for(node_id):
                    row = (
                        str(getattr(hypothesis, "cause_id", "")),
                        str(getattr(hypothesis, "effect_id", "")),
                        float(getattr(hypothesis, "confidence", 0.0)),
                        int(getattr(hypothesis, "evidence_count", 0)),
                    )
                    if row in seen:
                        continue
                    seen.add(row)
                    hypotheses.append(
                        {
                            "cause_id": row[0],
                            "effect_id": row[1],
                            "confidence": row[2],
                            "evidence_count": row[3],
                        }
                    )
            if hypotheses:
                if patterns is None:
                    patterns = {}
                if isinstance(patterns, list):
                    patterns = {"patterns": patterns, "causal_hypotheses": hypotheses}
                elif isinstance(patterns, dict):
                    patterns = {**patterns, "causal_hypotheses": hypotheses}
        except Exception:
            pass

    return equations, patterns


def _summarize_marketplace(marketplace: object) -> object:
    if _marketplace is not None and isinstance(marketplace, _marketplace.RealityMarketplace):
        return {
            "round_index": marketplace.round_index,
            "accounts": sorted(marketplace.accounts),
            "last_allocations": marketplace.get_last_allocations(),
        }
    if (
        hasattr(marketplace, "get_last_allocations")
        and hasattr(marketplace, "round_index")
    ):
        try:
            return {
                "round_index": marketplace.round_index,
                "last_allocations": _to_plain_data(marketplace.get_last_allocations()),
            }
        except Exception:
            return _to_plain_data(marketplace)
    return _to_plain_data(marketplace)


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _format_block(value: object, *, indent: int = 2) -> list[str]:
    plain = _to_plain_data(value)
    pad = " " * indent
    if plain is None:
        return []
    if isinstance(plain, Mapping):
        lines: list[str] = []
        for key, item in plain.items():
            if isinstance(item, (Mapping, list, tuple)):
                lines.append(f"{pad}{key}:")
                lines.extend(_format_block(item, indent=indent + 2))
            else:
                lines.append(f"{pad}{key}: {_format_scalar(item)}")
        return lines
    if isinstance(plain, list):
        lines = []
        for item in plain:
            if isinstance(item, (Mapping, list, tuple)):
                lines.append(f"{pad}-")
                lines.extend(_format_block(item, indent=indent + 2))
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
        return lines
    return [f"{pad}{_format_scalar(plain)}"]


@dataclass(slots=True)
class ObservatorySnapshot:
    """One renderable observatory state.

    Required metrics are the roadmap's core headline values. Optional fields can
    contain plain dict/list data or upstream QES dataclass objects; rendering
    converts them into plain summaries on demand.
    """

    population: int
    active_realities: int
    compute_pct: float
    convergence: float
    risk: float
    novelty: float
    reality_tree: dict[str, list[str]]
    equations: object | None = None
    divergence: object | None = None
    permission: object | None = None
    resource_allocation: object | None = None
    agent_interactions: object | None = None
    digital_twin_error: object | None = None
    failures: object | None = None
    discovered_patterns: object | None = None

    def __post_init__(self) -> None:
        self.population = _validate_non_negative_int("population", self.population)
        self.active_realities = _validate_non_negative_int(
            "active_realities",
            self.active_realities,
        )
        self.compute_pct = _validate_finite_float(
            "compute_pct",
            self.compute_pct,
            minimum=0.0,
            maximum=100.0,
        )
        self.convergence = _validate_finite_float("convergence", self.convergence)
        self.risk = _validate_finite_float("risk", self.risk)
        self.novelty = _validate_finite_float("novelty", self.novelty)
        self.reality_tree = _validate_reality_tree(self.reality_tree)


def render_reality_tree(tree: dict[str, list[str]], root: str) -> str:
    """Render a parent->children adjacency mapping as a box-drawing tree.

    Args:
        tree: parent -> children adjacency mapping.
        root: root node name to render from.

    Raises:
        ValueError: if the tree is empty, malformed, or contains a cycle.
    """

    normalized = _validate_reality_tree(tree)
    if root not in normalized:
        raise ValueError(f"root {root!r} is not present in reality_tree")

    lines = [root]

    def visit(node: str, prefix: str, ancestors: set[str]) -> None:
        if node in ancestors:
            raise ValueError(f"cycle detected while rendering reality_tree at {node!r}")
        children = normalized.get(node, [])
        updated_ancestors = ancestors | {node}
        for index, child in enumerate(children):
            is_last = index == len(children) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{child}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            if child in normalized:
                visit(child, child_prefix, updated_ancestors)

    visit(root, "", set())
    return "\n".join(lines)


class ObservatoryDashboard:
    """Stateful plain-text dashboard renderer for `ObservatorySnapshot` objects."""

    def __init__(self, snapshot: ObservatorySnapshot | None = None) -> None:
        self._snapshot = snapshot

    def update(self, snapshot: ObservatorySnapshot) -> None:
        """Replace the currently rendered snapshot."""
        if not isinstance(snapshot, ObservatorySnapshot):
            raise TypeError("snapshot must be an ObservatorySnapshot")
        self._snapshot = snapshot

    def render(self) -> str:
        """Render the current dashboard as dependency-free plain text."""
        if self._snapshot is None:
            raise ValueError("no ObservatorySnapshot has been loaded")

        snapshot = self._snapshot
        labels = [
            ("Population", f"{snapshot.population:,}"),
            ("Active Realities", f"{snapshot.active_realities:,}"),
            ("Compute", f"{snapshot.compute_pct:.1f}%"),
            ("Convergence", f"{snapshot.convergence:.2f}"),
            ("Risk", f"{snapshot.risk:.2f}"),
            ("Novelty", f"{snapshot.novelty:.2f}"),
        ]
        label_width = max(len(label) for label, _ in labels)
        lines = [_BANNER, "=" * len(_BANNER)]
        lines.extend(f"{label:<{label_width}}  {value}" for label, value in labels)

        lines.extend(["", "Reality Tree", "------------"])
        lines.append(render_reality_tree(snapshot.reality_tree, _infer_root(snapshot.reality_tree)))

        sections = [
            ("Equations", snapshot.equations),
            ("Divergence", snapshot.divergence),
            ("Permission", snapshot.permission),
            ("Resource Allocation", snapshot.resource_allocation),
            ("Agent Interactions", snapshot.agent_interactions),
            ("Digital Twin Error", snapshot.digital_twin_error),
            ("Failures", snapshot.failures),
            ("Discovered Patterns", snapshot.discovered_patterns),
        ]
        for title, value in sections:
            if value is None:
                continue
            block = _format_block(value)
            if not block:
                continue
            lines.extend(["", f"{title}:", *block])

        return "\n".join(lines)

    @staticmethod
    def render_live(
        snapshots: Iterable[ObservatorySnapshot],
        refresh_fn: Any = print,
    ) -> None:
        """Render a sequence of snapshots one after another via `refresh_fn`."""
        if not callable(refresh_fn):
            raise TypeError("refresh_fn must be callable")
        dashboard = ObservatoryDashboard()
        for snapshot in snapshots:
            dashboard.update(snapshot)
            refresh_fn(dashboard.render())


def build_snapshot_from_modules(
    *,
    population: int = 0,
    active_realities: int = 0,
    compute_pct: float = 0.0,
    convergence: float = 0.0,
    risk: float = 0.0,
    novelty: float = 0.0,
    reality_tree: Mapping[str, Sequence[str]] | None = None,
    equations: object | None = None,
    divergence: object | None = None,
    permission: object | None = None,
    resource_allocation: object | None = None,
    agent_interactions: object | None = None,
    digital_twin_error: object | None = None,
    failures: object | None = None,
    discovered_patterns: object | None = None,
    knowledge_graph: object | None = None,
) -> ObservatorySnapshot:
    """Assemble a snapshot from whichever upstream module outputs are available.

    All non-core inputs are optional. When omitted, the snapshot still renders
    using a minimal placeholder tree and the provided core metrics.
    """

    resolved_tree = (
        _validate_reality_tree(reality_tree) if reality_tree is not None else {"ROOT": []}
    )

    resolved_equations = _to_plain_data(equations)
    resolved_patterns = _to_plain_data(discovered_patterns)
    if knowledge_graph is not None:
        extracted_equations, extracted_patterns = _extract_knowledge_graph_sections(knowledge_graph)
        if resolved_equations is None:
            resolved_equations = _to_plain_data(extracted_equations)
        if resolved_patterns is None:
            resolved_patterns = _to_plain_data(extracted_patterns)

    resolved_divergence = _to_plain_data(divergence)
    resolved_permission = _to_plain_data(permission)
    resolved_agent_interactions = _to_plain_data(agent_interactions)
    resolved_digital_twin_error = _to_plain_data(digital_twin_error)
    resolved_failures = _to_plain_data(failures)
    resolved_resource_allocation = (
        _summarize_marketplace(resource_allocation)
        if resource_allocation is not None
        else None
    )

    return ObservatorySnapshot(
        population=population,
        active_realities=active_realities,
        compute_pct=compute_pct,
        convergence=convergence,
        risk=risk,
        novelty=novelty,
        reality_tree=resolved_tree,
        equations=resolved_equations,
        divergence=resolved_divergence,
        permission=resolved_permission,
        resource_allocation=resolved_resource_allocation,
        agent_interactions=resolved_agent_interactions,
        digital_twin_error=resolved_digital_twin_error,
        failures=resolved_failures,
        discovered_patterns=resolved_patterns,
    )
