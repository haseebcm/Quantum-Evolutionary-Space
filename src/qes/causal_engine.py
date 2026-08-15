"""Phase 5 -- Causal Reality Engine.

This module adds a classical causal-analysis layer on top of QES rooms and
state dictionaries. The central object is a directed acyclic graph (DAG) over
named variables/factors. On top of that graph, the engine can run:

* interventions (`do(X=x)`)
* counterfactual comparisons against an observed state
* perturbation-sensitivity scans
* simple hidden-variable heuristics

The propagation model used here is intentionally modest and explicit: edge
strengths define a linearized delta-flow through the DAG, so changing a parent
shifts each descendant by the weighted sum of its parents' deviations from the
baseline state. This is ordinary in-process Python/numpy bookkeeping; it is
not literal philosophical causality discovery or quantum computation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qes.events import EventLog
from qes.reality_operators import perturbation
from qes.room import Room


def _validate_name(name: str, label: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} must be a non-empty string")
    return name


def _validate_finite_float(value: float, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_positive_int(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value


@dataclass(frozen=True)
class CausalEdge:
    """One directed `cause -> effect` link in a `CausalGraph`.

    Attributes:
        cause: upstream variable/factor name.
        effect: downstream variable/factor name.
        strength: linearized propagation weight used by interventions and
            counterfactuals. A positive value means the effect tends to move in
            the same direction as the cause relative to the baseline state;
            negative means the opposite.
    """

    cause: str
    effect: str
    strength: float = 1.0

    def __post_init__(self) -> None:
        _validate_name(self.cause, "cause")
        _validate_name(self.effect, "effect")
        if self.cause == self.effect:
            raise ValueError("cause and effect must be different variables")
        _validate_finite_float(self.strength, "strength")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) representation of this edge."""
        return {"cause": self.cause, "effect": self.effect, "strength": self.strength}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CausalEdge:
        """Reconstruct a `CausalEdge` previously serialized via `to_dict()`."""
        return cls(cause=data["cause"], effect=data["effect"], strength=data.get("strength", 1.0))


@dataclass
class InterventionResult:
    """Result of one `do(X=x)` intervention over a baseline state."""

    variable: str
    baseline_value: float
    intervened_value: float
    baseline_outcome: float
    intervened_outcome: float
    baseline_state: dict[str, float]
    intervened_state: dict[str, float]
    state_delta: dict[str, float]
    outcome_delta: float


@dataclass
class CounterfactualResult:
    """Actual-world vs counterfactual-world comparison for one variable."""

    variable: str
    actual_value: float
    counterfactual_value: float
    actual_outcome: float
    counterfactual_outcome: float
    actual_state: dict[str, float]
    counterfactual_state: dict[str, float]
    state_delta: dict[str, float]
    outcome_delta: float


@dataclass(order=True)
class SensitivityResult:
    """Sensitivity summary for one variable under repeated small perturbations."""

    sort_index: float = field(init=False, repr=False)
    variable: str
    mean_absolute_outcome_delta: float
    max_absolute_outcome_delta: float
    mean_absolute_state_delta: float
    samples: int

    def __post_init__(self) -> None:
        self.sort_index = -self.mean_absolute_outcome_delta


@dataclass
class HiddenVariableHypothesis:
    """A heuristic latent-variable explanation for an otherwise unexplained correlation."""

    variable_a: str
    variable_b: str
    observed_correlation: float
    candidate_name: str
    reason: str

    def __post_init__(self) -> None:
        _validate_name(self.variable_a, "variable_a")
        _validate_name(self.variable_b, "variable_b")
        if self.variable_a == self.variable_b:
            raise ValueError("variable_a and variable_b must be different")
        self.observed_correlation = _validate_finite_float(
            self.observed_correlation, "observed_correlation"
        )
        _validate_name(self.candidate_name, "candidate_name")
        _validate_name(self.reason, "reason")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) representation of this hypothesis."""
        return {
            "variable_a": self.variable_a,
            "variable_b": self.variable_b,
            "observed_correlation": self.observed_correlation,
            "candidate_name": self.candidate_name,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HiddenVariableHypothesis:
        """Reconstruct a `HiddenVariableHypothesis` from plain dict data."""
        return cls(
            variable_a=data["variable_a"],
            variable_b=data["variable_b"],
            observed_correlation=data["observed_correlation"],
            candidate_name=data["candidate_name"],
            reason=data["reason"],
        )


@dataclass
class CausalDiscoveryResult:
    """Bundle of outputs from a full causal-analysis pass."""

    graph: CausalGraph
    interventions: list[InterventionResult]
    counterfactuals: list[CounterfactualResult]
    sensitivities: list[SensitivityResult]
    hidden_variable_hypotheses: list[HiddenVariableHypothesis]


class CausalGraph:
    """Directed acyclic graph (DAG) over named causal variables/factors.

    The graph stores only variable names and weighted edges. It does not try to
    infer a full symbolic structural equation model; instead, interventions and
    counterfactuals use the edge strengths as a linearized propagation heuristic
    from a caller-provided baseline state.
    """

    def __init__(self) -> None:
        self._variables: list[str] = []
        self._parents: dict[str, list[str]] = {}
        self._children: dict[str, list[str]] = {}
        self._strengths: dict[tuple[str, str], float] = {}

    def add_variable(self, name: str) -> None:
        """Register one named variable/factor in the graph."""
        variable = _validate_name(name, "name")
        if variable in self._parents:
            raise ValueError(f"duplicate variable: {variable!r}")
        self._variables.append(variable)
        self._parents[variable] = []
        self._children[variable] = []

    def add_edge(self, cause: str, effect: str, strength: float = 1.0) -> CausalEdge:
        """Register a weighted `cause -> effect` edge, rejecting cycles."""
        source = _validate_name(cause, "cause")
        target = _validate_name(effect, "effect")
        if source not in self._parents:
            raise KeyError(f"unknown cause variable: {source!r}")
        if target not in self._parents:
            raise KeyError(f"unknown effect variable: {target!r}")
        if source == target:
            raise ValueError("cause and effect must be different variables")
        if (source, target) in self._strengths:
            raise ValueError(f"duplicate edge: {source!r} -> {target!r}")
        weight = _validate_finite_float(strength, "strength")
        if self._has_path(target, source):
            raise ValueError(f"adding {source!r} -> {target!r} would create a cycle")
        self._parents[target].append(source)
        self._children[source].append(target)
        self._strengths[(source, target)] = weight
        return CausalEdge(cause=source, effect=target, strength=weight)

    def variables(self) -> list[str]:
        """All variable names in insertion order."""
        return list(self._variables)

    def parents(self, name: str) -> list[str]:
        """Immediate parents of `name`."""
        variable = _validate_name(name, "name")
        if variable not in self._parents:
            raise KeyError(f"unknown variable: {variable!r}")
        return list(self._parents[variable])

    def children(self, name: str) -> list[str]:
        """Immediate children of `name`."""
        variable = _validate_name(name, "name")
        if variable not in self._children:
            raise KeyError(f"unknown variable: {variable!r}")
        return list(self._children[variable])

    def edge_strength(self, cause: str, effect: str) -> float:
        """Weight associated with `cause -> effect`."""
        source = _validate_name(cause, "cause")
        target = _validate_name(effect, "effect")
        if (source, target) not in self._strengths:
            raise KeyError(f"unknown edge: {source!r} -> {target!r}")
        return self._strengths[(source, target)]

    def ancestors(self, name: str) -> list[str]:
        """All ancestors of `name`, nearest first, deduplicated."""
        variable = _validate_name(name, "name")
        if variable not in self._parents:
            raise KeyError(f"unknown variable: {variable!r}")
        seen: list[str] = []
        frontier = list(self._parents[variable])
        while frontier:
            parent = frontier.pop(0)
            if parent in seen:
                continue
            seen.append(parent)
            frontier.extend(self._parents[parent])
        return seen

    def descendants(self, name: str) -> list[str]:
        """All descendants of `name`, nearest first, deduplicated."""
        variable = _validate_name(name, "name")
        if variable not in self._children:
            raise KeyError(f"unknown variable: {variable!r}")
        seen: list[str] = []
        frontier = list(self._children[variable])
        while frontier:
            child = frontier.pop(0)
            if child in seen:
                continue
            seen.append(child)
            frontier.extend(self._children[child])
        return seen

    def topological_order(self) -> list[str]:
        """A topological ordering of the DAG's variables."""
        indegree = {name: len(parents) for name, parents in self._parents.items()}
        ready = [name for name in self._variables if indegree[name] == 0]
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for child in self._children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(order) != len(self._variables):
            raise ValueError("causal graph contains a cycle")
        return order

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) representation of the graph."""
        return {
            "variables": list(self._variables),
            "edges": [
                {"cause": cause, "effect": effect, "strength": strength}
                for (cause, effect), strength in self._strengths.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CausalGraph:
        """Reconstruct a `CausalGraph` previously serialized via `to_dict()`."""
        graph = cls()
        variables = data.get("variables", [])
        edges = data.get("edges", [])
        if not isinstance(variables, list):
            raise TypeError("variables must be a list")
        if not isinstance(edges, list):
            raise TypeError("edges must be a list")
        for name in variables:
            graph.add_variable(name)
        for edge in edges:
            graph.add_edge(edge["cause"], edge["effect"], edge.get("strength", 1.0))
        return graph

    def _has_path(self, source: str, target: str) -> bool:
        frontier = list(self._children[source])
        seen: set[str] = set()
        while frontier:
            node = frontier.pop(0)
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(self._children[node])
        return False

    def __contains__(self, item: object) -> bool:
        return isinstance(item, str) and item in self._parents

    def __len__(self) -> int:
        return len(self._variables)


def _room_variable_indices(graph: CausalGraph, room: Room) -> dict[str, int]:
    mapping = room.memory.get("causal_variables")
    if mapping is None:
        if len(graph) != room.dim:
            raise ValueError(
                "room has no causal_variables mapping, so graph variable count must equal room.dim"
            )
        return {name: index for index, name in enumerate(graph.variables())}
    if isinstance(mapping, Mapping):
        indices: dict[str, int] = {}
        for name in graph.variables():
            if name not in mapping:
                raise KeyError(f"room causal_variables mapping is missing {name!r}")
            index = mapping[name]
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("room causal_variables mapping values must be integers")
            if not 0 <= index < room.dim:
                raise ValueError(f"room causal_variables index out of bounds for {name!r}")
            indices[name] = index
        return indices
    if isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
        names = list(mapping)
        if len(names) != room.dim:
            raise ValueError("room causal_variables sequence length must match room.dim")
        lookup = {str(name): index for index, name in enumerate(names)}
        indices = {}
        for name in graph.variables():
            if name not in lookup:
                raise KeyError(f"room causal_variables sequence is missing {name!r}")
            indices[name] = lookup[name]
        return indices
    raise TypeError("room.memory['causal_variables'] must be a mapping or sequence of names")


def _state_from_mapping(graph: CausalGraph, state: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in graph.variables():
        if name not in state:
            raise KeyError(f"state is missing variable {name!r}")
        values[name] = _validate_finite_float(state[name], f"state[{name!r}]")
    return values


def _extract_state(graph: CausalGraph, room_or_state: Room | Mapping[str, Any]) -> dict[str, float]:
    if isinstance(room_or_state, Room):
        indices = _room_variable_indices(graph, room_or_state)
        return {name: float(room_or_state.x[index]) for name, index in indices.items()}
    if isinstance(room_or_state, Mapping):
        return _state_from_mapping(graph, room_or_state)
    raise TypeError("room_or_state must be a Room or mapping of variable -> value")


def _room_from_state(graph: CausalGraph, room: Room, state: Mapping[str, float]) -> Room:
    indices = _room_variable_indices(graph, room)
    x = room.x.copy()
    for name, value in state.items():
        x[indices[name]] = value
    return room.clone(x=x)


def _evaluate_outcome(
    graph: CausalGraph,
    baseline: Room | Mapping[str, Any],
    state: Mapping[str, float],
    evaluate_fn: Callable[[Any], float],
) -> float:
    subject: Any
    if isinstance(baseline, Room):
        subject = _room_from_state(graph, baseline, state)
    else:
        subject = dict(state)
    return _validate_finite_float(evaluate_fn(subject), "evaluate_fn result")


def _propagate_state(
    graph: CausalGraph,
    baseline_state: Mapping[str, float],
    interventions: Mapping[str, float],
) -> dict[str, float]:
    state = dict(baseline_state)
    blocked = set()
    for variable, value in interventions.items():
        if variable not in graph:
            raise KeyError(f"unknown variable: {variable!r}")
        state[variable] = _validate_finite_float(value, f"intervention value for {variable!r}")
        blocked.add(variable)
    for node in graph.topological_order():
        if node in blocked:
            continue
        parents = graph.parents(node)
        if not parents:
            continue
        propagated_delta = sum(
            graph.edge_strength(parent, node) * (state[parent] - baseline_state[parent])
            for parent in parents
        )
        state[node] = baseline_state[node] + propagated_delta
    return state


def intervene(
    graph: CausalGraph,
    variable: str,
    value: float,
    room_or_state: Room | Mapping[str, Any],
    evaluate_fn: Callable[[Any], float],
    event_log: EventLog | None = None,
) -> InterventionResult:
    """Apply a `do(variable=value)` intervention and measure the outcome delta."""
    name = _validate_name(variable, "variable")
    if name not in graph:
        raise KeyError(f"unknown variable: {name!r}")
    baseline_state = _extract_state(graph, room_or_state)
    baseline_outcome = _evaluate_outcome(graph, room_or_state, baseline_state, evaluate_fn)
    intervened_state = _propagate_state(graph, baseline_state, {name: value})
    intervened_outcome = _evaluate_outcome(graph, room_or_state, intervened_state, evaluate_fn)
    result = InterventionResult(
        variable=name,
        baseline_value=baseline_state[name],
        intervened_value=float(intervened_state[name]),
        baseline_outcome=baseline_outcome,
        intervened_outcome=intervened_outcome,
        baseline_state=baseline_state,
        intervened_state=intervened_state,
        state_delta={key: intervened_state[key] - baseline_state[key] for key in graph.variables()},
        outcome_delta=intervened_outcome - baseline_outcome,
    )
    if event_log is not None:
        event_log.record(
            "EXECUTE",
            payload={
                "analysis": "intervention",
                "variable": result.variable,
                "baseline_value": result.baseline_value,
                "intervened_value": result.intervened_value,
                "outcome_delta": result.outcome_delta,
            },
        )
    return result


def counterfactual(
    graph: CausalGraph,
    variable: str,
    counterfactual_value: float,
    room_or_state: Room | Mapping[str, Any],
    evaluate_fn: Callable[[Any], float],
    actual_value: float | None = None,
    event_log: EventLog | None = None,
) -> CounterfactualResult:
    """Compare the actual observed world against `X = counterfactual_value`."""
    name = _validate_name(variable, "variable")
    if name not in graph:
        raise KeyError(f"unknown variable: {name!r}")
    actual_state = _extract_state(graph, room_or_state)
    observed_value = actual_state[name]
    if actual_value is not None and not np.isclose(observed_value, float(actual_value)):
        raise ValueError(
            f"actual_value {actual_value!r} does not match observed state value {observed_value!r}"
        )
    actual_outcome = _evaluate_outcome(graph, room_or_state, actual_state, evaluate_fn)
    counterfactual_state = _propagate_state(graph, actual_state, {name: counterfactual_value})
    counterfactual_outcome = _evaluate_outcome(graph, room_or_state, counterfactual_state, evaluate_fn)
    result = CounterfactualResult(
        variable=name,
        actual_value=observed_value,
        counterfactual_value=float(counterfactual_state[name]),
        actual_outcome=actual_outcome,
        counterfactual_outcome=counterfactual_outcome,
        actual_state=actual_state,
        counterfactual_state=counterfactual_state,
        state_delta={key: counterfactual_state[key] - actual_state[key] for key in graph.variables()},
        outcome_delta=counterfactual_outcome - actual_outcome,
    )
    if event_log is not None:
        event_log.record(
            "DIVERGE",
            payload={
                "analysis": "counterfactual",
                "variable": result.variable,
                "actual_value": result.actual_value,
                "counterfactual_value": result.counterfactual_value,
                "outcome_delta": result.outcome_delta,
            },
        )
    return result


def perturbation_sensitivity(
    graph: CausalGraph,
    room_or_state: Room | Mapping[str, Any],
    evaluate_fn: Callable[[Any], float],
    scale: float = 0.05,
    samples_per_variable: int = 16,
    rng: np.random.Generator | None = None,
    event_log: EventLog | None = None,
) -> list[SensitivityResult]:
    """Rank variables by how much small perturbations change the measured outcome.

    For `Room` inputs this reuses `qes.reality_operators.perturbation()` along
    one explicit axis at a time. For plain mappings the same idea is applied by
    adding small Gaussian deltas directly to the selected variable.
    """

    sigma = _validate_finite_float(scale, "scale")
    if sigma < 0:
        raise ValueError("scale must be >= 0")
    count = _validate_positive_int(samples_per_variable, "samples_per_variable")
    generator = rng or np.random.default_rng(0)
    baseline_state = _extract_state(graph, room_or_state)
    baseline_outcome = _evaluate_outcome(graph, room_or_state, baseline_state, evaluate_fn)
    results: list[SensitivityResult] = []

    if isinstance(room_or_state, Room):
        indices = _room_variable_indices(graph, room_or_state)
        for name in graph.variables():
            outcome_deltas: list[float] = []
            state_deltas: list[float] = []
            for _ in range(count):
                directions = np.zeros((1, room_or_state.dim), dtype=float)
                directions[0, indices[name]] = 1.0
                candidate_room = perturbation(
                    room_or_state,
                    generator,
                    scale=sigma,
                    directions=directions,
                )
                candidate_state = _extract_state(graph, candidate_room)
                candidate_outcome = _evaluate_outcome(graph, candidate_room, candidate_state, evaluate_fn)
                outcome_deltas.append(abs(candidate_outcome - baseline_outcome))
                state_deltas.append(abs(candidate_state[name] - baseline_state[name]))
            results.append(
                SensitivityResult(
                    variable=name,
                    mean_absolute_outcome_delta=float(np.mean(outcome_deltas)),
                    max_absolute_outcome_delta=float(np.max(outcome_deltas)),
                    mean_absolute_state_delta=float(np.mean(state_deltas)),
                    samples=count,
                )
            )
    else:
        for name in graph.variables():
            outcome_deltas = []
            state_deltas = []
            for _ in range(count):
                delta = float(generator.normal(0.0, sigma))
                candidate_state = dict(baseline_state)
                candidate_state[name] = candidate_state[name] + delta
                candidate_outcome = _evaluate_outcome(graph, room_or_state, candidate_state, evaluate_fn)
                outcome_deltas.append(abs(candidate_outcome - baseline_outcome))
                state_deltas.append(abs(delta))
            results.append(
                SensitivityResult(
                    variable=name,
                    mean_absolute_outcome_delta=float(np.mean(outcome_deltas)),
                    max_absolute_outcome_delta=float(np.max(outcome_deltas)),
                    mean_absolute_state_delta=float(np.mean(state_deltas)),
                    samples=count,
                )
            )

    ranked = sorted(results)
    if event_log is not None:
        event_log.record(
            "MUTATE",
            payload={
                "analysis": "perturbation_sensitivity",
                "ranked_variables": [result.variable for result in ranked],
            },
        )
    return ranked


def hidden_variable_hypotheses(
    graph: CausalGraph,
    samples: Sequence[Room | Mapping[str, Any]],
    correlation_threshold: float = 0.8,
    event_log: EventLog | None = None,
) -> list[HiddenVariableHypothesis]:
    """Propose latent-variable hypotheses for strong unexplained correlations."""
    threshold = abs(_validate_finite_float(correlation_threshold, "correlation_threshold"))
    if threshold > 1.0:
        raise ValueError("correlation_threshold must be in [0, 1]")
    if len(samples) < 2:
        return []

    states = [_extract_state(graph, sample) for sample in samples]
    variables = graph.variables()
    matrix = np.asarray([[state[name] for name in variables] for state in states], dtype=float)
    correlations = np.corrcoef(matrix, rowvar=False)
    hypotheses: list[HiddenVariableHypothesis] = []

    for left in range(len(variables)):
        for right in range(left + 1, len(variables)):
            correlation = float(correlations[left, right])
            if not np.isfinite(correlation) or abs(correlation) < threshold:
                continue
            variable_a = variables[left]
            variable_b = variables[right]
            if _pair_explained_by_graph(graph, variable_a, variable_b):
                continue
            hypotheses.append(
                HiddenVariableHypothesis(
                    variable_a=variable_a,
                    variable_b=variable_b,
                    observed_correlation=correlation,
                    candidate_name=f"latent_{variable_a}_{variable_b}",
                    reason=(
                        f"{variable_a!r} and {variable_b!r} show correlation {correlation:.3f} "
                        "without a direct path or shared known ancestor in the current DAG."
                    ),
                )
            )

    ranked = sorted(hypotheses, key=lambda item: abs(item.observed_correlation), reverse=True)
    if event_log is not None and ranked:
        event_log.record(
            "SELECT",
            payload={
                "analysis": "hidden_variable_hypotheses",
                "count": len(ranked),
                "strongest_pair": [ranked[0].variable_a, ranked[0].variable_b],
            },
        )
    return ranked


def analyze_causality(
    graph: CausalGraph,
    room_or_state: Room | Mapping[str, Any],
    evaluate_fn: Callable[[Any], float],
    interventions: Sequence[tuple[str, float]] = (),
    counterfactual_queries: Sequence[tuple[str, float]] = (),
    sensitivity_scale: float = 0.05,
    sensitivity_samples: int = 16,
    correlation_samples: Sequence[Room | Mapping[str, Any]] = (),
    correlation_threshold: float = 0.8,
    rng: np.random.Generator | None = None,
    event_log: EventLog | None = None,
) -> CausalDiscoveryResult:
    """Run a bundled causal-analysis pass and return all outputs together."""
    intervention_results = [
        intervene(graph, variable, value, room_or_state, evaluate_fn, event_log=event_log)
        for variable, value in interventions
    ]
    counterfactual_results = [
        counterfactual(graph, variable, value, room_or_state, evaluate_fn, event_log=event_log)
        for variable, value in counterfactual_queries
    ]
    sensitivities = perturbation_sensitivity(
        graph,
        room_or_state,
        evaluate_fn,
        scale=sensitivity_scale,
        samples_per_variable=sensitivity_samples,
        rng=rng,
        event_log=event_log,
    )
    hidden = hidden_variable_hypotheses(
        graph,
        correlation_samples,
        correlation_threshold=correlation_threshold,
        event_log=event_log,
    )
    if event_log is not None:
        event_log.record(
            "MERGE",
            payload={
                "analysis": "causal_discovery",
                "interventions": len(intervention_results),
                "counterfactuals": len(counterfactual_results),
                "sensitivities": len(sensitivities),
                "hidden_variable_hypotheses": len(hidden),
            },
        )
    return CausalDiscoveryResult(
        graph=graph,
        interventions=intervention_results,
        counterfactuals=counterfactual_results,
        sensitivities=sensitivities,
        hidden_variable_hypotheses=hidden,
    )


def _pair_explained_by_graph(graph: CausalGraph, variable_a: str, variable_b: str) -> bool:
    if variable_b in graph.children(variable_a) or variable_a in graph.children(variable_b):
        return True
    if variable_b in graph.descendants(variable_a) or variable_a in graph.descendants(variable_b):
        return True
    common = set(graph.ancestors(variable_a)).intersection(graph.ancestors(variable_b))
    return bool(common)


__all__ = [
    "CausalDiscoveryResult",
    "CausalEdge",
    "CausalGraph",
    "CounterfactualResult",
    "HiddenVariableHypothesis",
    "InterventionResult",
    "SensitivityResult",
    "analyze_causality",
    "counterfactual",
    "hidden_variable_hypotheses",
    "intervene",
    "perturbation_sensitivity",
]
