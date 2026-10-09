"""Event-sourcing, lineage graph, and deterministic replay (Phase 1,
"Harden the core").

Every important QES state transition can be expressed as one of eight
canonical event kinds:

    SPAWN -> EXECUTE -> DIVERGE -> PERMISSION -> MUTATE -> SELECT -> MERGE -> COLLAPSE

`EventLog` is an append-only record of these events (with deterministic,
monotonically increasing ids), `LineageGraph` tracks parent/child
relationships between realities (rooms, equations, decisions, ...) built
from those events, and `state_hash` + `DeterministicReplay` let a caller
verify that re-running a recorded sequence of events from a checkpoint
reproduces bit-identical states -- the basic building block for
reproducible experiments (see also `Room.lineage`, `Universe.event_log`,
`QESSpace.history`, which this module complements rather than replaces).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EVENT_KINDS = {
    "SPAWN",
    "EXECUTE",
    "DIVERGE",
    "PERMISSION",
    "MUTATE",
    "SELECT",
    "MERGE",
    "COLLAPSE",
}

_event_id_counter = itertools.count(1)


def _next_event_id() -> str:
    return f"EV-{next(_event_id_counter):06d}"


@dataclass
class Event:
    """One recorded state transition.

    Attributes:
        kind: one of `EVENT_KINDS` (SPAWN, EXECUTE, DIVERGE, PERMISSION,
            MUTATE, SELECT, MERGE, COLLAPSE).
        payload: arbitrary JSON-ish metadata describing the transition
            (e.g. `{"room_id": "R-00001", "delta": 0.2}`).
        parent_ids: ids of the event(s)/entities this event descends from
            (empty for a root SPAWN).
        timestamp: virtual or wall-clock time the event occurred at.
        id: unique, monotonically increasing event identifier.
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    id: str = field(default_factory=_next_event_id)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {sorted(EVENT_KINDS)}, got {self.kind!r}")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dict")
        if not isinstance(self.parent_ids, list) or not all(
            isinstance(p, str) for p in self.parent_ids
        ):
            raise TypeError("parent_ids must be a list of strings")
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        self.timestamp = timestamp

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) representation of this event."""
        return {
            "id": self.id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "parent_ids": list(self.parent_ids),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Reconstruct an `Event` previously serialized via `to_dict()`."""
        return cls(
            kind=data["kind"],
            payload=dict(data.get("payload", {})),
            parent_ids=list(data.get("parent_ids", [])),
            timestamp=data.get("timestamp", 0.0),
            id=data["id"],
        )


class EventLog:
    """Append-only record of `Event`s: the QES event-sourcing layer.

    A caller reconstructs *any* prior QES state purely by replaying the
    ordered sequence of events recorded here (in principle) -- in practice
    this is used alongside, not instead of, the per-object `snapshot()`s,
    as a cheaper audit trail plus the basis for `DeterministicReplay`.
    """

    def __init__(self, max_events: int = 10000) -> None:
        if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self.max_events = max_events
        self._events: list[Event] = []

    def record(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
        timestamp: float = 0.0,
    ) -> Event:
        """Append and return a new event."""
        if len(self._events) >= self.max_events:
            raise OverflowError("event log capacity exceeded; archive before recording more events")
        event = Event(
            kind=kind,
            payload=copy.deepcopy(payload or {}),
            parent_ids=list(parent_ids or []),
            timestamp=timestamp,
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def filter(self, kind: str) -> list[Event]:
        """All recorded events of one `kind`."""
        if kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {sorted(EVENT_KINDS)}, got {kind!r}")
        return [e for e in self._events if e.kind == kind]

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize the whole log to a list of plain dicts."""
        return [e.to_dict() for e in self._events]

    def restore(self, records: list[dict[str, Any]]) -> None:
        """Replace this log's contents with previously serialized events."""
        self._events = [Event.from_dict(r) for r in records]

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)


class LineageGraph:
    """Parent/child provenance graph over realities (rooms, equations,
    decisions, resource allocations, selection outcomes, ...).

    Complements `Room.lineage` (a flat ancestor list on one room) with a
    full graph that can answer ancestor/descendant queries over *any*
    tracked entity id, not just rooms.
    """

    def __init__(self) -> None:
        self._parents: dict[str, list[str]] = {}
        self._children: dict[str, list[str]] = {}
        self._kinds: dict[str, str] = {}
        self._payloads: dict[str, dict[str, Any]] = {}

    def add_node(
        self,
        node_id: str,
        parents: tuple[str, ...] | list[str] = (),
        kind: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Register `node_id` with the given `parents` (must already exist,
        except for root nodes with no parents)."""
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a non-empty string")
        for parent_id in parents:
            if parent_id not in self._parents and parent_id not in self._children:
                raise KeyError(f"unknown parent node: {parent_id!r}")
        self._parents[node_id] = list(parents)
        self._children.setdefault(node_id, [])
        self._kinds[node_id] = kind
        self._payloads[node_id] = dict(payload or {})
        for parent_id in parents:
            self._children.setdefault(parent_id, []).append(node_id)

    def ancestors(self, node_id: str) -> list[str]:
        """All ancestors of `node_id`, nearest first, deduplicated."""
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
        seen: list[str] = []
        frontier = list(self._children.get(node_id, []))
        while frontier:
            child_id = frontier.pop(0)
            if child_id in seen:
                continue
            seen.append(child_id)
            frontier.extend(self._children.get(child_id, []))
        return seen

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) representation of this graph."""
        return {
            "parents": {k: list(v) for k, v in self._parents.items()},
            "kinds": dict(self._kinds),
            "payloads": {k: dict(v) for k, v in self._payloads.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageGraph:
        """Reconstruct a `LineageGraph` previously serialized via `to_dict()`."""
        graph = cls()
        parents = data.get("parents", {})
        kinds = data.get("kinds", {})
        payloads = data.get("payloads", {})
        # Insert in an order that satisfies each node's parent dependency.
        remaining = dict(parents)
        while remaining:
            ready = [
                node_id
                for node_id, node_parents in remaining.items()
                if all(p in graph._parents for p in node_parents)
            ]
            if not ready:
                raise ValueError("lineage data contains a cycle or unresolved parent reference")
            for node_id in ready:
                graph.add_node(
                    node_id,
                    parents=remaining[node_id],
                    kind=kinds.get(node_id, ""),
                    payload=payloads.get(node_id, {}),
                )
                del remaining[node_id]
        return graph


def state_hash(x: np.ndarray) -> str:
    """Deterministic content hash of a state vector/array.

    Used by `DeterministicReplay` to check bit-for-bit reproducibility
    without keeping every intermediate array in memory: two states hash
    identically iff they have the same shape, dtype, and bytes.
    """
    arr = np.asarray(x)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


@dataclass
class ReplayMismatch:
    """One state that failed to reproduce during a deterministic replay."""

    step_id: str
    expected_hash: str
    actual_hash: str


@dataclass
class ReplayReport:
    """Result of `DeterministicReplay.verify_all()`."""

    matches: bool
    mismatches: list[ReplayMismatch] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.matches


class DeterministicReplay:
    """Records state hashes for a sequence of steps, then verifies that a
    later re-run (e.g. after `restore()`-ing a checkpoint and re-applying
    the same `EventLog`) reproduces the exact same states.
    """

    def __init__(self) -> None:
        self._recorded: dict[str, str] = {}
        self._order: list[str] = []

    def record(self, step_id: str, state: np.ndarray) -> str:
        """Record the hash of `state` at `step_id` (e.g. an event id or
        tick index) for later verification. Returns the computed hash."""
        if not isinstance(step_id, str) or not step_id:
            raise ValueError("step_id must be a non-empty string")
        digest = state_hash(state)
        if step_id not in self._recorded:
            self._order.append(step_id)
        self._recorded[step_id] = digest
        return digest

    def verify(self, step_id: str, state: np.ndarray) -> bool:
        """True iff `state` hashes to the same value previously recorded
        at `step_id`.

        Raises:
            KeyError: if no hash was ever recorded for `step_id`.
        """
        if step_id not in self._recorded:
            raise KeyError(f"no recorded hash for step_id {step_id!r}")
        return state_hash(state) == self._recorded[step_id]

    def verify_all(self, replayed: dict[str, np.ndarray]) -> ReplayReport:
        """Verify a full batch of replayed states against every recorded
        step at once, reporting every mismatch (not just the first)."""
        mismatches = []
        for step_id in self._order:
            if step_id not in replayed:
                continue
            actual = state_hash(replayed[step_id])
            expected = self._recorded[step_id]
            if actual != expected:
                mismatches.append(
                    ReplayMismatch(step_id=step_id, expected_hash=expected, actual_hash=actual)
                )
        return ReplayReport(matches=not mismatches, mismatches=mismatches)


__all__ = [
    "EVENT_KINDS",
    "Event",
    "EventLog",
    "LineageGraph",
    "state_hash",
    "ReplayMismatch",
    "ReplayReport",
    "DeterministicReplay",
]
