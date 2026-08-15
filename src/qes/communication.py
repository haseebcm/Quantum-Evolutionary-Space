"""Phase 6 -- reality-to-reality communication.

This module adds a classical in-memory communication layer for QES objects:
`Room` instances (and optionally `World` / `Universe` objects) can exchange
state snapshots, knowledge payloads, equation populations, reusable patterns,
and simple resource/coordination proposals. Nothing here is literal
physics -- these are ordinary Python objects exchanging validated dict payloads
and mutating their in-process memory/compute fields on a conventional CPU.
"""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np

from qes.events import EventLog
from qes.patterns import Pattern, PatternMemory
from qes.permission import GenesisPermission
from qes.room import Room
from qes.space import QESSpace

MESSAGE_KINDS = {
    "STATE",
    "KNOWLEDGE",
    "EQUATION",
    "PATTERN",
    "RESOURCE_OFFER",
    "RESOURCE_REQUEST",
    "COMPETE",
    "COOPERATE",
    "COALITION_INVITE",
}


def _clone_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _require_memory(entity: Any) -> dict[str, Any]:
    memory = getattr(entity, "memory", None)
    if not isinstance(memory, dict):
        raise TypeError(f"participant {getattr(entity, 'id', '<unknown>')!r} must expose a dict memory")
    return memory


def _as_payload_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _finite_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _resource_amounts(compute: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in compute.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = float(value)
    return result


def _entity_state_snapshot(entity: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name in ("x", "x_star", "lower", "upper", "activation", "state", "weight", "fields", "time"):
        if hasattr(entity, name):
            snapshot[name] = _clone_value(getattr(entity, name))
    compute = getattr(entity, "compute", None)
    if isinstance(compute, dict):
        snapshot["compute"] = copy.deepcopy(compute)
    telemetry = getattr(entity, "telemetry", None)
    if callable(telemetry):
        value = telemetry()
        if is_dataclass(value) and not isinstance(value, type):
            snapshot["telemetry"] = asdict(value)
        else:
            snapshot["telemetry"] = copy.deepcopy(value)
    return snapshot


def _knowledge_view(entity: Any) -> dict[str, Any]:
    memory = _require_memory(entity)
    knowledge = memory.get("knowledge", {})
    if knowledge is None:
        return {}
    if not isinstance(knowledge, Mapping):
        raise TypeError("memory['knowledge'] must be a mapping when used for knowledge exchange")
    return dict(copy.deepcopy(knowledge))


def _equation_key(equation: Any, *, fallback_index: int) -> str:
    identifier = getattr(equation, "id", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    return f"equation-{fallback_index}:{equation!r}"


def _equation_payload(equation: Any, *, fallback_index: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": _equation_key(equation, fallback_index=fallback_index),
        "repr": repr(equation),
    }
    theta = getattr(equation, "theta", None)
    if isinstance(theta, Mapping):
        payload["theta"] = dict(theta)
    return payload


def _extract_equations(entity: Any) -> list[Any]:
    equations = getattr(entity, "equations", None)
    if isinstance(equations, list):
        return list(copy.deepcopy(equations))
    if isinstance(equations, dict):
        return [copy.deepcopy(value) for value in equations.values()]
    return []


def _merge_equations(entity: Any, incoming: Sequence[Any]) -> None:
    if not incoming:
        return
    equations = getattr(entity, "equations", None)
    if isinstance(equations, list):
        existing = {_equation_key(item, fallback_index=index) for index, item in enumerate(equations)}
        for index, item in enumerate(incoming):
            key = _equation_key(item, fallback_index=index)
            if key not in existing:
                equations.append(copy.deepcopy(item))
                existing.add(key)
    elif isinstance(equations, dict):
        for index, item in enumerate(incoming):
            key = _equation_key(item, fallback_index=index)
            equations.setdefault(key, copy.deepcopy(item))


def _extract_patterns(entity: Any) -> list[Pattern]:
    memory = _require_memory(entity)
    result: list[Pattern] = []
    stored = memory.get("patterns", [])
    if stored:
        if not isinstance(stored, list) or not all(isinstance(pattern, Pattern) for pattern in stored):
            raise TypeError("memory['patterns'] must be a list[Pattern] when used for pattern exchange")
        result.extend(copy.deepcopy(stored))
    stored_memory = memory.get("pattern_memory")
    if isinstance(stored_memory, PatternMemory):
        result.extend(copy.deepcopy(stored_memory.all_patterns()))

    deduped: dict[str, Pattern] = {}
    for pattern in result:
        deduped[pattern.id] = pattern
    return list(deduped.values())


def _pattern_payload(pattern: Pattern) -> dict[str, Any]:
    return {
        "id": pattern.id,
        "intent": pattern.intent,
        "context": dict(pattern.context),
        "phi": pattern.phi,
        "cci": pattern.cci,
        "margin": pattern.margin,
        "uses": pattern.uses,
    }


def _merge_patterns(memory: dict[str, Any], incoming: Sequence[Pattern]) -> None:
    patterns = memory.setdefault("patterns", [])
    if not isinstance(patterns, list):
        raise TypeError("memory['patterns'] must be a list when storing exchanged patterns")
    existing = {pattern.id for pattern in patterns if isinstance(pattern, Pattern)}
    for pattern in incoming:
        if pattern.id not in existing:
            patterns.append(copy.deepcopy(pattern))
            existing.add(pattern.id)


def _admitted(room: Room, permission_gate: GenesisPermission | None) -> bool:
    if permission_gate is None:
        return True
    result = permission_gate.evaluate(
        room.x,
        room.lower,
        room.upper,
        room.gates.get("cci_weights"),
        room.couplings,
    )
    return bool(result.admitted)


@dataclass
class Message:
    """Validated communication payload passed through `CommunicationFabric`.

    Phase 6 communication is plain in-memory message passing: sender/receiver
    ids name Python objects already registered with the fabric, and `payload`
    is a dict describing the state/knowledge/resource transfer being modelled.
    """

    sender_id: str
    receiver_id: str | None
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.sender_id, str):
            raise TypeError("sender_id must be a string")
        if not self.sender_id:
            raise ValueError("sender_id must be non-empty")
        if self.receiver_id not in (None, "*"):
            if not isinstance(self.receiver_id, str):
                raise TypeError("receiver_id must be a string, '*', or None")
            if not self.receiver_id:
                raise ValueError("receiver_id must be non-empty when provided")
        if self.kind not in MESSAGE_KINDS:
            raise ValueError(f"kind must be one of {sorted(MESSAGE_KINDS)}, got {self.kind!r}")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dict")
        self.timestamp = _finite_float("timestamp", self.timestamp)

    @property
    def is_broadcast(self) -> bool:
        return self.receiver_id in (None, "*")


@dataclass
class ResourceBid:
    """One room's request or offer for a finite resource pool."""

    room_id: str
    resource: str
    amount: float
    expected_utility: float
    max_allocation: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.room_id, str):
            raise TypeError("room_id must be a string")
        if not self.room_id:
            raise ValueError("room_id must be non-empty")
        if not isinstance(self.resource, str):
            raise TypeError("resource must be a string")
        if not self.resource:
            raise ValueError("resource must be non-empty")
        self.amount = _finite_float("amount", self.amount)
        self.expected_utility = _finite_float("expected_utility", self.expected_utility)
        if self.amount < 0.0:
            raise ValueError("amount must be >= 0")
        if self.expected_utility < 0.0:
            raise ValueError("expected_utility must be >= 0")
        if self.max_allocation is not None:
            self.max_allocation = _finite_float("max_allocation", self.max_allocation)
            if self.max_allocation < 0.0:
                raise ValueError("max_allocation must be >= 0")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")


@dataclass
class Coalition:
    """A named coalition of participants cooperating toward one shared goal."""

    name: str
    member_ids: list[str]
    goal: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name:
            raise ValueError("name must be non-empty")
        if not isinstance(self.member_ids, list) or not all(
            isinstance(member_id, str) for member_id in self.member_ids
        ):
            raise TypeError("member_ids must be a list of strings")
        if not self.member_ids:
            raise ValueError("member_ids must be non-empty")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("member_ids must be unique")
        if not isinstance(self.goal, dict):
            raise TypeError("goal must be a dict")


class CommunicationFabric:
    """In-memory Phase 6 message bus for rooms/worlds/universes.

    The fabric sits beside a `QESSpace` (or any explicit participant list) and
    provides validated message passing plus concrete state mutation helpers:
    exchanges record what was sent *and* update receiver memory/knowledge/
    equation/pattern stores so communication has an observable computational
    effect.
    """

    def __init__(
        self,
        participants: QESSpace | Sequence[Any],
        *,
        event_log: EventLog | None = None,
        pattern_memory: PatternMemory | None = None,
    ) -> None:
        if isinstance(participants, QESSpace):
            participant_iterable: Iterable[Any] = participants.rooms.values()
        elif isinstance(participants, Sequence):
            participant_iterable = participants
        else:
            raise TypeError("participants must be a QESSpace or a sequence of objects with ids")

        self._participants: dict[str, Any] = {}
        for participant in participant_iterable:
            participant_id = getattr(participant, "id", None)
            if not isinstance(participant_id, str) or not participant_id:
                raise ValueError("every participant must expose a non-empty string id")
            if participant_id in self._participants:
                raise ValueError(f"duplicate participant id {participant_id!r}")
            self._participants[participant_id] = participant
        self._messages: list[Message] = []
        self.event_log = event_log
        self.pattern_memory = pattern_memory

    @property
    def participants(self) -> dict[str, Any]:
        return dict(self._participants)

    def send(self, message: Message) -> Message:
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")
        self._require_registered(message.sender_id)
        if not message.is_broadcast:
            assert message.receiver_id is not None
            self._require_registered(message.receiver_id)
        self._messages.append(message)
        self._record_event(
            "EXECUTE",
            payload={
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "kind": message.kind,
            },
            parent_ids=[message.sender_id],
            timestamp=message.timestamp,
        )
        return message

    def broadcast(self, sender_id: str, kind: str, payload: Mapping[str, Any]) -> Message:
        message = Message(sender_id=sender_id, receiver_id="*", kind=kind, payload=_as_payload_dict(payload))
        return self.send(message)

    def inbox(self, room_id: str) -> list[Message]:
        self._require_registered(room_id)
        return [
            message
            for message in self._messages
            if message.receiver_id == room_id or message.receiver_id in (None, "*")
        ]

    def exchange_state(self, room_a: Any, room_b: Any) -> None:
        state_a = _entity_state_snapshot(room_a)
        state_b = _entity_state_snapshot(room_b)
        _require_memory(room_a).setdefault("received_state", {})[room_b.id] = _clone_value(state_b)
        _require_memory(room_b).setdefault("received_state", {})[room_a.id] = _clone_value(state_a)
        self.send(Message(room_a.id, room_b.id, "STATE", {"state": state_a}))
        self.send(Message(room_b.id, room_a.id, "STATE", {"state": state_b}))
        self._record_event(
            "MUTATE",
            payload={"exchange": "state", "participants": [room_a.id, room_b.id]},
            parent_ids=[room_a.id, room_b.id],
        )

    def exchange_knowledge(self, room_a: Any, room_b: Any) -> None:
        knowledge_a = _knowledge_view(room_a)
        knowledge_b = _knowledge_view(room_b)
        memory_a = _require_memory(room_a)
        memory_b = _require_memory(room_b)
        memory_a.setdefault("received_knowledge", {})[room_b.id] = copy.deepcopy(knowledge_b)
        memory_b.setdefault("received_knowledge", {})[room_a.id] = copy.deepcopy(knowledge_a)
        self.send(Message(room_a.id, room_b.id, "KNOWLEDGE", {"knowledge": knowledge_a}))
        self.send(Message(room_b.id, room_a.id, "KNOWLEDGE", {"knowledge": knowledge_b}))
        self._record_event(
            "MUTATE",
            payload={"exchange": "knowledge", "participants": [room_a.id, room_b.id]},
            parent_ids=[room_a.id, room_b.id],
        )

    def exchange_equations(self, room_a: Any, room_b: Any) -> None:
        equations_a = _extract_equations(room_a)
        equations_b = _extract_equations(room_b)
        memory_a = _require_memory(room_a)
        memory_b = _require_memory(room_b)
        memory_a.setdefault("received_equations", {})[room_b.id] = copy.deepcopy(equations_b)
        memory_b.setdefault("received_equations", {})[room_a.id] = copy.deepcopy(equations_a)
        _merge_equations(room_a, equations_b)
        _merge_equations(room_b, equations_a)
        self.send(
            Message(
                room_a.id,
                room_b.id,
                "EQUATION",
                {
                    "equations": [
                        _equation_payload(eq, fallback_index=index)
                        for index, eq in enumerate(equations_a)
                    ]
                },
            )
        )
        self.send(
            Message(
                room_b.id,
                room_a.id,
                "EQUATION",
                {
                    "equations": [
                        _equation_payload(eq, fallback_index=index)
                        for index, eq in enumerate(equations_b)
                    ]
                },
            )
        )
        self._record_event(
            "MUTATE",
            payload={"exchange": "equation", "participants": [room_a.id, room_b.id]},
            parent_ids=[room_a.id, room_b.id],
        )

    def exchange_pattern(self, room_a: Any, room_b: Any) -> None:
        patterns_a = _extract_patterns(room_a)
        patterns_b = _extract_patterns(room_b)
        memory_a = _require_memory(room_a)
        memory_b = _require_memory(room_b)
        memory_a.setdefault("received_patterns", {})[room_b.id] = copy.deepcopy(patterns_b)
        memory_b.setdefault("received_patterns", {})[room_a.id] = copy.deepcopy(patterns_a)
        _merge_patterns(memory_a, patterns_b)
        _merge_patterns(memory_b, patterns_a)
        if self.pattern_memory is not None:
            for pattern in patterns_a + patterns_b:
                self.pattern_memory.store(copy.deepcopy(pattern))
        self.send(
            Message(
                room_a.id,
                room_b.id,
                "PATTERN",
                {"patterns": [_pattern_payload(p) for p in patterns_a]},
            )
        )
        self.send(
            Message(
                room_b.id,
                room_a.id,
                "PATTERN",
                {"patterns": [_pattern_payload(p) for p in patterns_b]},
            )
        )
        self._record_event(
            "MUTATE",
            payload={"exchange": "pattern", "participants": [room_a.id, room_b.id]},
            parent_ids=[room_a.id, room_b.id],
        )

    def _require_registered(self, participant_id: str) -> None:
        if participant_id not in self._participants:
            raise KeyError(f"unknown participant id: {participant_id!r}")

    def _record_event(
        self,
        kind: str,
        *,
        payload: dict[str, Any],
        parent_ids: list[str],
        timestamp: float | None = None,
    ) -> None:
        if self.event_log is None:
            return
        self.event_log.record(
            kind,
            payload=payload,
            parent_ids=parent_ids,
            timestamp=time.time() if timestamp is None else timestamp,
        )


class ResourceNegotiation:
    """Simple classical resource-bidding protocol for room compute budgets."""

    def __init__(
        self,
        rooms: QESSpace | Sequence[Room],
        *,
        permission_gate: GenesisPermission | None = None,
        event_log: EventLog | None = None,
    ) -> None:
        if isinstance(rooms, QESSpace):
            self._rooms = {room.id: room for room in rooms.rooms.values()}
        else:
            self._rooms = {}
            for room in rooms:
                if not isinstance(room, Room):
                    raise TypeError("rooms must contain Room instances")
                self._rooms[room.id] = room
        self.permission_gate = permission_gate
        self.event_log = event_log
        self._bids: list[ResourceBid] = []

    @property
    def bids(self) -> list[ResourceBid]:
        return list(self._bids)

    def submit_bid(
        self,
        room_id: str,
        bid: Mapping[str, Any] | ResourceBid,
    ) -> ResourceBid:
        if isinstance(bid, ResourceBid):
            resource_bid = bid
        else:
            bid_data = dict(bid)
            bid_data.setdefault("room_id", room_id)
            resource_bid = ResourceBid(**bid_data)
        if resource_bid.room_id != room_id:
            raise ValueError("room_id argument must match bid.room_id")
        room = self._require_room(room_id)
        if not _admitted(room, self.permission_gate):
            raise ValueError(f"room {room_id!r} is not admitted by the supplied permission gate")
        self._bids.append(resource_bid)
        self._record_event(
            "MUTATE",
            payload={
                "action": "submit_bid",
                "room_id": room_id,
                "resource": resource_bid.resource,
                "amount": resource_bid.amount,
                "expected_utility": resource_bid.expected_utility,
            },
            parent_ids=[room_id],
        )
        return resource_bid

    def allocate(
        self,
        resource: str,
        total_amount: float,
        *,
        default_cap: float | None = None,
    ) -> dict[str, float]:
        if not isinstance(resource, str):
            raise TypeError("resource must be a string")
        if not resource:
            raise ValueError("resource must be non-empty")
        total = _finite_float("total_amount", total_amount)
        if total < 0.0:
            raise ValueError("total_amount must be >= 0")
        cap = None if default_cap is None else _finite_float("default_cap", default_cap)
        if cap is not None and cap < 0.0:
            raise ValueError("default_cap must be >= 0")

        active_bids = [bid for bid in self._bids if bid.resource == resource]
        allocations = {bid.room_id: 0.0 for bid in active_bids}
        if not active_bids or total == 0.0:
            self._bids = [bid for bid in self._bids if bid.resource != resource]
            return allocations

        caps = {
            bid.room_id: min(
                bid.amount,
                bid.max_allocation if bid.max_allocation is not None else bid.amount,
                cap if cap is not None else bid.amount,
            )
            for bid in active_bids
        }
        scores = {bid.room_id: max(0.0, bid.expected_utility) for bid in active_bids}
        remaining = total
        open_ids = {bid.room_id for bid in active_bids if caps[bid.room_id] > 0.0}

        while remaining > 1e-12 and open_ids:
            total_score = sum(scores[room_id] for room_id in open_ids)
            if total_score <= 0.0:
                shares = {room_id: remaining / len(open_ids) for room_id in open_ids}
            else:
                shares = {room_id: remaining * scores[room_id] / total_score for room_id in open_ids}

            capped_this_round: list[str] = []
            distributed = 0.0
            for room_id in list(open_ids):
                room_remaining = caps[room_id] - allocations[room_id]
                grant = min(room_remaining, shares[room_id])
                allocations[room_id] += grant
                distributed += grant
                if allocations[room_id] >= caps[room_id] - 1e-12:
                    capped_this_round.append(room_id)
            remaining = max(0.0, remaining - distributed)
            for room_id in capped_this_round:
                open_ids.discard(room_id)
            if distributed <= 1e-12:
                break

        for room_id, grant in allocations.items():
            room = self._require_room(room_id)
            room.compute[resource] = float(room.compute.get(resource, 0.0)) + grant
            room.compute.setdefault("negotiated", {})[resource] = grant

        self._record_event(
            "MUTATE",
            payload={
                "action": "allocate",
                "resource": resource,
                "total_amount": total,
                "allocations": allocations,
            },
            parent_ids=list(allocations),
        )
        self._bids = [bid for bid in self._bids if bid.resource != resource]
        return allocations

    def _record_event(
        self,
        kind: str,
        *,
        payload: dict[str, Any],
        parent_ids: list[str],
    ) -> None:
        if self.event_log is None:
            return
        self.event_log.record(kind, payload=payload, parent_ids=parent_ids, timestamp=time.time())

    def _require_room(self, room_id: str) -> Room:
        room = self._rooms.get(room_id)
        if room is None:
            raise KeyError(f"unknown room id: {room_id!r}")
        return room


def compete(
    room_a: Room,
    room_b: Room,
    score_fn: Callable[[Room], float],
    *,
    event_log: EventLog | None = None,
) -> str:
    """Score two rooms and return the winning room id.

    Larger scores win. Ties are broken deterministically by lexical room id.
    """

    score_a = _finite_float("score_a", score_fn(room_a))
    score_b = _finite_float("score_b", score_fn(room_b))
    if score_a > score_b:
        winner = room_a
    elif score_b > score_a:
        winner = room_b
    else:
        winner = min((room_a, room_b), key=lambda room: room.id)
    room_a.memory.setdefault("competitions", []).append(
        {"opponent": room_b.id, "score": score_a, "winner": winner.id}
    )
    room_b.memory.setdefault("competitions", []).append(
        {"opponent": room_a.id, "score": score_b, "winner": winner.id}
    )
    if event_log is not None:
        event_log.record(
            "SELECT",
            payload={
                "participants": [room_a.id, room_b.id],
                "winner": winner.id,
                "scores": {room_a.id: score_a, room_b.id: score_b},
            },
            parent_ids=[room_a.id, room_b.id],
            timestamp=time.time(),
        )
    return winner.id


def cooperate(
    rooms: Sequence[Room],
    aggregate_fn: Callable[[Sequence[Room]], dict[str, Any]],
    *,
    event_log: EventLog | None = None,
) -> dict[str, Any]:
    """Aggregate cooperative output across several rooms and share it back."""

    if not rooms:
        raise ValueError("rooms must be non-empty")
    payload = aggregate_fn(rooms)
    if not isinstance(payload, dict):
        raise TypeError("aggregate_fn must return a dict")
    for room in rooms:
        room.memory.setdefault("cooperations", []).append(copy.deepcopy(payload))
    if event_log is not None:
        event_log.record(
            "MERGE",
            payload={"participants": [room.id for room in rooms], "payload": payload},
            parent_ids=[room.id for room in rooms],
            timestamp=time.time(),
        )
    return payload


def form_coalition(
    rooms: Sequence[Room],
    name: str,
    *,
    goal: Mapping[str, Any] | None = None,
    permission_gate: GenesisPermission | None = None,
    event_log: EventLog | None = None,
) -> Coalition:
    """Create a coalition of admissible rooms and store membership in memory."""

    if not rooms:
        raise ValueError("rooms must be non-empty")
    if permission_gate is not None:
        inadmissible = [room.id for room in rooms if not _admitted(room, permission_gate)]
        if inadmissible:
            raise ValueError(f"cannot form coalition with inadmissible rooms: {inadmissible}")
    coalition = Coalition(name=name, member_ids=[room.id for room in rooms], goal=dict(goal or {}))
    for room in rooms:
        room.memory.setdefault("coalitions", {})[name] = {
            "members": list(coalition.member_ids),
            "goal": copy.deepcopy(coalition.goal),
        }
    if event_log is not None:
        event_log.record(
            "MERGE",
            payload={"coalition": coalition.name, "members": coalition.member_ids, "goal": coalition.goal},
            parent_ids=list(coalition.member_ids),
            timestamp=time.time(),
        )
    return coalition
