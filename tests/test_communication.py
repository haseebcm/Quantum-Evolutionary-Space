import numpy as np
import pytest

from qes.communication import (
    Coalition,
    CommunicationFabric,
    Message,
    ResourceNegotiation,
    compete,
    cooperate,
    form_coalition,
)
from qes.equation_forge import EquationForge
from qes.events import EventLog
from qes.patterns import Pattern, PatternMemory
from qes.permission import GenesisPermission
from qes.room import Room
from qes.space import QESSpace


def make_room(x=None) -> Room:
    state = np.asarray(x if x is not None else [0.0, 0.0], dtype=float)
    return Room(
        x=state.copy(),
        x_star=np.ones_like(state) * 0.5,
        lower=-np.ones_like(state),
        upper=np.ones_like(state),
        activation=np.ones_like(state),
    )


def test_message_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Message(sender_id="R-1", receiver_id="R-2", kind="UNKNOWN")


def test_message_rejects_non_dict_payload():
    with pytest.raises(TypeError):
        Message(sender_id="R-1", receiver_id="R-2", kind="STATE", payload=["bad"])  # type: ignore[arg-type]


def test_fabric_send_broadcast_and_inbox():
    room_a = make_room()
    room_b = make_room()
    fabric = CommunicationFabric([room_a, room_b])

    direct = fabric.send(Message(room_a.id, room_b.id, "STATE", {"delta": 1.0}))
    broadcast = fabric.broadcast(room_a.id, "KNOWLEDGE", {"topic": "shared"})

    inbox = fabric.inbox(room_b.id)
    assert direct in inbox
    assert broadcast in inbox
    assert fabric.inbox(room_a.id) == [broadcast]


def test_exchange_state_records_peer_snapshot_in_memory():
    room_a = make_room([0.1, 0.2])
    room_b = make_room([0.7, 0.9])
    fabric = CommunicationFabric([room_a, room_b])

    fabric.exchange_state(room_a, room_b)

    np.testing.assert_allclose(room_a.memory["received_state"][room_b.id]["x"], room_b.x)
    np.testing.assert_allclose(room_b.memory["received_state"][room_a.id]["x"], room_a.x)


def test_exchange_knowledge_stores_peer_knowledge():
    room_a = make_room()
    room_b = make_room()
    room_a.memory["knowledge"] = {"alpha": 1}
    room_b.memory["knowledge"] = {"beta": 2}
    fabric = CommunicationFabric([room_a, room_b])

    fabric.exchange_knowledge(room_a, room_b)

    assert room_a.memory["received_knowledge"][room_b.id] == {"beta": 2}
    assert room_b.memory["received_knowledge"][room_a.id] == {"alpha": 1}


def test_exchange_equations_merges_equation_populations():
    room_a = make_room()
    room_b = make_room()
    forge = EquationForge()
    eq_a = forge.seed(theta={"a": 1.0})
    eq_b = forge.seed(theta={"b": 2.0})
    room_a.equations = [eq_a]
    room_b.equations = [eq_b]
    fabric = CommunicationFabric([room_a, room_b])

    fabric.exchange_equations(room_a, room_b)

    assert {equation.id for equation in room_a.equations} == {eq_a.id, eq_b.id}
    assert {equation.id for equation in room_b.equations} == {eq_a.id, eq_b.id}
    assert room_a.memory["received_equations"][room_b.id][0].id == eq_b.id


def test_exchange_pattern_merges_patterns_and_shared_memory():
    room_a = make_room()
    room_b = make_room()
    pattern_a = Pattern(intent="design", context={"source": room_a.id}, payload={"x": [1, 2]})
    pattern_b = Pattern(intent="design", context={"source": room_b.id}, payload={"x": [3, 4]})
    room_a.memory["patterns"] = [pattern_a]
    room_b.memory["patterns"] = [pattern_b]
    shared_memory = PatternMemory()
    fabric = CommunicationFabric([room_a, room_b], pattern_memory=shared_memory)

    fabric.exchange_pattern(room_a, room_b)

    assert {pattern.id for pattern in room_a.memory["patterns"]} == {pattern_a.id, pattern_b.id}
    assert {pattern.id for pattern in room_b.memory["patterns"]} == {pattern_a.id, pattern_b.id}
    assert {pattern.id for pattern in shared_memory.all_patterns("design")} == {pattern_a.id, pattern_b.id}


def test_resource_negotiation_allocates_fixed_pool_proportionally_and_respects_caps():
    room_a = make_room()
    room_b = make_room()
    negotiation = ResourceNegotiation([room_a, room_b])
    negotiation.submit_bid(
        room_a.id,
        {"room_id": room_a.id, "resource": "compute", "amount": 10.0, "expected_utility": 1.0},
    )
    negotiation.submit_bid(
        room_b.id,
        {
            "room_id": room_b.id,
            "resource": "compute",
            "amount": 10.0,
            "expected_utility": 3.0,
            "max_allocation": 4.0,
        },
    )

    allocations = negotiation.allocate("compute", 8.0)

    assert allocations[room_b.id] == pytest.approx(4.0)
    assert allocations[room_a.id] == pytest.approx(4.0)
    assert sum(allocations.values()) == pytest.approx(8.0)
    assert room_a.compute["compute"] == pytest.approx(4.0)
    assert room_b.compute["compute"] == pytest.approx(4.0)


def test_resource_negotiation_rejects_bid_from_inadmissible_room():
    room = make_room([9.0, 9.0])
    negotiation = ResourceNegotiation([room], permission_gate=GenesisPermission(theta=1.0))

    with pytest.raises(ValueError):
        negotiation.submit_bid(
            room.id,
            {"room_id": room.id, "resource": "compute", "amount": 1.0, "expected_utility": 1.0},
        )


def test_compete_returns_higher_scoring_room_id():
    room_a = make_room([0.0, 0.0])
    room_b = make_room([0.5, 0.5])

    winner = compete(room_a, room_b, lambda room: float(np.sum(room.x)))

    assert winner == room_b.id
    assert room_a.memory["competitions"][0]["winner"] == room_b.id


def test_cooperate_returns_aggregate_payload_and_records_it():
    room_a = make_room([1.0, 2.0])
    room_b = make_room([3.0, 4.0])

    payload = cooperate(
        [room_a, room_b],
        lambda rooms: {"mean_state": np.mean([room.x for room in rooms], axis=0).tolist()},
    )

    assert payload == {"mean_state": [2.0, 3.0]}
    assert room_a.memory["cooperations"][0] == payload
    assert room_b.memory["cooperations"][0] == payload


def test_form_coalition_returns_valid_coalition_and_updates_room_memory():
    room_a = make_room()
    room_b = make_room()

    coalition = form_coalition([room_a, room_b], "design-alliance", goal={"target": "refinement"})

    assert isinstance(coalition, Coalition)
    assert coalition.member_ids == [room_a.id, room_b.id]
    assert room_a.memory["coalitions"]["design-alliance"]["goal"] == {"target": "refinement"}
    assert room_b.memory["coalitions"]["design-alliance"]["members"] == [room_a.id, room_b.id]


def test_fabric_and_coordination_helpers_record_events_when_event_log_supplied():
    room_a = make_room()
    room_b = make_room()
    room_a.memory["knowledge"] = {"alpha": 1}
    room_b.memory["knowledge"] = {"beta": 2}
    events = EventLog()
    fabric = CommunicationFabric([room_a, room_b], event_log=events)

    fabric.send(Message(room_a.id, room_b.id, "STATE", {"ping": True}))
    fabric.exchange_knowledge(room_a, room_b)
    compete(room_a, room_b, lambda room: float(np.sum(room.x)), event_log=events)
    cooperate([room_a, room_b], lambda rooms: {"count": len(rooms)}, event_log=events)
    form_coalition([room_a, room_b], "coalition", event_log=events)

    kinds = [event.kind for event in events.events]
    assert "EXECUTE" in kinds
    assert "MUTATE" in kinds
    assert "SELECT" in kinds
    assert "MERGE" in kinds


def test_fabric_accepts_qes_space_participants():
    room_a = make_room()
    room_b = make_room()
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    space.spawn([room_a, room_b])
    fabric = CommunicationFabric(space)

    fabric.broadcast(room_a.id, "COOPERATE", {"goal": "share"})

    assert len(fabric.inbox(room_b.id)) == 1


def test_internal_helpers_cover_validation_and_optional_payload_shapes():
    from dataclasses import dataclass

    import qes.communication as communication_module

    class NoMemory:
        id = "broken"
        memory = None

    with pytest.raises(TypeError, match="must expose a dict memory"):
        communication_module._require_memory(NoMemory())
    with pytest.raises(ValueError, match="must be finite"):
        communication_module._finite_float("value", float("nan"))

    amounts = communication_module._resource_amounts({"a": 1, "b": 2.5, "c": True, "d": "x"})
    assert amounts == {"a": 1.0, "b": 2.5}

    @dataclass
    class Telemetry:
        value: int

    class Entity:
        def __init__(self) -> None:
            self.id = "entity"
            self.memory = {}
            self.x = np.array([1.0])
            self.compute = {"compute": 2.0}

        def telemetry(self) -> Telemetry:
            return Telemetry(3)

    snapshot = communication_module._entity_state_snapshot(Entity())
    assert snapshot["compute"] == {"compute": 2.0}
    assert snapshot["telemetry"] == {"value": 3}

    class PlainTelemetryEntity(Entity):
        def __init__(self) -> None:
            super().__init__()
            self.compute = None

        def telemetry(self) -> dict[str, int]:
            return {"value": 4}

    assert communication_module._entity_state_snapshot(PlainTelemetryEntity())["telemetry"] == {"value": 4}


def test_knowledge_equation_and_pattern_helpers_cover_edge_cases():
    import qes.communication as communication_module

    room = make_room()
    room.memory["knowledge"] = None
    assert communication_module._knowledge_view(room) == {}
    room.memory["knowledge"] = ["bad"]
    with pytest.raises(TypeError, match="must be a mapping"):
        communication_module._knowledge_view(room)

    assert communication_module._equation_key(object(), fallback_index=1).startswith("equation-1:")
    payload = communication_module._equation_payload(type("Eq", (), {"theta": {"x": 1.0}})(), fallback_index=0)
    assert payload["theta"] == {"x": 1.0}
    assert "theta" not in communication_module._equation_payload(object(), fallback_index=0)

    holder = type("Holder", (), {"equations": {"a": {"id": "a"}}})()
    assert len(communication_module._extract_equations(holder)) == 1
    assert communication_module._extract_equations(type("Empty", (), {"equations": 1})()) == []

    as_list = type("ListEntity", (), {"equations": [{"id": "a"}]})()
    communication_module._merge_equations(as_list, [])
    communication_module._merge_equations(as_list, [{"id": "a"}, {"id": "b"}])
    assert len(as_list.equations) == 2

    as_dict = type("DictEntity", (), {"equations": {}})()
    communication_module._merge_equations(as_dict, [{"id": "a"}])
    assert "equation-0:{'id': 'a'}" in as_dict.equations or "a" in as_dict.equations
    communication_module._merge_equations(type("OtherEntity", (), {"equations": 1})(), [{"id": "ignored"}])

    room = make_room()
    room.memory["patterns"] = "bad"
    with pytest.raises(TypeError, match="list\\[Pattern\\]"):
        communication_module._extract_patterns(room)

    room.memory["patterns"] = [Pattern(intent="design", context={}, payload={})]
    stored_memory = PatternMemory()
    stored_memory.store(Pattern(intent="design", context={"x": 1}, payload={}))
    room.memory["pattern_memory"] = stored_memory
    assert len(communication_module._extract_patterns(room)) == 2
    room.memory["patterns"] = []
    assert len(communication_module._extract_patterns(room)) == 1

    with pytest.raises(TypeError, match="must be a list when storing exchanged patterns"):
        communication_module._merge_patterns({"patterns": {}}, [])  # type: ignore[arg-type]
    duplicate = Pattern(intent="design", context={}, payload={})
    memory = {"patterns": [duplicate]}
    communication_module._merge_patterns(memory, [duplicate])
    assert memory["patterns"] == [duplicate]


def test_message_resource_bid_and_coalition_validate_inputs():
    with pytest.raises(TypeError, match="sender_id must be a string"):
        Message(sender_id=1, receiver_id="R-1", kind="STATE")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sender_id must be non-empty"):
        Message(sender_id="", receiver_id="R-1", kind="STATE")
    with pytest.raises(TypeError, match="receiver_id must be a string"):
        Message(sender_id="R-1", receiver_id=1, kind="STATE")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="receiver_id must be non-empty"):
        Message(sender_id="R-1", receiver_id="", kind="STATE")
    with pytest.raises(ValueError, match="timestamp must be finite"):
        Message(sender_id="R-1", receiver_id="R-2", kind="STATE", timestamp=float("nan"))
    assert Message(sender_id="R-1", receiver_id=None, kind="STATE").is_broadcast is True

    with pytest.raises(TypeError, match="room_id must be a string"):
        ResourceNegotiation([make_room()]).submit_bid(  # type: ignore[arg-type]
            make_room().id,
            {"room_id": 1, "resource": "compute", "amount": 1.0, "expected_utility": 1.0},
        )
    with pytest.raises(ValueError, match="room_id must be non-empty"):
        communication_module = __import__("qes.communication", fromlist=["ResourceBid"])
        communication_module.ResourceBid(room_id="", resource="compute", amount=1.0, expected_utility=1.0)
    with pytest.raises(TypeError, match="resource must be a string"):
        __import__("qes.communication", fromlist=["ResourceBid"]).ResourceBid(  # type: ignore[attr-defined]
            room_id="r",
            resource=1,
            amount=1.0,
            expected_utility=1.0,
        )
    with pytest.raises(ValueError, match="resource must be non-empty"):
        __import__("qes.communication", fromlist=["ResourceBid"]).ResourceBid(
            room_id="r",
            resource="",
            amount=1.0,
            expected_utility=1.0,
        )
    with pytest.raises(ValueError, match="amount must be >= 0"):
        __import__("qes.communication", fromlist=["ResourceBid"]).ResourceBid(
            room_id="r",
            resource="compute",
            amount=-1.0,
            expected_utility=1.0,
        )
    with pytest.raises(ValueError, match="expected_utility must be >= 0"):
        __import__("qes.communication", fromlist=["ResourceBid"]).ResourceBid(
            room_id="r",
            resource="compute",
            amount=1.0,
            expected_utility=-1.0,
        )
    with pytest.raises(ValueError, match="max_allocation must be >= 0"):
        __import__("qes.communication", fromlist=["ResourceBid"]).ResourceBid(
            room_id="r",
            resource="compute",
            amount=1.0,
            expected_utility=1.0,
            max_allocation=-1.0,
        )
    with pytest.raises(TypeError, match="metadata must be a dict"):
        __import__("qes.communication", fromlist=["ResourceBid"]).ResourceBid(  # type: ignore[attr-defined]
            room_id="r",
            resource="compute",
            amount=1.0,
            expected_utility=1.0,
            metadata=[],
        )

    with pytest.raises(TypeError, match="name must be a string"):
        Coalition(name=1, member_ids=["a"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name must be non-empty"):
        Coalition(name="", member_ids=["a"])
    with pytest.raises(TypeError, match="member_ids must be a list of strings"):
        Coalition(name="c", member_ids=["a", 1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="member_ids must be non-empty"):
        Coalition(name="c", member_ids=[])
    with pytest.raises(ValueError, match="member_ids must be unique"):
        Coalition(name="c", member_ids=["a", "a"])
    with pytest.raises(TypeError, match="goal must be a dict"):
        Coalition(name="c", member_ids=["a"], goal=[])  # type: ignore[arg-type]


def test_fabric_constructor_and_send_validation_paths():
    class Participant:
        def __init__(self, participant_id: str) -> None:
            self.id = participant_id
            self.memory: dict[str, object] = {}

    with pytest.raises(TypeError, match="QESSpace or a sequence"):
        CommunicationFabric(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string id"):
        CommunicationFabric([Participant("")])
    dup = Participant("dup")
    with pytest.raises(ValueError, match="duplicate participant id"):
        CommunicationFabric([dup, dup])

    room_a = make_room()
    room_b = make_room()
    fabric = CommunicationFabric([room_a, room_b])
    assert set(fabric.participants) == {room_a.id, room_b.id}
    with pytest.raises(TypeError, match="message must be a Message"):
        fabric.send("bad")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="unknown participant id"):
        fabric.send(Message("missing", room_b.id, "STATE"))
    with pytest.raises(KeyError, match="unknown participant id"):
        fabric.send(Message(room_a.id, "missing", "STATE"))


def test_exchange_pattern_without_shared_memory_and_unknown_inbox_lookup():
    room_a = make_room()
    room_b = make_room()
    room_a.memory["patterns"] = [Pattern(intent="design", context={}, payload={})]
    room_b.memory["patterns"] = [Pattern(intent="design", context={"b": True}, payload={})]
    fabric = CommunicationFabric([room_a, room_b])
    fabric.exchange_pattern(room_a, room_b)
    with pytest.raises(KeyError, match="unknown participant id"):
        fabric.inbox("missing")


def test_resource_negotiation_validation_and_allocation_edge_cases():
    import qes.communication as communication_module

    room_a = make_room()
    room_b = make_room()
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    space.spawn([room_a, room_b])
    negotiation = ResourceNegotiation(space, event_log=EventLog())
    bid = communication_module.ResourceBid(
        room_id=room_a.id,
        resource="compute",
        amount=1.0,
        expected_utility=0.0,
    )
    returned = negotiation.submit_bid(room_a.id, bid)
    assert returned is bid
    assert negotiation.bids == [bid]

    with pytest.raises(TypeError, match="rooms must contain Room instances"):
        ResourceNegotiation([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must match"):
        negotiation.submit_bid(
            room_a.id,
            {"room_id": room_b.id, "resource": "compute", "amount": 1.0, "expected_utility": 1.0},
        )
    with pytest.raises(KeyError, match="unknown room id"):
        negotiation.submit_bid(
            "missing",
            {"resource": "compute", "amount": 1.0, "expected_utility": 1.0},
        )
    with pytest.raises(TypeError, match="resource must be a string"):
        negotiation.allocate(1, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resource must be non-empty"):
        negotiation.allocate("", 1.0)
    with pytest.raises(ValueError, match="total_amount must be >= 0"):
        negotiation.allocate("compute", -1.0)
    with pytest.raises(ValueError, match="default_cap must be >= 0"):
        negotiation.allocate("compute", 1.0, default_cap=-1.0)

    empty_negotiation = ResourceNegotiation([make_room()])
    assert empty_negotiation.allocate("compute", 0.0) == {}

    zero_score = ResourceNegotiation([room_a, room_b])
    zero_score.submit_bid(
        room_a.id,
        {"room_id": room_a.id, "resource": "shared", "amount": 10.0, "expected_utility": 0.0},
    )
    zero_score.submit_bid(
        room_b.id,
        {"room_id": room_b.id, "resource": "shared", "amount": 10.0, "expected_utility": 0.0},
    )
    allocations = zero_score.allocate("shared", 6.0)
    assert allocations[room_a.id] == pytest.approx(3.0)
    assert allocations[room_b.id] == pytest.approx(3.0)

    tiny_cap = ResourceNegotiation([make_room()])
    tiny_room = next(iter(tiny_cap._rooms.values()))
    tiny_cap.submit_bid(
        tiny_room.id,
        {"room_id": tiny_room.id, "resource": "tiny", "amount": 1.0, "expected_utility": 1.0},
    )
    tiny_alloc = tiny_cap.allocate("tiny", 1.0, default_cap=1e-15)
    assert tiny_alloc[tiny_room.id] <= 1e-12


def test_compete_cooperate_and_form_coalition_cover_error_and_tie_paths():
    room_a = make_room([0.0, 0.0])
    room_b = make_room([0.0, 0.0])
    winner = compete(room_a, room_b, lambda room: 1.0)
    assert winner == min(room_a.id, room_b.id)
    assert compete(room_a, room_b, lambda room: 2.0 if room is room_a else 1.0) == room_a.id

    with pytest.raises(ValueError, match="rooms must be non-empty"):
        cooperate([], lambda rooms: {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return a dict"):
        cooperate([room_a], lambda rooms: 1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="rooms must be non-empty"):
        form_coalition([], "none")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inadmissible rooms"):
        form_coalition(
            [make_room([9.0, 9.0])],
            "blocked",
            permission_gate=GenesisPermission(theta=1.0),
        )
    coalition = form_coalition(
        [room_a, room_b],
        "allowed",
        permission_gate=GenesisPermission(theta=10.0),
    )
    assert coalition.name == "allowed"
