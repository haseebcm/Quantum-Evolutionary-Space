import numpy as np
import pytest

from qes.events import (
    DeterministicReplay,
    Event,
    EventLog,
    LineageGraph,
    ReplayReport,
    state_hash,
)


def test_event_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Event(kind="NOT_A_KIND")


def test_event_rejects_non_finite_timestamp():
    with pytest.raises(ValueError):
        Event(kind="SPAWN", timestamp=float("nan"))


def test_event_validates_payload_and_parent_ids_types():
    with pytest.raises(TypeError, match="payload must be a dict"):
        Event(kind="SPAWN", payload=["bad"])
    with pytest.raises(TypeError, match="parent_ids must be a list of strings"):
        Event(kind="SPAWN", parent_ids=["ok", 1])


def test_event_round_trips_through_dict():
    event = Event(kind="SPAWN", payload={"room_id": "R-00001"}, parent_ids=["EV-000001"])
    restored = Event.from_dict(event.to_dict())
    assert restored.id == event.id
    assert restored.kind == event.kind
    assert restored.payload == event.payload
    assert restored.parent_ids == event.parent_ids


def test_event_log_records_and_filters():
    log = EventLog()
    spawn = log.record("SPAWN", payload={"room_id": "R-1"})
    log.record("EXECUTE", payload={"room_id": "R-1"}, parent_ids=[spawn.id])
    log.record("SPAWN", payload={"room_id": "R-2"})

    assert len(log) == 3
    assert log.events is not log._events
    spawns = log.filter("SPAWN")
    assert len(spawns) == 2
    assert all(e.kind == "SPAWN" for e in spawns)


def test_event_log_filter_rejects_unknown_kind():
    with pytest.raises(ValueError):
        EventLog().filter("NOPE")


def test_event_log_round_trips_through_serialization():
    log = EventLog()
    log.record("SPAWN", payload={"room_id": "R-1"})
    log.record("COLLAPSE", payload={"room_id": "R-1"})
    records = log.to_list()

    restored = EventLog()
    restored.restore(records)
    assert [e.kind for e in restored] == ["SPAWN", "COLLAPSE"]
    assert restored.to_list() == records


def test_lineage_graph_tracks_ancestors_and_descendants():
    graph = LineageGraph()
    graph.add_node("R-1", kind="SPAWN")
    graph.add_node("R-2", parents=["R-1"], kind="MUTATE")
    graph.add_node("R-3", parents=["R-2"], kind="MUTATE")
    graph.add_node("R-4", parents=["R-1"], kind="MUTATE")

    assert graph.ancestors("R-3") == ["R-2", "R-1"]
    assert set(graph.descendants("R-1")) == {"R-2", "R-3", "R-4"}
    assert graph.ancestors("R-1") == []
    assert graph.descendants("R-3") == []


def test_lineage_graph_rejects_unknown_parent():
    graph = LineageGraph()
    with pytest.raises(KeyError):
        graph.add_node("R-2", parents=["R-missing"])


def test_lineage_graph_validates_node_id_and_deduplicates_traversals():
    graph = LineageGraph()
    with pytest.raises(ValueError, match="non-empty string"):
        graph.add_node("")

    graph.add_node("R-1", kind="SPAWN")
    graph.add_node("R-2", parents=["R-1"], kind="MUTATE")
    graph.add_node("R-3", parents=["R-1"], kind="MUTATE")
    graph.add_node("R-4", parents=["R-2", "R-3"], kind="MERGE")

    assert graph.ancestors("R-4") == ["R-2", "R-3", "R-1"]
    assert graph.descendants("R-1") == ["R-2", "R-3", "R-4"]


def test_lineage_graph_round_trips_through_dict():
    graph = LineageGraph()
    graph.add_node("R-1", kind="SPAWN", payload={"seed": True})
    graph.add_node("R-2", parents=["R-1"], kind="MUTATE", payload={"scale": 0.1})

    restored = LineageGraph.from_dict(graph.to_dict())
    assert restored.ancestors("R-2") == ["R-1"]
    assert restored.to_dict() == graph.to_dict()


def test_lineage_graph_from_dict_rejects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        LineageGraph.from_dict({"parents": {"R-1": ["R-2"], "R-2": ["R-1"]}})


def test_state_hash_is_deterministic_and_shape_sensitive():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    c = np.array([1.0, 2.0, 3.0, 4.0])
    assert state_hash(a) == state_hash(b)
    assert state_hash(a) != state_hash(c)


def test_deterministic_replay_verifies_matching_state():
    replay = DeterministicReplay()
    state = np.array([1.0, 2.0])
    replay.record("tick-0", state)
    assert replay.verify("tick-0", state.copy()) is True
    assert replay.verify("tick-0", state + 1.0) is False


def test_deterministic_replay_verify_unknown_step_raises():
    with pytest.raises(KeyError):
        DeterministicReplay().verify("tick-0", np.zeros(2))


def test_deterministic_replay_record_validates_id_and_preserves_order():
    replay = DeterministicReplay()

    with pytest.raises(ValueError, match="non-empty string"):
        replay.record("", np.zeros(1))

    replay.record("tick-0", np.array([1.0]))
    replay.record("tick-0", np.array([2.0]))
    replay.record("tick-1", np.array([3.0]))

    report = replay.verify_all({"tick-0": np.array([2.0])})
    assert report.matches
    assert report.mismatches == []


def test_deterministic_replay_verify_all_reports_every_mismatch():
    replay = DeterministicReplay()
    replay.record("tick-0", np.array([1.0]))
    replay.record("tick-1", np.array([2.0]))

    report = replay.verify_all({"tick-0": np.array([1.0]), "tick-1": np.array([99.0])})
    assert isinstance(report, ReplayReport)
    assert not report
    assert len(report.mismatches) == 1
    assert report.mismatches[0].step_id == "tick-1"


def test_deterministic_replay_verify_all_passes_when_everything_matches():
    replay = DeterministicReplay()
    replay.record("tick-0", np.array([1.0]))
    report = replay.verify_all({"tick-0": np.array([1.0])})
    assert bool(report) is True
    assert report.mismatches == []
