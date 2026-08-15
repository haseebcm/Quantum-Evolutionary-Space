"""Phase 6 demo: classical reality-to-reality communication inside QES.

This walkthrough wires a few `Room` objects into `CommunicationFabric`,
exchanges state/knowledge/pattern payloads, runs a compute negotiation round,
executes one competition and one cooperation step, and forms a coalition.
Everything measured below is ordinary CPU/RAM cost for Python objects and
numpy arrays communicating in memory -- not literal physical realities.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.communication import (
    CommunicationFabric,
    ResourceNegotiation,
    compete,
    cooperate,
    form_coalition,
)
from qes.events import EventLog
from qes.patterns import Pattern, PatternMemory
from qes.permission import GenesisPermission
from qes.room import Room


def make_room(
    label: str,
    x: list[float],
    knowledge: dict[str, object],
    pattern_payload: dict[str, object],
) -> Room:
    room = Room(
        x=np.asarray(x, dtype=float),
        x_star=np.ones(len(x), dtype=float) * 0.5,
        lower=-np.ones(len(x), dtype=float),
        upper=np.ones(len(x), dtype=float),
        activation=np.ones(len(x), dtype=float),
        compute={"compute": 0.0},
    )
    room.memory["knowledge"] = dict(knowledge)
    room.memory["patterns"] = [
        Pattern(intent="phase-6-demo", context={"room": label}, payload=dict(pattern_payload), margin=0.5)
    ]
    room.tag("label", label)
    return room


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 6: REALITY-TO-REALITY COMMUNICATION")
    print("=" * 60)

    room_a = make_room("alpha", [0.10, 0.25, 0.40], {"insight": "stable near target"}, {"bias": "exploit"})
    room_b = make_room("beta", [0.60, 0.35, 0.15], {"insight": "needs more compute"}, {"bias": "explore"})
    room_c = make_room("gamma", [0.20, 0.80, 0.30], {"insight": "good fallback branch"}, {"bias": "hedge"})

    events = EventLog()
    shared_patterns = PatternMemory()
    fabric = CommunicationFabric([room_a, room_b, room_c], event_log=events, pattern_memory=shared_patterns)

    fabric.exchange_state(room_a, room_b)
    print(f"[state]       {room_a.id} <-> {room_b.id}: "
          f"alpha saw beta.x={np.round(room_a.memory['received_state'][room_b.id]['x'], 3).tolist()}, "
          f"beta saw alpha.x={np.round(room_b.memory['received_state'][room_a.id]['x'], 3).tolist()}")

    fabric.exchange_knowledge(room_b, room_c)
    print(f"[knowledge]   {room_b.id} <-> {room_c.id}: "
          f"beta learned {room_b.memory['received_knowledge'][room_c.id]}, "
          f"gamma learned {room_c.memory['received_knowledge'][room_b.id]}")

    fabric.exchange_pattern(room_a, room_c)
    print(f"[patterns]    shared pattern ids now in alpha memory: "
          f"{[pattern.id for pattern in room_a.memory['patterns']]}")
    print(f"[patterns]    shared PatternMemory entries: "
          f"{[pattern.id for pattern in shared_patterns.all_patterns('phase-6-demo')]}")

    negotiation = ResourceNegotiation(
        [room_a, room_b, room_c],
        permission_gate=GenesisPermission(theta=2.0),
        event_log=events,
    )
    negotiation.submit_bid(
        room_a.id,
        {"room_id": room_a.id, "resource": "compute", "amount": 8.0, "expected_utility": 1.0},
    )
    negotiation.submit_bid(
        room_b.id,
        {"room_id": room_b.id, "resource": "compute", "amount": 8.0, "expected_utility": 2.0},
    )
    negotiation.submit_bid(
        room_c.id,
        {
            "room_id": room_c.id,
            "resource": "compute",
            "amount": 8.0,
            "expected_utility": 3.0,
            "max_allocation": 5.0,
        },
    )
    allocations = negotiation.allocate("compute", 12.0)
    print(f"[resources]   negotiated compute allocations: {allocations}")

    winner = compete(room_a, room_b, lambda room: float(np.sum(room.x)), event_log=events)
    print(f"[compete]     winner between {room_a.id} and {room_b.id}: {winner}")

    cooperative_payload = cooperate(
        [room_a, room_b, room_c],
        lambda rooms: {"mean_state": np.mean([room.x for room in rooms], axis=0).round(4).tolist()},
        event_log=events,
    )
    print(f"[cooperate]   cooperative aggregate payload: {cooperative_payload}")

    coalition = form_coalition(
        [room_a, room_c],
        "stability-coalition",
        goal={"target": "share good local minima"},
        permission_gate=GenesisPermission(theta=2.0),
        event_log=events,
    )
    print(f"[coalition]   formed {coalition.name} with members {coalition.member_ids}")
    print(f"[events]      recorded {len(events)} events across EXECUTE/MUTATE/SELECT/MERGE")

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  objects communicating: 3 Room objects + 1 in-memory fabric + 1 event log")
    print("  This is a classical in-process message/coordination demo: numpy "
          "arrays and dict payloads exchanged by Python objects in memory.")


if __name__ == "__main__":
    main()
