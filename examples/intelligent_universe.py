"""Example: QES as an intelligent virtual universe (docs/QES-architecture.md,
sections 33-36), demonstrating the containment hierarchy

    Q superset R_i superset A_j          (a world hosting an agent)
    Q superset R_i superset Q'_i         (a world hosting a nested universe)

A top-level `Universe` hosts two `World`s: one running a room population
(a `QESSpace` searching for a design, as in `design_space_search.py`) plus
a monitoring `Agent`, and another that itself hosts a fully independent
nested `Universe` -- illustrating the recursive containment chain
`Q^(0) superset Q^(1) superset Q^(2) superset ...`.

Run with:  python examples/intelligent_universe.py
"""
from __future__ import annotations

import numpy as np

from qes.agent import Agent
from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace
from qes.universe import Universe
from qes.world import World


def make_seed_room(dim: int = 2) -> Room:
    return Room(
        x=np.zeros(dim),
        x_star=np.ones(dim) * 0.5,
        lower=-np.ones(dim),
        upper=np.ones(dim),
        activation=np.ones(dim),
    )


def mean_reverting_step(room: Room, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    pull = -0.2 * (room.x - room.x_star)
    noise = rng.normal(0.0, 0.02, size=room.dim)
    return room.x + dt * pull + noise


def monitor_policy(agent: Agent, observation: dict, t: float) -> str:
    """A simple monitoring agent that logs the tick count into its memory."""
    agent.memory["last_tick"] = t
    agent.memory["ticks_seen"] = agent.memory.get("ticks_seen", 0) + 1
    return f"observed tick {t:.1f}"


def main() -> None:
    rng = np.random.default_rng(3)

    # --- World 1: hosts a room population plus a monitoring agent ---
    seed = make_seed_room()
    space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=lambda room, t, dt: mean_reverting_step(room, t, dt, rng),
        dt=1.0,
    )
    space.spawn(RealityGenerator(rng=rng).branch(seed, count=30, scale=0.2))
    world_with_rooms = World(name="design-space", space=space)
    world_with_rooms.add_agent(Agent(name="monitor", policy=monitor_policy))

    # --- World 2: hosts nothing of its own except a nested universe ---
    world_with_nested = World(name="sandbox")
    nested_universe = Universe(dt=1.0)
    nested_seed = make_seed_room(dim=1)
    nested_space = QESSpace(permission_gate=GenesisPermission(theta=1.0), dt=1.0)
    nested_space.spawn(RealityGenerator(rng=rng).branch(nested_seed, count=10, scale=0.1))
    nested_universe.add_world(World(name="inner-experiment", space=nested_space))

    universe = Universe(dt=1.0)
    universe.add_world(world_with_rooms)
    universe.add_world(world_with_nested)
    universe.spawn_nested_universe(world_with_nested.id, nested_universe)

    print("QES INTELLIGENT VIRTUAL UNIVERSE")
    print("=" * 40)
    for _ in range(5):
        universe.step(event="tick")

    telemetry = universe.telemetry()
    monitor = world_with_rooms.agents[next(iter(world_with_rooms.agents))]
    print(f"Universe time        : {telemetry.time}")
    print(f"Worlds (R)           : {telemetry.world_count}")
    print(f"Agents (A, all)      : {telemetry.agent_count}")
    print(f"Events logged (I(t)) : {telemetry.event_count}")
    print(f"Monitor agent memory : {monitor.memory}")
    print(f"Nested universe time : {nested_universe.time}  (Q' inside world '{world_with_nested.name}')")


if __name__ == "__main__":
    main()
