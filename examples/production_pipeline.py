"""Production-like QES workflow.

This example shows how the framework can be used as a small service-style
execution loop: generate candidate rooms, run several search steps, apply a
permission gate, and emit a final best result.

Run with:  python examples/production_pipeline.py
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


def make_seed_room(dim: int = 3) -> Room:
    return Room(
        x=np.zeros(dim),
        x_star=np.zeros(dim),
        lower=-2.0 * np.ones(dim),
        upper=2.0 * np.ones(dim),
        activation=np.ones(dim),
    )


def mean_reverting_step(room: Room, t: float, dt: float) -> np.ndarray:
    pull = -0.15 * (room.x - room.x_star)
    noise = np.random.default_rng(abs(hash(room.id)) % (2**32)).normal(0.0, 0.03, size=room.dim)
    return room.x + dt * pull + noise


def monitor_policy(agent: Agent, observation: dict, t: float) -> str:
    agent.memory["tick"] = t
    agent.memory["seen"] = agent.memory.get("seen", 0) + 1
    return f"tick={t:.2f}"


def main() -> None:
    rng = np.random.default_rng(7)
    seed = make_seed_room(dim=3)
    generator = RealityGenerator(rng=rng)
    children = generator.branch(seed, count=80, scale=0.25)

    space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=mean_reverting_step,
        dt=1.0,
    )
    space.spawn(children)

    world = World(name="production-experiment", space=space)
    world.add_agent(Agent(name="observer", policy=monitor_policy))

    universe = Universe(dt=1.0)
    universe.add_world(world)

    print("PRODUCTION-LIKE QES PIPELINE")
    print("=" * 40)
    for tick in range(6):
        telemetry = universe.step(event=f"run-{tick}")
        print(
            f"tick={telemetry.time:3.0f} "
            f"active={telemetry.world_count} worlds "
            f"event_count={telemetry.event_count} "
            f"time={telemetry.time}"
        )

    best = space.dominant_room()
    if best is not None:
        print("best room:", best.id)
        print("best state:", best.x)
        print("best weight:", best.weight)


if __name__ == "__main__":
    main()
