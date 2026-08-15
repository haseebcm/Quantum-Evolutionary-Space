"""Basic QES run: spawn a population of rooms from a seed reality, evolve
them under the Genesis permission kernel for several ticks, and print the
"living universe" telemetry (docs/QES-architecture.md, sections 1 and 19).

Run with:  python examples/basic_run.py
"""
from __future__ import annotations

import numpy as np

from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace


def make_seed_room(dim: int = 3) -> Room:
    """Q_0: the parent room representing the current observed state."""
    return Room(
        x=np.zeros(dim),
        x_star=np.zeros(dim),
        lower=-np.ones(dim) * 2.0,
        upper=np.ones(dim) * 2.0,
        activation=np.ones(dim),
    )


def mean_reverting_step(room: Room, t: float, dt: float) -> np.ndarray:
    """A simple stochastic mean-reverting dynamic used to evolve each room."""
    pull = -0.1 * (room.x - room.x_star)
    noise = np.random.default_rng(hash(room.id) % (2**32)).normal(0.0, 0.05, size=room.dim)
    return room.x + dt * pull + noise


def main() -> None:
    rng = np.random.default_rng(42)

    seed = make_seed_room(dim=3)
    generator = RealityGenerator(rng=rng)
    children = generator.branch(seed, count=200, scale=0.3)

    space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=mean_reverting_step,
        dt=1.0,
    )
    space.spawn(children)

    print("QUANTUM COMPUTE SPACE")
    print("=" * 40)
    for _step_index in range(10):
        telemetry = space.step()
        print(
            f"t={telemetry.time:5.1f}  "
            f"active={telemetry.active:4d}  "
            f"collapsed={telemetry.collapsed:4d}  "
            f"entropy={telemetry.entropy:6.3f}  "
            f"convergence={telemetry.convergence:5.3f}  "
            f"dominant={telemetry.dominant_room_id}"
        )

    dominant = space.dominant_room()
    if dominant is not None:
        print("\nDominant reality:", dominant)
        print("  state          :", dominant.x)
        print("  permission (pi):", dominant.weight)


if __name__ == "__main__":
    main()
