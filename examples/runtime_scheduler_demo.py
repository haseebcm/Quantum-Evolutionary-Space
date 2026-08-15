"""Production-oriented scheduler demo for QES workloads.

This example shows how a runtime can prioritize exploration jobs, persist their
results, and execute them through a small scheduler while keeping the actual QES
search logic domain-agnostic.
"""
from __future__ import annotations

import numpy as np

from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.runtime import ProductionRuntime
from qes.space import QESSpace


def make_seed_room(value: float = 0.0) -> Room:
    return Room(
        x=np.array([value, value]),
        x_star=np.array([0.0, 0.0]),
        lower=np.array([-2.0, -2.0]),
        upper=np.array([2.0, 2.0]),
        activation=np.ones(2),
    )


def evaluate_space(label: str, drift: float, room_count: int = 20) -> dict:
    rng = np.random.default_rng(42 + hash(label) % 1000)
    generator = RealityGenerator(rng=rng)
    seed = make_seed_room(value=drift)
    candidates = generator.branch(seed, count=room_count, scale=0.35)

    space = QESSpace(
        permission_gate=GenesisPermission(theta=1.5),
        dt=1.0,
        step_fn=lambda room, t, dt: room.x + np.array([0.03 * drift, -0.02 * drift]),
    )
    space.spawn(candidates)
    for _ in range(4):
        space.step()

    best = max(space.active_rooms(), key=lambda room: room.weight, default=None)
    report = {
        "label": label,
        "active": len(space.active_rooms()),
        "collapsed": len(space.collapsed_rooms()),
        "best_weight": float(best.weight) if best is not None else 0.0,
        "best_state": (best.x.tolist() if best is not None else []),
    }
    return report


def main() -> None:
    runtime = ProductionRuntime(max_workers=2)

    runtime.submit(
        "stability-search",
        lambda: evaluate_space("stability-search", drift=0.25, room_count=25),
        priority=5,
        metadata={"domain": "control"},
    )
    runtime.submit(
        "optimization-search",
        lambda: evaluate_space("optimization-search", drift=-0.45, room_count=30),
        priority=8,
        metadata={"domain": "design"},
    )
    runtime.submit(
        "recovery-search",
        lambda: evaluate_space("recovery-search", drift=0.75, room_count=18),
        priority=3,
        metadata={"domain": "recovery"},
    )

    print("QES production runtime scheduler")
    print("=" * 40)
    results = runtime.run_all()
    for job in results:
        payload = runtime.snapshot().get(job.job_id, {})
        result = payload.get("result", {})
        print(f"{job.name:20s} | priority={job.priority:2d} | status={job.status} | result={result}")


if __name__ == "__main__":
    main()
