"""Example: scientific inference with QES.

This example treats a candidate parameter set as a QES room and searches for
parameter values that best fit a noisy observation model. The framework remains
fully generic: the room's state is simply a vector of model parameters.
"""
from __future__ import annotations

import numpy as np

from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace


def build_observations() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 5.0, 30)
    true_intercept = 2.5
    true_slope = 1.4
    y = true_intercept + true_slope * x + rng.normal(0.0, 0.2, size=x.shape)
    return x, y


def objective(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    intercept, slope = params
    prediction = intercept + slope * x
    residual = prediction - y
    return float(np.mean(residual ** 2))


def make_seed_room() -> Room:
    return Room(
        x=np.array([0.0, 0.0]),
        x_star=np.array([2.5, 1.4]),
        lower=np.array([-5.0, -5.0]),
        upper=np.array([10.0, 10.0]),
        activation=np.ones(2),
    )


def main() -> None:
    x_obs, y_obs = build_observations()
    rng = np.random.default_rng(5)
    seed = make_seed_room()
    generator = RealityGenerator(rng=rng)
    candidates = generator.branch(seed, count=40, scale=0.7)

    def step(room: Room, t: float, dt: float) -> np.ndarray:
        current_loss = objective(room.x, x_obs, y_obs)
        proposal = room.x + rng.normal(0.0, 0.25, size=room.dim)
        proposal = np.clip(proposal, room.lower, room.upper)
        proposal_loss = objective(proposal, x_obs, y_obs)
        return proposal if proposal_loss < current_loss else room.x

    space = QESSpace(
        permission_gate=GenesisPermission(theta=1.0),
        step_fn=step,
        dt=1.0,
    )
    space.spawn(candidates)

    for _ in range(30):
        space.step()

    best = min(space.active_rooms(), key=lambda room: objective(room.x, x_obs, y_obs), default=None)
    print("QES scientific inference")
    print("=" * 35)
    if best is None:
        print("No admissible room remained.")
        return
    print(f"Best intercept: {best.x[0]:.4f}")
    print(f"Best slope    : {best.x[1]:.4f}")
    print(f"MSE           : {objective(best.x, x_obs, y_obs):.6f}")
    print("True params   : (2.5, 1.4)")


if __name__ == "__main__":
    main()
