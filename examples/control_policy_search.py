"""Example: QES applied to a control-policy search problem (non-market domain).

Demonstrates that QES's rooms/permission/selection machinery is entirely
domain-agnostic: here a room's state vector `x` represents a candidate set
of PID controller gains `[Kp, Ki, Kd]` instead of a market quantity. The
Genesis permission kernel enforces that gains stay within a safe,
stability-preserving envelope; a local random-search step function nudges
each candidate toward lower closed-loop tracking error.

Run with:  python examples/control_policy_search.py
"""
from __future__ import annotations

import numpy as np

from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace

# A simple first-order plant: dx/dt = -a*x + b*u. We evaluate a candidate
# PID gain vector by simulating a short closed-loop step response and
# scoring it by tracking error (lower is better).
PLANT_A = 0.5
PLANT_B = 1.0
SETPOINT = 1.0
SIM_STEPS = 30
SIM_DT = 0.05


def simulate_tracking_error(gains: np.ndarray) -> float:
    """Roll out a short closed-loop simulation and return integrated |error|."""
    kp, ki, kd = gains
    x = 0.0
    integral = 0.0
    prev_error = SETPOINT - x
    total_error = 0.0
    for _ in range(SIM_STEPS):
        error = SETPOINT - x
        integral += error * SIM_DT
        derivative = (error - prev_error) / SIM_DT
        u = kp * error + ki * integral + kd * derivative
        x += SIM_DT * (-PLANT_A * x + PLANT_B * u)
        prev_error = error
        total_error += abs(error) * SIM_DT
    return total_error


def make_seed_room() -> Room:
    """Q_0: an initial, conservative PID gain guess."""
    gains = np.array([1.0, 0.1, 0.01])
    return Room(
        x=gains,
        x_star=gains,  # no fixed reference point -- we search freely within bounds
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([10.0, 5.0, 2.0]),
        activation=np.ones(3),
    )


def local_search_step(room: Room, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Propose a small random perturbation to the gains and keep it only if
    it improves tracking error (a simple hill-climb move)."""
    current_error = simulate_tracking_error(room.x)
    proposal = room.x + rng.normal(0.0, 0.15, size=room.dim)
    proposal = np.clip(proposal, room.lower, room.upper)
    proposal_error = simulate_tracking_error(proposal)
    return proposal if proposal_error < current_error else room.x


def main() -> None:
    rng = np.random.default_rng(7)

    seed = make_seed_room()
    generator = RealityGenerator(rng=rng)
    candidates = generator.branch(seed, count=40, scale=0.5)

    space = QESSpace(
        permission_gate=GenesisPermission(theta=1.5),
        step_fn=lambda room, t, dt: local_search_step(room, t, dt, rng),
        dt=1.0,
    )
    space.spawn(candidates)

    print("QES CONTROL POLICY SEARCH")
    print("=" * 40)
    for _ in range(25):
        space.step()

    best = min(space.active_rooms(), key=lambda r: simulate_tracking_error(r.x))
    kp, ki, kd = best.x
    print(f"Active candidates : {len(space.active_rooms())}")
    print(f"Collapsed          : {len(space.collapsed_rooms())}")
    print(f"Best gains         : Kp={kp:.3f}  Ki={ki:.3f}  Kd={kd:.3f}")
    print(f"Tracking error     : {simulate_tracking_error(best.x):.5f}")


if __name__ == "__main__":
    main()
