"""Example: QES applied to an engineering design-parameter search (non-market
domain), demonstrating the framework's domain-agnostic nature.

A room's state vector `x` represents a candidate structural design: the
width, height, and wall thickness of a hollow rectangular beam. We search
for the lightest design that still meets a minimum bending-stiffness
requirement, subject to manufacturing bounds enforced by the Genesis
permission kernel.

Run with:  python examples/design_space_search.py
"""
from __future__ import annotations

import numpy as np

from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace

# Target: minimize mass (~ cross-sectional area) while keeping the second
# moment of area I above a minimum stiffness requirement.
MIN_STIFFNESS_I = 2.0e-5  # m^4 (illustrative units)
MATERIAL_DENSITY = 2700.0  # kg/m^3 (aluminum, illustrative)
BEAM_LENGTH = 1.0  # m


def second_moment_of_area(width: float, height: float, thickness: float) -> float:
    """I for a thin-walled hollow rectangular section (illustrative model)."""
    outer = (width * height**3) / 12.0
    inner_w = max(width - 2 * thickness, 0.0)
    inner_h = max(height - 2 * thickness, 0.0)
    inner = (inner_w * inner_h**3) / 12.0
    return outer - inner


def mass(width: float, height: float, thickness: float) -> float:
    area = width * height - max(width - 2 * thickness, 0.0) * max(height - 2 * thickness, 0.0)
    return MATERIAL_DENSITY * area * BEAM_LENGTH


def design_score(x: np.ndarray) -> float:
    """Lower is better: mass, penalized heavily if stiffness requirement fails."""
    width, height, thickness = x
    stiffness = second_moment_of_area(width, height, thickness)
    penalty = 0.0 if stiffness >= MIN_STIFFNESS_I else 1e6 * (MIN_STIFFNESS_I - stiffness)
    return mass(width, height, thickness) + penalty


def make_seed_room() -> Room:
    """Q_0: a conservative, likely over-built starting design."""
    design = np.array([0.08, 0.12, 0.010])  # width, height, thickness (m)
    return Room(
        x=design,
        x_star=design,
        lower=np.array([0.02, 0.02, 0.002]),
        upper=np.array([0.20, 0.20, 0.030]),
        activation=np.ones(3),
    )


def local_search_step(room: Room, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Hill-climb toward lower mass while respecting manufacturing bounds."""
    current_score = design_score(room.x)
    proposal = room.x + rng.normal(0.0, 0.004, size=room.dim)
    proposal = np.clip(proposal, room.lower, room.upper)
    return proposal if design_score(proposal) < current_score else room.x


def main() -> None:
    rng = np.random.default_rng(11)

    seed = make_seed_room()
    generator = RealityGenerator(rng=rng)
    candidates = generator.branch(seed, count=50, scale=0.02)

    space = QESSpace(
        permission_gate=GenesisPermission(theta=1.5),
        step_fn=lambda room, t, dt: local_search_step(room, t, dt, rng),
        dt=1.0,
    )
    space.spawn(candidates)

    print("QES ENGINEERING DESIGN SEARCH")
    print("=" * 40)
    for _ in range(30):
        space.step()

    best = min(space.active_rooms(), key=lambda r: design_score(r.x))
    width, height, thickness = best.x
    stiffness = second_moment_of_area(width, height, thickness)
    print(f"Active candidates : {len(space.active_rooms())}")
    print(f"Collapsed          : {len(space.collapsed_rooms())}")
    print(f"Best design (m)    : width={width:.4f}  height={height:.4f}  thickness={thickness:.4f}")
    print(f"Mass (kg)          : {mass(width, height, thickness):.3f}")
    print(f"Stiffness I (m^4)  : {stiffness:.6f}  (requirement: {MIN_STIFFNESS_I:.6f})")


if __name__ == "__main__":
    main()
