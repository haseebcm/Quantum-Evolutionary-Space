"""Phase 22 demo: the QES SDK front door.

This example shows the SDK reducing a governed possibility-space search to a
few lines while still using the existing QES engine under the hood.
"""
from __future__ import annotations

import numpy as np

from qes.sdk import QESClient, SDKPermissionConfig


def sphere(x: np.ndarray) -> float:
    """Simple bounded objective used for the demo."""
    return float(np.dot(x, x))


def main() -> None:
    bounds = (-2.0 * np.ones(3), 2.0 * np.ones(3))
    client = QESClient(
        bounds,
        objective=sphere,
        population=24,
        branch_scale=0.35,
        permission=SDKPermissionConfig(theta=5.0),
        rng=42,
        intent="sdk-demo",
    )
    result = client.run(steps=20)

    print("QES SDK DEMO")
    print("=" * 40)
    print("Search type       : classical governed possibility-space search")
    print(f"Steps executed    : {result.steps}")
    print(f"Wall time (s)     : {result.wall_time_seconds:.6f}")
    print(f"Active rooms      : {result.active_rooms}")
    print(f"Collapsed rooms   : {result.collapsed_rooms}")
    print(f"Final entropy     : {result.final_entropy:.6f}")
    print(f"Final convergence : {result.final_convergence:.6f}")
    if result.best_score is not None:
        print(f"Best score        : {result.best_score:.6f}")
    else:
        print("Best score        : n/a")
    print(
        "Best state        : "
        + np.array2string(result.best_state, precision=6)
        if result.best_state is not None
        else "Best state        : n/a"
    )


if __name__ == "__main__":
    main()
