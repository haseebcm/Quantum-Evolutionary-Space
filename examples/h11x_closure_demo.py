"""Demonstrate the H^11X admissibility stack together with closure checks.

Run with:  python examples/h11x_closure_demo.py
"""
from __future__ import annotations

from qes.closure import CrossDomainVerifier, PermissionClosure, action_selection
from qes.h11x import H11X


def main() -> None:
    result = H11X().evaluate(
        n=1.0,
        stress=1.0,
        load_paths=0.8,
        energy_flow=0.7,
        temporal_stability=1.2,
        domain_checks={"material": True, "thermal": True, "structural": True},
        push_to_failure_fn=lambda: 120.0,
        recovery_check_fn=lambda value: value < 100.0,
        internal_state={"x": 2.0},
        derive_fn=lambda state: state["x"] * 10.0,
        reintegrate_fn=lambda state: {"x": state["x"] + 3.0},
        relax_fn=lambda state: {"x": state["x"] * 0.5},
        reform_fn=lambda state: {"x": state["x"] + 1.0},
    )

    print("H^11X demo")
    print("=" * 40)
    print("admitted:", result.admitted)
    print("geometry:", result.geometry)
    print("denied_at_layer:", result.denied_at_layer)
    print("correction:", result.correction)
    print("export:", result.export)

    permission = PermissionClosure(
        candidate_actions=[1, 2, 3, 4],
        constraint_fn=lambda x, u: u - 3.0,
    )
    print("permission closure:", permission.evaluate(None))

    chosen = action_selection(
        x=0,
        actions=[1, 2, 3],
        j_fn=lambda x, u: abs(u - 2.0),
        s_fn=lambda x: 0.0,
        transition_fn=lambda x, u: x + u,
        lam=1.0,
    )
    print("selected action:", chosen)

    verifier = CrossDomainVerifier(
        encoder=lambda y: y,
        decoder=lambda x: x,
        transition_fn=lambda x, u, xi: x + u,
        state_space_check=lambda x: x < 10,
        constraint_fn=lambda x, u: u - 2.0,
    )
    print("verification:", verifier.verify(y=3.0, u=2.0, xi=0.0))


if __name__ == "__main__":
    main()
