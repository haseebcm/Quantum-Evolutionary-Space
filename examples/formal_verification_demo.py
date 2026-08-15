"""Phase 15 demo: staged evidence packages for candidate verification.

This walkthrough produces `Evidence(candidate)` rather than only a scalar
score. The checks below are classical numerical/logical diagnostics over
numpy state vectors, permission gates, invariants, and cross-domain closure
hooks -- not a theorem prover and not free compute.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.closure import CrossDomainVerifier
from qes.invariants import InvariantEngine
from qes.permission import GenesisPermission
from qes.room import Room
from qes.verification import Evidence, VerificationEngine


def make_room(label: str, x: list[float]) -> Room:
    room = Room(
        x=np.asarray(x, dtype=float),
        x_star=np.zeros(len(x), dtype=float),
        lower=-np.ones(len(x), dtype=float),
        upper=np.ones(len(x), dtype=float),
        activation=np.ones(len(x), dtype=float),
        compute={"cpu": 1.0},
    )
    room.tag("label", label)
    return room


def make_cross_domain_verifier() -> CrossDomainVerifier:
    return CrossDomainVerifier(
        encoder=lambda y: np.asarray(y, dtype=float),
        decoder=lambda x: x,
        transition_fn=lambda x, u, xi: x
        + np.asarray(u, dtype=float)
        + (np.zeros_like(x) if xi is None else np.asarray(xi, dtype=float)),
        state_space_check=lambda x: bool(np.all(np.abs(x) <= 1.5)),
        constraint_fn=lambda x, u: float(np.max(np.abs(np.asarray(u, dtype=float))) - 0.5),
    )


def print_evidence(label: str, evidence: Evidence) -> None:
    print(f"{label}: {evidence.summary} -> overall={'PASS' if evidence.passed else 'FAIL'}")
    for stage in evidence.stages:
        print(f"  [{stage.stage:<12}] {stage.status.upper():<7} {stage.message}")


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 15: FORMAL VERIFICATION LAYER")
    print("=" * 60)
    print("Framing: Evidence(candidate) instead of merely Score(candidate)")

    verifier = make_cross_domain_verifier()
    invariant_engine = InvariantEngine()
    permission_gate = GenesisPermission(theta=1.0)

    good_candidate = make_room("good", [0.25, -0.25])
    bad_candidate = make_room("bad", [1e-6, 0.2])

    good_engine = VerificationEngine(
        lower=-1.0,
        upper=1.0,
        permission_gate=permission_gate,
        invariant_engine=invariant_engine,
        cross_domain_verifier=verifier,
        domains=[{"name": "engineering", "action": np.array([0.1, 0.0]), "disturbance": np.zeros(2)}],
        evaluate_fn=lambda state: float(np.sum(state ** 2)),
        stability_threshold=10.0,
    )
    bad_engine = VerificationEngine(
        lower=-1.0,
        upper=1.0,
        permission_gate=permission_gate,
        invariant_engine=invariant_engine,
        cross_domain_verifier=verifier,
        domains=[{"name": "engineering", "action": np.array([0.1, 0.0]), "disturbance": np.zeros(2)}],
        evaluate_fn=lambda state: float(1.0 / state[0]),
        stability_epsilon=1e-6,
        stability_threshold=1e9,
    )

    good_evidence = good_engine.verify(good_candidate)
    bad_evidence = bad_engine.verify(bad_candidate)

    print_evidence("Evidence(good_candidate)", good_evidence)
    print("-" * 60)
    print_evidence("Evidence(bad_candidate)", bad_evidence)

    elapsed = time.perf_counter() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  verification model   : bounds checks + permission gate + invariant checks")
    print("                       + perturbation-based stability probe + cross-domain closure")
    print("  This is classical numerical/logical verification in Python/numpy,")
    print("  not a formal theorem-prover and not literal quantum computing.")


if __name__ == "__main__":
    main()
