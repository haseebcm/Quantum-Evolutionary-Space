"""Phase 20 demo: plain-text QES observatory dashboard.

This demo intentionally stays honest about the implementation: it renders a
Unicode/plain-text dashboard over ordinary Python/numpy data and, when
available, folds in a few real objects from other QES modules.
"""
from __future__ import annotations

import sys
import time
import tracemalloc

import numpy as np

from qes.observatory import ObservatoryDashboard, build_snapshot_from_modules


def _build_knowledge_graph() -> object | None:
    try:
        from qes.knowledge_graph import KnowledgeGraph
    except ImportError:
        return None

    graph = KnowledgeGraph()
    reality = graph.add_node("reality", payload={"seed": 11, "family": "alpha"})
    observation = graph.add_node(
        "observation",
        payload={"note": "cluster R2-R6 stabilizes after compute rebalance"},
        parents=[reality.id],
    )
    equation = graph.add_node(
        "equation",
        payload={"expr": "utility - risk + 0.5*novelty"},
        parents=[observation.id],
    )
    pattern = graph.add_node(
        "pattern",
        payload={"theme": "risk-aware diversification"},
        parents=[equation.id],
        success=True,
    )
    result = graph.record_experiment(
        equation.id,
        {"fitness": 0.91, "iterations": 24},
        success=True,
        pattern_id=pattern.id,
    )
    graph.record_causal_hypothesis(equation.id, result.id, confidence=0.84)
    return graph


def _build_divergence() -> object:
    try:
        from qes.divergence import DSA
    except ImportError:
        return {"dr": 0.41, "health": -0.09, "state": "synthetic-placeholder"}

    dsa = DSA()
    result = dsa.update(
        x=np.array([0.15, -0.28, 0.42]),
        x_star=np.array([0.05, -0.10, 0.35]),
        dt=1.0,
        w=np.eye(3),
        baseline=0.15,
    )
    return result


def _build_permission() -> object:
    try:
        from qes.permission import GenesisPermission
    except ImportError:
        return {"admitted": True, "soft_permission": 0.97, "source": "synthetic-placeholder"}

    gate = GenesisPermission(theta=0.8, m_min=0.05)
    return gate.evaluate(
        x=np.array([0.15, -0.20, 0.30]),
        lower=-np.ones(3),
        upper=np.ones(3),
        w=np.array([1.0, 1.0, 1.0]),
    )


def _build_marketplace() -> object:
    try:
        from qes.marketplace import MarketBid, RealityAccount, RealityMarketplace
    except ImportError:
        return {
            "round_index": 0,
            "last_allocations": {
                "R1": {"compute": 28.0, "memory": 12.0, "energy": 8.0},
                "R2": {"compute": 35.0, "memory": 17.0, "energy": 9.0},
                "R3": {"compute": 20.0, "memory": 10.0, "energy": 6.0},
            },
            "source": "synthetic-placeholder",
        }

    marketplace = RealityMarketplace()
    marketplace.register(
        RealityAccount(
            "R1",
            compute_budget=50.0,
            memory_budget=24.0,
            energy_budget=12.0,
            expected_utility=0.88,
            novelty=0.32,
            risk=0.09,
            priority=1.2,
        )
    )
    marketplace.register(
        RealityAccount(
            "R2",
            compute_budget=55.0,
            memory_budget=28.0,
            energy_budget=14.0,
            expected_utility=0.93,
            novelty=0.45,
            risk=0.12,
            priority=1.3,
        )
    )
    marketplace.register(
        RealityAccount(
            "R3",
            compute_budget=40.0,
            memory_budget=18.0,
            energy_budget=10.0,
            expected_utility=0.71,
            novelty=0.28,
            risk=0.07,
            priority=1.0,
        )
    )
    marketplace.submit_bid(
        MarketBid(
            "R1",
            requested_resources={"compute": 30.0, "memory": 10.0, "energy": 4.0},
            uncertainty=0.33,
            information_gain=0.20,
        )
    )
    marketplace.submit_bid(
        MarketBid(
            "R2",
            requested_resources={"compute": 38.0, "memory": 14.0, "energy": 5.0},
            uncertainty=0.39,
            information_gain=0.26,
        )
    )
    marketplace.submit_bid(
        MarketBid(
            "R3",
            requested_resources={"compute": 24.0, "memory": 8.0, "energy": 3.5},
            uncertainty=0.25,
            information_gain=0.12,
        )
    )
    marketplace.clear_market({"compute": 83.0, "memory": 31.0, "energy": 12.5})
    return marketplace


def _build_digital_twin_error() -> object:
    try:
        from qes.digital_twin_loop import anomaly_detection, drift_detection
    except ImportError:
        return {
            "anomaly": {"flagged": False, "max_z_score": 1.4},
            "drift": {"flagged": False, "mean_shift_sigma": 1.2},
            "source": "synthetic-placeholder",
        }

    anomaly = anomaly_detection(
        observation=np.array([0.95, 1.02, 1.08]),
        expected=np.array([1.00, 0.98, 1.03]),
        uncertainty=np.array([0.05, 0.04, 0.06]),
        threshold=3.0,
    )
    drift = drift_detection(
        residuals=[0.02, 0.01, -0.01, 0.00, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03],
        window=5,
    )
    return {"anomaly": anomaly, "drift": drift}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tracemalloc.start()
    started = time.perf_counter()

    print("QES PHASE 20: OBSERVATORY")
    print("=" * 60)
    print("[setup] building synthetic reality genealogy and probing optional integrations...")

    reality_tree = {
        "ROOT": ["R1", "R2", "R3"],
        "R1": ["R4"],
        "R2": ["R5", "R6"],
        "R3": ["R7"],
        "R4": [],
        "R5": [],
        "R6": [],
        "R7": [],
    }

    snapshot = build_snapshot_from_modules(
        population=12_482,
        active_realities=4_281,
        compute_pct=83.4,
        convergence=0.73,
        risk=0.08,
        novelty=0.41,
        reality_tree=reality_tree,
        knowledge_graph=_build_knowledge_graph(),
        divergence=_build_divergence(),
        permission=_build_permission(),
        resource_allocation=_build_marketplace(),
        digital_twin_error=_build_digital_twin_error(),
        agent_interactions=[
            {"agents": ["exploration-1", "safety-2"], "shared_focus": "R2", "strength": 0.74},
            {"agents": ["resource-3", "prediction-4"], "shared_focus": "R6", "strength": 0.68},
        ],
        failures=[
            {"reality_id": "R5", "category": "resource_exhaustion", "severity": 0.42},
            {"reality_id": "R7", "category": "boundary_failure", "severity": 0.18},
        ],
    )

    dashboard = ObservatoryDashboard(snapshot)
    rendered = dashboard.render()

    print()
    print(rendered)

    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print()
    print("=" * 60)
    print("REAL MEASURED COST (ordinary CPU/RAM for a text dashboard)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  Honest note: this is a plain-text monitoring dashboard for classical")
    print("  computation and numpy-based search state, not a literal astronomical")
    print("  observatory and not literal quantum computing.")


if __name__ == "__main__":
    main()
