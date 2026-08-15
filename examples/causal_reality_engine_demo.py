"""Phase 5 demo: classical causal analysis on a small QES room/state model.

Builds a causal DAG, runs one intervention, one counterfactual, a
perturbation-sensitivity scan, and a hidden-variable heuristic check over a
small synthetic sample. Everything measured below is ordinary CPU/RAM cost for
Python objects and numpy arrays; this is not literal causality discovery,
parallel universes, or quantum compute.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.causal_engine import (
    CausalGraph,
    analyze_causality,
    counterfactual,
    hidden_variable_hypotheses,
    intervene,
    perturbation_sensitivity,
)
from qes.events import EventLog
from qes.room import Room


def make_room() -> Room:
    room = Room(
        x=np.asarray([0.80, 1.10, 0.70, 0.50, 0.40], dtype=float),
        x_star=np.zeros(5, dtype=float),
        lower=-np.ones(5, dtype=float),
        upper=np.ones(5, dtype=float) * 2.0,
        activation=np.ones(5, dtype=float),
        compute={"cpu_budget": 1.0},
    )
    room.memory["causal_variables"] = {
        "effort": 0,
        "quality": 1,
        "adoption": 2,
        "cost": 3,
        "support_load": 4,
    }
    room.tag("label", "phase-5-demo")
    return room


def make_graph() -> CausalGraph:
    graph = CausalGraph()
    for name in ("effort", "quality", "adoption", "cost", "support_load"):
        graph.add_variable(name)
    graph.add_edge("effort", "quality", strength=0.9)
    graph.add_edge("quality", "adoption", strength=0.8)
    graph.add_edge("effort", "cost", strength=0.5)
    return graph


def score_room(room: Room) -> float:
    return float(1.7 * room.x[2] - 0.6 * room.x[3] - 0.3 * room.x[4])


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 5: CAUSAL REALITY ENGINE")
    print("=" * 60)

    room = make_room()
    graph = make_graph()
    events = EventLog()

    print(f"[graph]       variables: {graph.variables()}")
    print(f"[graph]       topological order: {graph.topological_order()}")
    print(f"[baseline]    room state: {np.round(room.x, 3).tolist()}, score={score_room(room):.4f}")

    intervention = intervene(graph, "effort", 1.40, room, score_room, event_log=events)
    print(
        f"[intervene]   do(effort=1.40) -> quality={intervention.intervened_state['quality']:.3f}, "
        f"adoption={intervention.intervened_state['adoption']:.3f}, "
        f"cost={intervention.intervened_state['cost']:.3f}, "
        f"score delta={intervention.outcome_delta:+.4f}"
    )

    cf = counterfactual(graph, "quality", 0.70, room, score_room, event_log=events)
    print(
        f"[counterfact] if quality had been 0.70 instead of {cf.actual_value:.2f}, "
        f"adoption would be {cf.counterfactual_state['adoption']:.3f}, "
        f"score delta={cf.outcome_delta:+.4f}"
    )

    sensitivities = perturbation_sensitivity(
        graph,
        room,
        score_room,
        scale=0.08,
        samples_per_variable=24,
        rng=np.random.default_rng(7),
        event_log=events,
    )
    print("[sensitivity] ranked mean |score delta| under small one-axis perturbations:")
    for result in sensitivities:
        print(
            f"              {result.variable:12s} "
            f"mean={result.mean_absolute_outcome_delta:.4f} "
            f"max={result.max_absolute_outcome_delta:.4f}"
        )

    latent_driver = np.linspace(-1.0, 1.0, 8)
    samples = []
    for value in latent_driver:
        samples.append(
            {
                "effort": float(0.80 + 0.30 * value),
                "quality": float(1.10 + 0.27 * value),
                "adoption": float(0.70 + 0.22 * value),
                "cost": float(0.50 + 0.75 * value),
                "support_load": float(0.40 + 0.78 * value),
            }
        )
    hypotheses = hidden_variable_hypotheses(graph, samples, correlation_threshold=0.95, event_log=events)
    if hypotheses:
        top = hypotheses[0]
        print(
            f"[hidden]      strongest latent hypothesis: {top.candidate_name} for "
            f"{top.variable_a}<->{top.variable_b}, correlation={top.observed_correlation:.3f}"
        )
    else:
        print("[hidden]      no strong unexplained correlation crossed the threshold")

    bundled = analyze_causality(
        graph,
        room,
        score_room,
        interventions=[("effort", 1.20)],
        counterfactual_queries=[("quality", 0.90)],
        sensitivity_scale=0.08,
        sensitivity_samples=8,
        correlation_samples=samples,
        correlation_threshold=0.95,
        rng=np.random.default_rng(11),
        event_log=events,
    )
    print(
        f"[bundle]      analyze_causality returned {len(bundled.interventions)} intervention, "
        f"{len(bundled.counterfactuals)} counterfactual, "
        f"{len(bundled.sensitivities)} sensitivities, "
        f"{len(bundled.hidden_variable_hypotheses)} hidden-variable hypotheses"
    )
    print(f"[events]      recorded {len(events)} events across EXECUTE/DIVERGE/MUTATE/SELECT/MERGE")

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print(f"  events recorded      : {len(events)}")
    print("  This is a classical DAG + numpy state propagation demo. It does not")
    print("  prove literal causality in the philosophical sense and it is not")
    print("  quantum computation or zero-cost compute; it is ordinary CPU/RAM work.")


if __name__ == "__main__":
    main()
