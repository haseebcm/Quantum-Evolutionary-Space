"""Phase 12 demo: persistent knowledge graph across a small run of QES.

Walks through the roadmap's causal chain -- Reality -> Observation ->
Equation -> Pattern -> Result -> Causal relation -- recording provenance,
success/failure labels, a strengthened causal hypothesis, and a
counterexample, then round-trips the whole graph through a JSON file on
disk. Everything below is plain Python dict/list bookkeeping and a JSON
file write; no different in kind from checkpointing any other structured
run log.
"""
from __future__ import annotations

import tempfile
import time
import tracemalloc
from pathlib import Path

from qes.knowledge_graph import KnowledgeGraph


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 12: PERSISTENT KNOWLEDGE GRAPH")
    print("=" * 60)

    graph = KnowledgeGraph()

    reality = graph.add_node(kind="reality", payload={"seed": 7, "bounds": [-1.0, 1.0]})
    observation = graph.add_node(
        kind="observation", payload={"note": "trajectory converges near x=0.4"}, parents=[reality.id]
    )
    equation = graph.add_node(
        kind="equation", payload={"expr": "x**2 - 0.4*x"}, parents=[observation.id]
    )
    pattern = graph.add_node(
        kind="pattern", payload={"bias": "exploit-local-minimum"}, parents=[equation.id]
    )
    result = graph.record_experiment(
        equation_id=equation.id,
        result_payload={"fitness": 0.92, "iterations": 40},
        success=True,
        pattern_id=pattern.id,
    )
    print(f"[provenance]  reality={reality.id} -> observation={observation.id} -> "
          f"equation={equation.id} -> pattern={pattern.id} -> result={result.id}")
    print(f"[ancestors]   result's ancestors (nearest first): {graph.ancestors(result.id)}")

    failed_pattern = graph.add_node(kind="pattern", payload={"bias": "explore-wide"})
    failed_result = graph.record_experiment(
        equation_id=equation.id, result_payload={"fitness": 0.05}, success=False, pattern_id=failed_pattern.id
    )
    print(f"[configs]     successful results: {[n.id for n in graph.successful_configurations('result')]}")
    print(f"[configs]     failed results:     {[n.id for n in graph.failed_configurations('result')]}")

    graph.record_causal_hypothesis(equation.id, result.id, confidence=0.5)
    updated = graph.strengthen_hypothesis(equation.id, result.id, delta_confidence=0.3)
    print(f"[causal]      hypothesis {equation.id} -> {result.id}: "
          f"confidence={updated.confidence:.2f}, evidence_count={updated.evidence_count}")

    graph.add_counterexample("exploit-local-minimum-always-wins", failed_result.id)
    print(f"[counterexample] disproving 'exploit-local-minimum-always-wins': "
          f"{[n.id for n in graph.counterexamples('exploit-local-minimum-always-wins')]}")

    graph.add_domain_relationship("finance", "engineering", "shares_bounds")
    print(f"[domains]     relationships involving 'finance': {graph.domain_relationships('finance')}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "knowledge_graph.json"
        graph.save(path)
        file_bytes = path.stat().st_size
        reloaded = KnowledgeGraph.load(path)
        print(f"[persistence] saved {len(graph)} nodes to {file_bytes} bytes of JSON, "
              f"reloaded {len(reloaded)} nodes, ancestors match: "
              f"{reloaded.ancestors(result.id) == graph.ancestors(result.id)}")

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print(f"  nodes in graph       : {len(graph)}")
    print("  This is an in-memory dict-of-dicts graph with JSON save/load, no "
          "different in kind from any other structured run-log checkpoint.")


if __name__ == "__main__":
    main()
