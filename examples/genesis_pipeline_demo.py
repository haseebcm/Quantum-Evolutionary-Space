"""Demonstrate the layered Genesis-Selection pipeline.

Run with:  python examples/genesis_pipeline_demo.py
"""
from __future__ import annotations

from qes.selection import GenesisSelectionPipeline


def main() -> None:
    candidates = [
        ("alpha", 1.0),
        ("alpha", 0.5),
        ("beta", 2.0),
        ("beta", 0.25),
        ("gamma", 3.0),
    ]

    pipeline = GenesisSelectionPipeline(
        gates=[
            lambda candidate: 0.0 if candidate[1] < 2.5 else 1.0,
            lambda candidate: 0.0 if candidate[1] < 1.5 else 1.0,
            lambda candidate: 0.0 if candidate[1] < 1.0 else 1.0,
        ],
        signature_fn=lambda candidate: candidate[0],
        score_fn=lambda candidate: candidate[1],
    )

    result = pipeline.run(candidates)
    print("Genesis pipeline demo")
    print("=" * 40)
    print("population history:", result.population_history)
    for signature, trace in sorted(result.kinds.items()):
        print(signature, "->", {
            "first_seen_layer": trace.first_seen_layer,
            "survivor": trace.survivor,
            "score": trace.score,
            "eliminated_at_layer": trace.eliminated_at_layer,
        })


if __name__ == "__main__":
    main()
