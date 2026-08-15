"""Phase 19 demo: reproducibility infrastructure with a real replay command."""
from __future__ import annotations

import json
import random
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from qes.reproducibility import (
    ExperimentRecorder,
    reproduce_experiment,
    save_artifact,
    verify_reproduction,
)

try:
    from qes.verification import Evidence, VerificationStageResult
except ImportError:  # pragma: no cover - demo fallback only.
    Evidence = None  # type: ignore[assignment]
    VerificationStageResult = None  # type: ignore[assignment]


def run_toy_experiment(
    configuration: dict[str, Any],
    random_seeds: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    """Small deterministic classical computation standing in for a QES run."""
    if isinstance(random_seeds, dict):
        rng = random_seeds.get("_numpy_rng")
        if rng is None:
            rng = np.random.default_rng(int(random_seeds["numpy"]))
        python_seed = int(random_seeds["python"])
    else:
        rng = np.random.default_rng(int(random_seeds[0]))
        python_seed = int(random_seeds[1])

    random.seed(python_seed)
    sample_count = int(configuration["draw_count"])
    offset = float(configuration["offset"])
    scale = float(configuration["scale"])
    samples = np.asarray(rng.normal(loc=offset, scale=scale, size=sample_count), dtype=float)
    return {
        "objective": float(samples.sum() + random.random()),
        "samples": samples.tolist(),
        "best_sample": float(samples.max()),
        "mean_sample": float(samples.mean()),
    }


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 19: REPRODUCIBILITY INFRASTRUCTURE")
    print("=" * 72)

    configuration = {"draw_count": 5, "offset": 0.2, "scale": 0.05}
    seeds = {"numpy": 12345, "python": 54321}
    artifact_path = Path(__file__).with_name("_reproducibility_demo_artifact.json")

    recorder = ExperimentRecorder()
    recorder.record_configuration(configuration)
    recorder.record_seed(seeds)
    recorder.record_population(
        {
            "candidate_ids": ["room-001", "room-002", "room-003"],
            "state_scores": np.array([0.42, 0.57, 0.61]),
        }
    )
    recorder.record_lineage_step(step="initial_population", source="synthetic_demo")
    recorder.record_lineage_step(step="selection", selected="room-003", criterion="highest_state_score")
    recorder.record_equation({"expression": "sum(samples) + random.random()", "role": "toy objective"})
    recorder.record_decision(
        "Used fixed NumPy and Python seeds",
        rationale="prove deterministic replay with one command",
    )
    recorder.record_resource_allocation(
        cpu_threads=1,
        budget_candidates=3,
        evaluations=1,
    )

    random.seed(seeds["python"])
    results = run_toy_experiment(configuration, seeds)
    recorder.record_result(results)

    if Evidence is not None and VerificationStageResult is not None:
        stage = VerificationStageResult(stage="determinism", passed=True, message="seeded replay ready")
        recorder.record_evidence(
            Evidence(
                candidate="toy-demo",
                stages=(stage,),
                passed=True,
                summary="Deterministic classical replay prepared",
            )
        )
    else:
        recorder.record_evidence({"summary": "Deterministic classical replay prepared", "passed": True})

    artifact = recorder.finalize()
    save_artifact(artifact, artifact_path)

    print("[artifact tree]")
    print(json.dumps(artifact.to_dict(), indent=2))
    print()
    print(f"[saved]        {artifact_path}")
    reproduced = reproduce_experiment(artifact_path, run_toy_experiment)
    check = verify_reproduction(artifact, reproduced)
    print("[reproduction] one command: reproduce_experiment(saved_artifact, run_toy_experiment)")
    print(f"[reproduction] new experiment_id: {reproduced.experiment_id}")
    print(f"[verification] {check}")

    elapsed = time.perf_counter() - t_start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 72)
    print("REAL MEASURED COST (ordinary local CPU/RAM for classical computation)")
    print(f"  wall time            : {elapsed:.6f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("HONESTY STATEMENT: the hardware field above is local machine info from")
    print("  Python's platform module, and the replay is a seeded deterministic")
    print("  NumPy/Python computation inside this repo, not literal quantum hardware.")

    artifact_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
