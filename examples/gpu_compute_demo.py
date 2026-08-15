# ruff: noqa: E402
"""Demonstrate optional GPU acceleration for classical QES workloads.

This script reports whether a real GPU runtime was detected in the current
process, then runs three numeric operations through :class:`qes.gpu_compute.
GPUComputeBackend`. On the development machine used in this session, the demo
was validated by running the CPU fallback path to completion. No real GPU
execution was observed in this session because neither CuPy nor CUDA-enabled
PyTorch was available here.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qes.gpu_compute import GPUComputeBackend, SimulatedGPUBackend, probe_gpu_runtime

_T = TypeVar("_T")


def _format_preview(values: np.ndarray, *, max_items: int = 4) -> str:
    flat = np.ravel(values)
    preview = flat[:max_items]
    return np.array2string(preview, precision=4, separator=", ")


def _time_call(label: str, func: Callable[[], _T], backend: GPUComputeBackend | SimulatedGPUBackend) -> _T:
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    print(f"{label}: path={backend.last_execution_path}, wall_time={elapsed:.6f}s")
    return result


def main() -> None:
    """Run the demo and print honest runtime/timing information."""
    report = probe_gpu_runtime()
    backend = GPUComputeBackend()

    print("QES GPU compute demo")
    print(f"GPU available: {report.is_available}")
    print(f"Runtime: {report.runtime_name or 'none'}")
    print(f"Detail: {report.detail}")
    if report.is_available:
        print("Timings below reflect real GPU execution plus transfer/synchronization overhead.")
    else:
        print("Timings below reflect CPU fallback execution only; no GPU was used on this machine.")

    rng = np.random.default_rng(7)
    states = rng.normal(size=(4096, 8))
    drift_matrix = rng.normal(scale=0.15, size=(8, 8))
    control_matrix = rng.normal(scale=0.2, size=(8, 3))
    controls = rng.normal(size=(4096, 3))
    reference_state = rng.normal(size=8)
    divergence_weights = np.linspace(0.5, 1.5, 8)
    sigma = np.linspace(0.8, 1.6, 8)
    convergence_weights = np.abs(rng.normal(size=(4096, 8))) + 0.1
    convergence_reference = np.full(8, 1.0 / 8.0)

    rk4_result = _time_call(
        "batched_rk4_step",
        lambda: backend.batched_rk4_step(states, drift_matrix, control_matrix, controls, 0.02),
        backend,
    )
    print(f"  output_shape={rk4_result.shape}, preview={_format_preview(rk4_result)}")

    divergence_result = _time_call(
        "population_divergence",
        lambda: backend.population_divergence(
            states,
            reference_state,
            divergence_weights,
            sigma=sigma,
            scale=1.25,
            baseline=0.5,
        ),
        backend,
    )
    print(
        "  scores_preview="
        f"{_format_preview(divergence_result.scores)}, "
        f"health_preview={_format_preview(divergence_result.health)}"
    )

    convergence_result = _time_call(
        "population_convergence",
        lambda: backend.population_convergence(
            convergence_weights,
            reference=convergence_reference,
        ),
        backend,
    )
    print(
        "  convergence_preview="
        f"{_format_preview(convergence_result.convergence_coefficient)}, "
        f"gini_preview={_format_preview(convergence_result.gini)}"
    )

    print()
    print("--- SimulatedGPUBackend (software simulation, NOT real GPU hardware) ---")
    print(
        "This backend never uses real GPU silicon. It partitions batches across "
        "real separate OS processes to model GPU-lane-style parallel dispatch, "
        "purely to exercise that execution shape when no CUDA/ROCm runtime is "
        "present. Numeric results below are identical to the CPU path above; "
        "only the execution strategy differs. Wall-clock timings here must "
        "never be reported as GPU performance numbers."
    )
    with SimulatedGPUBackend() as simulated_backend:
        sim_rk4_result = _time_call(
            "simulated.batched_rk4_step",
            lambda: simulated_backend.batched_rk4_step(states, drift_matrix, control_matrix, controls, 0.02),
            simulated_backend,
        )
        print(
            f"  output_shape={sim_rk4_result.shape}, "
            f"preview={_format_preview(sim_rk4_result)}, "
            f"lane_pids={simulated_backend.last_lane_pids}"
        )
        print(
            f"  distinct_real_processes_used={len(set(simulated_backend.last_lane_pids))} "
            f"of {simulated_backend.lanes} configured lanes"
        )


if __name__ == "__main__":
    main()
