"""Phase 9 demo: classical digital-twin closed-loop estimation in QES.

This walkthrough simulates a small 1D damped system, fuses two noisy sensors,
runs the Phase 9 `DigitalTwinLoop`, and prints model selection, prediction
interval, anomaly, drift, and calibration diagnostics. Everything measured
below is ordinary NumPy/Python estimation over synthetic data in memory.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from qes.digital_twin_loop import DigitalTwinLoop, Observation, StateEstimator
from qes.reality_generator import RealityGenerator


def damped_model(state: np.ndarray) -> np.ndarray:
    return 0.88 * np.asarray(state, dtype=float)


def biased_growth_model(state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=float) + 0.35


def true_system_step(state: np.ndarray, drive: float) -> np.ndarray:
    return 0.88 * np.asarray(state, dtype=float) + drive


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 9: DIGITAL TWIN 2.0 CLOSED LOOP")
    print("=" * 60)

    rng = np.random.default_rng(7)
    true_state = np.array([0.0], dtype=float)

    estimator = StateEstimator(
        initial_state=[0.0],
        initial_covariance=[[1.0]],
        transition_matrix=[[0.9]],
        process_covariance=[[0.02]],
        observation_covariance=[[0.04]],
    )
    loop = DigitalTwinLoop(
        state_estimator=estimator,
        reality_generator=RealityGenerator(np.random.default_rng(17)),
        models=[damped_model, biased_growth_model],
        reference_state=[0.0],
        reality_count=10,
        reality_scale=0.03,
        prediction_confidence=0.95,
        anomaly_threshold=3.0,
        drift_window=3,
        drift_mean_shift_threshold=1.5,
        drift_std_ratio_threshold=2.0,
    )

    anomaly_steps: list[int] = []
    drift_steps: list[int] = []

    for step in range(10):
        drive = 0.15 if step < 6 else 0.60
        true_state = true_system_step(true_state, drive)

        sensor_a = true_state + rng.normal(0.0, 0.15, size=1)
        sensor_b_bias = 0.0 if step < 6 else 0.45
        sensor_b = true_state + sensor_b_bias + rng.normal(0.0, 0.30, size=1)
        if step == 4:
            sensor_b = sensor_b + 4.0

        result = loop.observe(
            [
                Observation(
                    values=sensor_a,
                    step=step,
                    timestamp=float(step),
                    uncertainty=[0.15],
                    sensor_id="sensor-a",
                ),
                Observation(
                    values=sensor_b,
                    step=step,
                    timestamp=float(step) + 0.01,
                    uncertainty=[0.30],
                    sensor_id="sensor-b",
                ),
            ]
        )

        state_std = float(np.sqrt(result.estimated_covariance[0, 0]))
        interval_low = float(result.prediction_interval[0][0])
        interval_high = float(result.prediction_interval[1][0])
        calibration = (
            "n/a" if result.calibration_score is None else f"{result.calibration_score:.3f}"
        )
        model_name = result.selected_model_name or "not-enough-history-yet"

        print(
            f"[step {step:02d}] true={true_state[0]:6.3f} "
            f"fused={result.fused_observation.values[0]:6.3f} "
            f"state={result.estimated_state[0]:6.3f} +/- {state_std:5.3f} "
            f"pred={result.predicted_observation[0]:6.3f} "
            f"interval=[{interval_low:6.3f}, {interval_high:6.3f}]"
        )
        print(
            f"           model={model_name:<24} "
            f"anomaly={result.anomaly_report.flagged!s:<5} "
            f"drift={result.drift_report.flagged!s:<5} "
            f"calibration={calibration} "
            f"action={result.recommended_action}"
        )

        if result.anomaly_report.flagged:
            anomaly_steps.append(step)
            print(
                f"           anomaly detail: max z-score="
                f"{result.anomaly_report.max_z_score:.2f}"
            )
        if result.drift_report.flagged:
            drift_steps.append(step)
            print(
                f"           drift detail: mean_shift_sigma="
                f"{result.drift_report.mean_shift_sigma:.2f}, "
                f"std_ratio={result.drift_report.std_ratio:.2f}"
            )

    elapsed = time.perf_counter() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=" * 60)
    print(f"[summary]     anomaly steps: {anomaly_steps}")
    print(f"[summary]     drift steps:   {drift_steps}")
    print(
        f"[summary]     final model choice: "
        f"{loop.history[-1].selected_model_name or 'not-enough-history-yet'}"
    )
    print(
        f"[summary]     final calibration score: "
        f"{loop.history[-1].calibration_score:.3f}"
    )
    print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
    print(f"  wall time            : {elapsed:.4f} s")
    print(f"  peak Python heap     : {peak / 1024:.1f} KB")
    print("  simulated system     : 1D damped linear process with noisy synthetic sensors")
    print(
        "  This is ordinary classical numerical estimation over synthetic data in "
        "NumPy/Python, not literal control of a real physical system."
    )


if __name__ == "__main__":
    main()
