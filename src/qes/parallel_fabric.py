"""Unified-33 Parallel Compute Fabric.

Source-derived domains (see the PDF-extraction summary, "Parallel Compute
Fabric"): the compute environment beneath QES is organized into five
domains, each with named engines:

    Parallel Compute Domain  -- MetaGPI Core, Thread Pool Generator,
                                 Matrix Multiplex Engine
    Synchronization Domain   -- Thread Synchronizer, CPU-GPI Link Engine,
                                 Temporal Match Filter
    Prediction Domain        -- Forecast Engine, Pattern Trajectory Mapper,
                                 Probability Drift Layer
    Execution Domain         -- Instruction Executor, Branch Prediction
                                 Engine, Output Normalizer
    Monitoring Domain         -- Metrics Dashboard, Stability Probe, Drift
                                 Monitor
    Error Correction Domain   -- ECC Engine, H^11 Tri-Cycle Corrector,
                                 Integrity Validator

`qes.space.QESSpace` already provides the room-level generate/execute loop
with an optional thread pool; this module provides the lower-level,
domain-agnostic primitives those named engines describe, usable directly or
as building blocks for a custom QESSpace `step_fn`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def _validate_non_negative_series(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_same_shape_arrays(values: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
    arrays = [np.asarray(value, dtype=float) for value in values]
    if not arrays:
        raise ValueError(f"{name} must be non-empty")
    first_shape = arrays[0].shape
    for array in arrays:
        if array.shape != first_shape:
            raise ValueError(f"all {name} must share the same shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    return arrays


class ParallelComputeDomain:
    """MetaGPI Core + Thread Pool Generator + Matrix Multiplex Engine."""

    def __init__(self, max_workers: int | None = None) -> None:
        if max_workers is not None:
            if not isinstance(max_workers, int) or isinstance(max_workers, bool):
                raise TypeError("max_workers must be an integer or None")
            if max_workers <= 0:
                raise ValueError("max_workers must be > 0")
        self.max_workers = max_workers

    def map(self, fn: Callable[[InputT], OutputT], items: Sequence[InputT]) -> list[OutputT]:
        """Thread Pool Generator: fan `fn` out across `items`."""
        if not callable(fn):
            raise TypeError("fn must be callable")
        materialized_items = list(items)
        if self.max_workers is None or len(materialized_items) <= 1:
            return [fn(item) for item in materialized_items]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(fn, materialized_items))

    @staticmethod
    def matrix_multiplex(
        matrices: Sequence[np.ndarray],
        vectors: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Matrix Multiplex Engine: batched matrix-vector multiplication,
        one (matrix, vector) pair per room/task."""
        matrix_list = [np.asarray(matrix, dtype=float) for matrix in matrices]
        vector_list = [np.asarray(vector, dtype=float) for vector in vectors]
        if len(matrix_list) != len(vector_list):
            raise ValueError("matrices and vectors must have the same length")
        if not matrix_list:
            return []

        for matrix, vector in zip(matrix_list, vector_list, strict=True):
            if matrix.ndim != 2:
                raise ValueError("each matrix must be two-dimensional")
            if vector.ndim != 1:
                raise ValueError("each vector must be one-dimensional")
            if matrix.shape[1] != vector.shape[0]:
                raise ValueError(
                    f"matrix/vector shape mismatch: {matrix.shape} cannot multiply {vector.shape}"
                )
            if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
                raise ValueError("matrices and vectors must contain only finite values")

        matrix_shapes = {matrix.shape for matrix in matrix_list}
        vector_shapes = {vector.shape for vector in vector_list}
        if len(matrix_shapes) == 1 and len(vector_shapes) == 1:
            stacked_matrices = np.stack(matrix_list, axis=0)
            stacked_vectors = np.stack(vector_list, axis=0)
            return [result for result in np.einsum("nij,nj->ni", stacked_matrices, stacked_vectors)]

        return [matrix @ vector for matrix, vector in zip(matrix_list, vector_list, strict=True)]


class SynchronizationDomain:
    """Thread Synchronizer + CPU-GPI Link Engine + Temporal Match Filter."""

    @staticmethod
    def barrier(results: Sequence[Any]) -> list[Any]:
        """Thread Synchronizer: materialize all results before proceeding
        (a no-op barrier for already-completed values; documents the sync
        point where parallel work must rejoin before the next stage)."""
        return list(results)

    @staticmethod
    def temporal_match(
        series_a: Sequence[float],
        series_b: Sequence[float],
        tolerance: float,
    ) -> list[int]:
        """Temporal Match Filter: indices where two time-aligned series
        agree within `tolerance`."""
        a = _validate_non_negative_series(series_a, "series_a")
        b = _validate_non_negative_series(series_b, "series_b")
        tolerance = float(tolerance)
        if a.shape != b.shape:
            raise ValueError("series_a and series_b must have the same shape")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and >= 0")
        return list(np.flatnonzero(np.abs(a - b) <= tolerance))


class PredictionDomain:
    """Forecast Engine + Pattern Trajectory Mapper + Probability Drift Layer."""

    @staticmethod
    def forecast_next(series: Sequence[float]) -> float:
        """Forecast Engine: linear (first-difference) extrapolation of the
        next value in a time series."""
        arr = _validate_non_negative_series(series, "series")
        if arr.size == 0:
            return 0.0
        if arr.size == 1:
            return float(arr[0])
        return float(arr[-1] + (arr[-1] - arr[-2]))

    @staticmethod
    def trajectory(series: Sequence[float], horizon: int) -> list[float]:
        """Pattern Trajectory Mapper: extrapolate `horizon` future points by
        repeated linear forecasting."""
        arr = _validate_non_negative_series(series, "series")
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise TypeError("horizon must be an integer")
        if horizon < 0:
            raise ValueError("horizon must be >= 0")
        if horizon == 0:
            return []
        if arr.size == 0:
            return [0.0] * horizon
        if arr.size == 1:
            return [float(arr[0])] * horizon

        step = arr[-1] - arr[-2]
        future = arr[-1] + step * np.arange(1, horizon + 1, dtype=float)
        return future.astype(float).tolist()

    @staticmethod
    def probability_drift(weights: Sequence[float], discount: float = 0.9) -> np.ndarray:
        """Probability Drift Layer: exponentially discount older weights and
        renormalize -- models belief drifting toward recent evidence."""
        w = _validate_non_negative_series(weights, "weights")
        if np.any(w < 0):
            raise ValueError("weights must be >= 0")
        discount = float(discount)
        if not np.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError("discount must be finite and in [0, 1]")
        if w.size == 0:
            return np.asarray([], dtype=float)
        decay = np.power(discount, np.arange(w.shape[0] - 1, -1, -1, dtype=float))
        drifted = w * decay
        total = float(drifted.sum())
        return drifted / total if total > 0 else drifted


@dataclass
class ExecutionResult:
    """Output of `ExecutionDomain.execute()`."""

    output: np.ndarray
    branch: int


class ExecutionDomain:
    """Instruction Executor + Branch Prediction Engine + Output Normalizer."""

    @staticmethod
    def execute_branches(
        branches: Sequence[Callable[[], np.ndarray]],
        score_fn: Callable[[np.ndarray], float],
    ) -> ExecutionResult:
        """Branch Prediction Engine: evaluate every branch, pick the one
        `score_fn` ranks best, then normalize its output (Output
        Normalizer: unit-sum if non-negative, else no-op)."""
        branch_list = list(branches)
        if not branch_list:
            raise ValueError("branches must be non-empty")
        if not callable(score_fn):
            raise TypeError("score_fn must be callable")

        outputs: list[np.ndarray] = []
        scores: list[float] = []
        for branch in branch_list:
            if not callable(branch):
                raise TypeError("branches must contain callables")
            output = np.asarray(branch(), dtype=float)
            if output.ndim != 1:
                raise ValueError("branch outputs must be one-dimensional")
            if not np.all(np.isfinite(output)):
                raise ValueError("branch outputs must contain only finite values")
            score = float(score_fn(output))
            if not np.isfinite(score):
                raise ValueError("score_fn must return finite scores")
            outputs.append(output)
            scores.append(score)

        best_idx = int(np.argmax(scores))
        best = outputs[best_idx]
        total = float(best.sum())
        normalized = best / total if total > 0 and np.all(best >= 0) else best
        return ExecutionResult(output=normalized, branch=best_idx)


class MonitoringDomain:
    """Metrics Dashboard + Stability Probe + Drift Monitor."""

    @staticmethod
    def dashboard(series: Sequence[float]) -> dict[str, float]:
        """Metrics Dashboard: summary statistics of a metric series."""
        arr = _validate_non_negative_series(series, "series")
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def stability_probe(series: Sequence[float], threshold: float) -> bool:
        """Stability Probe: True iff the series' standard deviation stays
        below `threshold` (a coarse "is this metric behaving?" check)."""
        arr = _validate_non_negative_series(series, "series")
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and >= 0")
        return bool(arr.size == 0 or np.std(arr) <= threshold)

    @staticmethod
    def drift_monitor(baseline: float, current: float, threshold: float) -> bool:
        """Drift Monitor: True iff `current` has drifted from `baseline` by
        more than `threshold` (absolute)."""
        baseline = float(baseline)
        current = float(current)
        threshold = float(threshold)
        if not np.isfinite(baseline) or not np.isfinite(current):
            raise ValueError("baseline and current must be finite")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and >= 0")
        return abs(current - baseline) > threshold


class ErrorCorrectionDomain:
    """ECC Engine + H^11 Tri-Cycle Corrector + Integrity Validator."""

    @staticmethod
    def ecc_correct(replicas: Sequence[np.ndarray]) -> np.ndarray:
        """ECC Engine: elementwise majority-style correction via the
        componentwise median across redundant replicas of the same state."""
        replica_arrays = _validate_same_shape_arrays(replicas, "replicas")
        stacked = np.stack(replica_arrays, axis=0)
        return np.median(stacked, axis=0)

    @staticmethod
    def tri_cycle_correct(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """H^11 Tri-Cycle Corrector: three-pass correction cycle -- clip to
        the admissible envelope, damp toward center, clip again -- used to
        pull an out-of-envelope state back toward stability."""
        x_arr = np.asarray(x, dtype=float)
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
        if x_arr.shape != lower_arr.shape or x_arr.shape != upper_arr.shape:
            raise ValueError("x, lower, and upper must have matching shapes")
        if not (
            np.all(np.isfinite(x_arr))
            and np.all(np.isfinite(lower_arr))
            and np.all(np.isfinite(upper_arr))
        ):
            raise ValueError("x, lower, and upper must contain only finite values")
        if np.any(lower_arr > upper_arr):
            raise ValueError("lower must be <= upper elementwise")

        clipped = np.clip(x_arr, lower_arr, upper_arr)
        center = (lower_arr + upper_arr) / 2.0
        damped = clipped + 0.5 * (center - clipped)
        return np.clip(damped, lower_arr, upper_arr)

    @staticmethod
    def integrity_validate(replicas: Sequence[np.ndarray], tolerance: float = 1e-6) -> bool:
        """Integrity Validator: True iff all replicas of a state agree with
        the ECC-corrected consensus within `tolerance`."""
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and >= 0")
        replica_arrays = _validate_same_shape_arrays(replicas, "replicas")
        consensus = ErrorCorrectionDomain.ecc_correct(replica_arrays)
        stacked = np.stack(replica_arrays, axis=0)
        return bool(np.all(np.abs(stacked - consensus) <= tolerance))
