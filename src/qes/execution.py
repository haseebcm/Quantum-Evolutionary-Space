"""Cooperative execution budgets for trusted, bounded objective callbacks."""
from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from time import monotonic

import numpy as np


class ExecutionStopped(RuntimeError):
    """A call was refused because its run reached a budget or was cancelled."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class ExecutionBudget:
    """Count every objective call and check deadlines before new work.

    Deadlines are cooperative: an already-running callback is not interrupted.
    Evaluation limits apply across repeated SDK runs for one client.
    """

    def __init__(self, max_evaluations: int | None = None):
        if max_evaluations is not None and (
            not isinstance(max_evaluations, int) or isinstance(max_evaluations, bool) or max_evaluations < 0
        ):
            raise ValueError("max_evaluations must be a nonnegative integer or None")
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.deadline: float | None = None
        self.cancelled = False
        self._lock = threading.Lock()

    def __deepcopy__(self, memo: dict) -> ExecutionBudget:
        cloned = ExecutionBudget(self.max_evaluations)
        memo[id(self)] = cloned
        with self._lock:
            cloned.evaluations = self.evaluations
            cloned.deadline = self.deadline
            cloned.cancelled = self.cancelled
        return cloned

    def start(self, max_wall_time: float | None = None) -> None:
        if max_wall_time is not None and (not np.isfinite(max_wall_time) or max_wall_time < 0):
            raise ValueError("max_wall_time must be finite and nonnegative")
        self.deadline = None if max_wall_time is None else monotonic() + max_wall_time

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True

    def check(self) -> None:
        if self.cancelled:
            raise ExecutionStopped("cancelled")
        if self.deadline is not None and monotonic() >= self.deadline:
            raise ExecutionStopped("max_wall_time")

    def reserve_evaluation(self) -> None:
        with self._lock:
            self.check()
            if self.max_evaluations is not None and self.evaluations >= self.max_evaluations:
                raise ExecutionStopped("max_evaluations")
            self.evaluations += 1


class CountedObjective:
    """Cloneable wrapper retaining successful finite results without extra calls."""

    def __init__(self, objective: Callable[[np.ndarray], float], budget: ExecutionBudget):
        self.objective = objective
        self.budget = budget

    def __call__(self, x: np.ndarray) -> float:
        self.budget.reserve_evaluation()
        value = float(self.objective(x))
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    def __deepcopy__(self, memo: dict) -> CountedObjective:
        cloned = CountedObjective(copy.deepcopy(self.objective, memo), copy.deepcopy(self.budget, memo))
        memo[id(self)] = cloned
        return cloned
