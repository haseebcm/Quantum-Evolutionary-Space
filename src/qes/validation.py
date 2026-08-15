"""Validation, Functional Safety, and Failure Recovery.

Source-derived components (see the PDF-extraction summary, "Digital Twin +
Virtual Engineering Stack" and "Omega-Loop"):

    Validation & Verification Framework
    Functional Safety System
    Failure Recovery Loop (Omega-Loop): error -> null -> regenerate -> restabilize

The virtual-engineering loop these support:

    Digital Twin -> Simulation -> Sandbox -> Stress Test -> Virtual
    Commission -> Validate -> Recover/Correct

This module provides the Validate/Recover tail of that loop; the earlier
stages are covered by `qes.digital_twin`, `qes.multi_reality.SimulationEngine`,
and `qes.reality_generator`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ValidationCheck:
    """A single named admissibility/safety check."""

    name: str
    predicate: Callable[[np.ndarray], bool]


@dataclass
class ValidationReport:
    """Result of running a `ValidationFramework` over one state."""

    passed: bool
    failures: list = field(default_factory=list)


class ValidationFramework:
    """V&V Framework: runs a battery of named checks against a state and
    reports which (if any) failed."""

    def __init__(self, checks: Sequence[ValidationCheck] | None = None):
        self.checks: list[ValidationCheck] = list(checks or [])

    def add_check(self, name: str, predicate: Callable[[np.ndarray], bool]) -> None:
        self.checks.append(ValidationCheck(name=name, predicate=predicate))

    def validate(self, x: np.ndarray) -> ValidationReport:
        failures = [c.name for c in self.checks if not c.predicate(x)]
        return ValidationReport(passed=not failures, failures=failures)


class FunctionalSafetySystem:
    """Functional Safety System: scores how far a state sits inside its
    safety envelope (1.0 = centered/safest, 0.0 = at or past the boundary)."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray, margin: float = 0.1):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"lower and upper must have the same shape, got {self.lower.shape} vs {self.upper.shape}"
            )
        if np.any(self.upper < self.lower):
            raise ValueError("upper must be >= lower elementwise")
        if margin < 0:
            raise ValueError("margin must be >= 0")
        self.margin = margin

    def safety_score(self, x: np.ndarray) -> float:
        """Fraction of the *safety margin* (envelope shrunk by `margin` on
        each side) still available, averaged across dimensions; 0 outside
        the safety-margined envelope."""
        x = np.asarray(x, dtype=float)
        span = self.upper - self.lower
        safe_lower = self.lower + self.margin * span
        safe_upper = self.upper - self.margin * span
        center = (safe_lower + safe_upper) / 2.0
        half_span = np.maximum(safe_upper - safe_lower, 1e-12) / 2.0
        distance_ratio = np.abs(x - center) / half_span
        per_dim_score = np.clip(1.0 - distance_ratio, 0.0, 1.0)
        return float(np.mean(per_dim_score))

    def is_safe(self, x: np.ndarray, threshold: float = 0.0) -> bool:
        return self.safety_score(x) > threshold


@dataclass
class RecoveryOutcome:
    """Result of one `FailureRecoveryLoop.recover()` cycle."""

    recovered_state: np.ndarray
    was_null: bool


class FailureRecoveryLoop:
    """Omega-Loop: error -> null -> regenerate -> restabilize.

    On a detected failure (an inadmissible / non-finite state), the state
    is first collapsed to a null baseline, then regenerated via a supplied
    `regenerate_fn`, and finally restabilized (clipped) back into the
    admissible envelope.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray, baseline: np.ndarray | None = None):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"lower and upper must have the same shape, got {self.lower.shape} vs {self.upper.shape}"
            )
        if np.any(self.upper < self.lower):
            raise ValueError("upper must be >= lower elementwise")
        self.baseline = (
            np.asarray(baseline, dtype=float)
            if baseline is not None
            else (self.lower + self.upper) / 2.0
        )
        if self.baseline.shape != self.lower.shape:
            raise ValueError(
                f"baseline must match lower/upper shape {self.lower.shape}, got {self.baseline.shape}"
            )

    def is_failed(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(not np.all(np.isfinite(x)) or np.any(x < self.lower) or np.any(x > self.upper))

    def recover(
        self, x: np.ndarray, regenerate_fn: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> RecoveryOutcome:
        """Run the Omega-Loop if `x` has failed; otherwise return `x` unchanged."""
        if not self.is_failed(x):
            return RecoveryOutcome(recovered_state=np.asarray(x, dtype=float), was_null=False)

        # error -> null
        null_state = self.baseline.copy()
        # -> regenerate
        regenerated = (
            np.asarray(regenerate_fn(null_state), dtype=float)
            if regenerate_fn is not None
            else null_state
        )
        # -> restabilize
        restabilized = np.clip(regenerated, self.lower, self.upper)
        return RecoveryOutcome(recovered_state=restabilized, was_null=True)


class IntegrityValidator:
    """Integrity Validator: cross-checks redundant replicas of the same
    computed state for consistency before it is trusted as output."""

    @staticmethod
    def consensus(replicas: Sequence[np.ndarray]) -> np.ndarray:
        """Componentwise median across replicas (robust to a minority of
        corrupted/outlier replicas)."""
        if not replicas:
            raise ValueError("replicas must be non-empty")
        stacked = np.stack([np.asarray(r, dtype=float) for r in replicas], axis=0)
        return np.median(stacked, axis=0)

    @staticmethod
    def is_consistent(replicas: Sequence[np.ndarray], tolerance: float = 1e-6) -> bool:
        """True iff every replica agrees with the consensus within `tolerance`."""
        consensus = IntegrityValidator.consensus(replicas)
        return all(
            np.allclose(np.asarray(r, dtype=float), consensus, atol=tolerance) for r in replicas
        )
