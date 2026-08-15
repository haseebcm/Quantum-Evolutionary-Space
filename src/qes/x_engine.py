"""H^11 X-Engine — constraint-driven generative intelligence stack.

Source-derived roles (see the PDF-extraction summary, "X-Engine components
that fit inside QES"):

    H^11 Genesis     -- permission layer            (qes.permission)
    ACROS V12-BIE    -- equation perfection/stabilization
    ACROS V13        -- adaptive logic amplification/evolution
    H^11 Apex-I      -- power/core executor
    H^11 GeoM        -- geometry engine
    H^11 PatternS    -- pattern library + generative rules  (qes.patterns)

An engineering object inside QES moves through:

    Intent -> Genesis -> Equation -> Evolution -> Execution -> Geometry -> Patterns

`x_engine_pipeline()` wires that chain together using the pieces already
implemented elsewhere in QES (Genesis permission, EquationForge, Pattern
Memory) plus the three new components defined here (V12-BIE, V13, Apex-I,
GeoM).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qes.equation_forge import Equation, EquationForge
from qes.patterns import Pattern, PatternMemory
from qes.permission import GenesisPermission


class AcrosV12BIE:
    """Equation perfection / stabilization: refines an equation's theta
    toward a lower-Phi, lower-CCI configuration via repeated local descent
    against a supplied scoring function (smaller is better)."""

    def __init__(self, forge: EquationForge, steps: int = 5, scale: float = 0.5):
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        if scale < 0:
            raise ValueError(f"scale must be >= 0, got {scale}")
        self.forge = forge
        self.steps = steps
        self.scale = scale

    def stabilize(self, equation: Equation, score_fn: Callable[[Equation], float]) -> Equation:
        """Hill-climb `equation`'s theta for `steps` mutations, keeping only
        improving candidates (source: "ACROS V12-BIE -- equation perfection")."""
        best = equation
        best_score = score_fn(best)
        for _ in range(self.steps):
            candidate = self.forge.mutate(best, scale=self.scale)
            candidate_score = score_fn(candidate)
            if candidate_score <= best_score:
                best, best_score = candidate, candidate_score
        return best


class AcrosV13:
    """Adaptive logic amplification/evolution:

        (A, w, Theta)* = argmin[Risk + Inconsistency + Instability]

    over a population of candidate configurations, re-run repeatedly so the
    selected configuration adapts as the population itself evolves."""

    def __init__(
        self,
        risk_fn: Callable[[Any], float],
        inconsistency_fn: Callable[[Any], float],
        instability_fn: Callable[[Any], float],
    ):
        self.risk_fn = risk_fn
        self.inconsistency_fn = inconsistency_fn
        self.instability_fn = instability_fn
        self.history: list[float] = []

    def score(self, candidate: Any) -> float:
        return (
            self.risk_fn(candidate)
            + self.inconsistency_fn(candidate)
            + self.instability_fn(candidate)
        )

    def select(self, candidates: list) -> Any:
        """Adaptive selection: pick the argmin, remembering the winning
        score so `amplify()` can detect ongoing improvement across calls."""
        if not candidates:
            raise ValueError("candidates must be non-empty")
        best = min(candidates, key=self.score)
        self.history.append(self.score(best))
        return best

    def amplifying(self) -> bool:
        """True iff the selected score has been strictly improving
        (decreasing) over the last two `select()` calls."""
        return len(self.history) >= 2 and self.history[-1] < self.history[-2]


class ApexI:
    """Power/core executor: runs a pipeline of callables in sequence,
    threading the output of each stage into the next, and recording a
    per-stage execution trace."""

    def __init__(self) -> None:
        self.trace: list[Any] = []

    def execute(self, stages: list[Callable[[Any], Any]], initial: Any) -> Any:
        value = initial
        for stage in stages:
            if not callable(stage):
                raise TypeError(f"every stage must be callable, got {type(stage).__name__}")
            value = stage(value)
            self.trace.append(value)
        return value


class GeoM:
    """H^11 geometry engine: geometric primitives over room/state
    geometry -- distance, projection, and bounding-volume operations
    shared across the domain/state-space layers."""

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
        """Weighted Euclidean distance between two state vectors."""
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        if a_arr.shape != b_arr.shape:
            raise ValueError(f"a and b must have the same shape, got {a_arr.shape} vs {b_arr.shape}")
        d = a_arr - b_arr
        if w is None:
            return float(np.sqrt(np.sum(d ** 2)))
        w_arr = np.asarray(w, dtype=float)
        if w_arr.shape != d.shape:
            raise ValueError(f"w must match a/b shape {d.shape}, got {w_arr.shape}")
        return float(np.sqrt(np.sum(w_arr * d ** 2)))

    @staticmethod
    def project(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Orthogonal projection of `x` onto the bounding box [lower, upper]."""
        return np.clip(np.asarray(x, dtype=float), lower, upper)

    @staticmethod
    def centroid(points: list) -> np.ndarray:
        """Geometric centroid of a set of state vectors."""
        if not points:
            raise ValueError("points must be non-empty")
        return np.mean(np.asarray(points, dtype=float), axis=0)

    @staticmethod
    def bounding_radius(points: list, center: np.ndarray | None = None) -> float:
        """Radius of the smallest ball (centered at `center`, or the
        centroid) enclosing every point -- a coarse geometric envelope size."""
        arr = np.asarray(points, dtype=float)
        c = np.asarray(center, dtype=float) if center is not None else GeoM.centroid(points)
        return float(np.max(np.sqrt(np.sum((arr - c) ** 2, axis=1))))


@dataclass
class XEnginePipelineResult:
    """Output of `x_engine_pipeline()`: the full Intent -> ... -> Patterns trace."""

    intent: str
    admitted: bool
    equation: Equation | None
    executed: Any
    geometry: dict = field(default_factory=dict)
    pattern: Pattern | None = None


def x_engine_pipeline(
    intent: str,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    theta: dict,
    permission: GenesisPermission,
    forge: EquationForge,
    score_fn,
    executor_stages: list,
    memory: PatternMemory | None = None,
    w: np.ndarray | None = None,
) -> XEnginePipelineResult:
    """Intent -> Genesis -> Equation -> Evolution -> Execution -> Geometry -> Patterns.

    Args:
        intent: the request/problem label driving this pipeline run.
        x, lower, upper: the state and its admissible envelope (Genesis check).
        theta: seed equation parameters.
        permission: the Genesis permission gate.
        forge: EquationForge used to seed/stabilize the equation.
        score_fn: callable(Equation) -> float, smaller is better (used by V12-BIE).
        executor_stages: list of callable(value) -> value run by Apex-I.
        memory: optional PatternMemory to store the resulting pattern into.
        w: optional per-component weight vector for the Genesis check.
    """
    w = w if w is not None else np.ones_like(np.asarray(x, dtype=float))
    result = permission.evaluate(x, lower, upper, w)
    if not result.admitted:
        return XEnginePipelineResult(intent=intent, admitted=False, equation=None, executed=None)

    equation = forge.seed(theta=theta)
    v12 = AcrosV12BIE(forge)
    equation = v12.stabilize(equation, score_fn)

    apex = ApexI()
    executed = apex.execute(executor_stages, equation)

    geometry = {
        "distance_to_reference": GeoM.distance(x, np.zeros_like(np.asarray(x, dtype=float))),
        "projected": GeoM.project(x, lower, upper),
    }

    pattern = None
    if memory is not None:
        pattern = Pattern(intent=intent, context={"theta": equation.theta}, payload=executed)
        memory.store(pattern)

    return XEnginePipelineResult(
        intent=intent,
        admitted=True,
        equation=equation,
        executed=executed,
        geometry=geometry,
        pattern=pattern,
    )
