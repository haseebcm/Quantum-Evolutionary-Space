"""Phase 16: classical adversarial candidate generation and red-team analysis.

This module generates deliberately stressful `Room` candidates aimed at
boundary, stability, coupling, numerical, and resource failure modes. It is a
classical synthetic stress-testing layer for governed numeric search -- not
literal quantum computing and not literal "reality breaking."
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from qes.permission import GenesisPermission, PermissionResult
from qes.room import Room

try:  # Optional Phase 15 dependency built by a concurrent session.
    import qes.verification as _verification
except ImportError:  # pragma: no cover - depends on concurrent workspace state
    _verification = None  # type: ignore[assignment]

try:
    from qes.equation_ast import Constant, EquationAST, Operator, Variable
except ImportError:  # pragma: no cover - defensive fallback
    Constant = None  # type: ignore[misc,assignment]
    EquationAST = None  # type: ignore[misc,assignment]
    Operator = None  # type: ignore[misc,assignment]
    Variable = None  # type: ignore[misc,assignment]

_FLOAT_EPS = np.finfo(float).eps


class FailureCategory(str, Enum):
    """Failure modes covered by the adversarial red-team layer."""

    BOUNDARY_FAILURE = "boundary_failure"
    UNSTABLE_STATE = "unstable_state"
    NUMERICAL_SINGULARITY = "numerical_singularity"
    ADVERSARIAL_INPUT = "adversarial_input"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PATHOLOGICAL_EQUATION = "pathological_equation"
    UNEXPECTED_COUPLING = "unexpected_coupling"
    CASCADING_FAILURE = "cascading_failure"


@dataclass
class AdversarialCandidate:
    """One generated red-team candidate compatible with the rest of QES."""

    room: Room
    category: FailureCategory
    rationale: str
    stress_score: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.room, Room):
            raise TypeError("room must be a Room")
        if not isinstance(self.category, FailureCategory):
            raise TypeError("category must be a FailureCategory")
        if not isinstance(self.rationale, str):
            raise TypeError("rationale must be a string")
        if not self.rationale.strip():
            raise ValueError("rationale must be non-empty")
        score = float(self.stress_score)
        if not np.isfinite(score):
            raise ValueError("stress_score must be finite")
        if score < 0.0:
            raise ValueError("stress_score must be >= 0")
        self.stress_score = score
        self.metadata.setdefault("designed_category", self.category.value)


@dataclass
class FailureAnalysis:
    """Outcome of evaluating one adversarial candidate."""

    candidate_id: str
    category: FailureCategory
    observed_category: FailureCategory | None
    failed: bool
    permission_admitted: bool
    evaluation_passed: bool
    severity: float
    failure_reason: str
    permission_result: PermissionResult | None
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class RedTeamReport:
    """Aggregate result of a red-team campaign."""

    analyses: list[FailureAnalysis]
    category_counts: dict[FailureCategory, int]
    category_failure_counts: dict[FailureCategory, int]
    category_failure_rates: dict[FailureCategory, float]
    suggested_improvements: list[str]

    @property
    def total_candidates(self) -> int:
        """Total number of candidates processed."""
        return len(self.analyses)

    @property
    def total_failures(self) -> int:
        """Total number of failed candidates."""
        return sum(1 for analysis in self.analyses if analysis.failed)

    @property
    def overall_failure_rate(self) -> float:
        """Failure fraction across the whole campaign."""
        if not self.analyses:
            return 0.0
        return self.total_failures / len(self.analyses)


class AdversarialGenerator:
    """Generate deterministic worst-case `Room` candidates from a template room."""

    def __init__(
        self,
        template_room: Room,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        if rng is not None and seed is not None:
            raise ValueError("pass rng or seed, not both")
        if not isinstance(template_room, Room):
            raise TypeError("template_room must be a Room")
        if template_room.dim < 1:
            raise ValueError("template_room must have at least one dimension")
        self.template_room = template_room
        self.rng = rng or np.random.default_rng(seed)

    def generate_boundary_failure(
        self, count: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate candidates that step just outside lower/upper bounds."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(count)
        span = self._span()
        midpoint = self._midpoint()
        candidates: list[AdversarialCandidate] = []
        for index in range(count):
            axis = index % self.template_room.dim
            direction = -1.0 if index % 2 == 0 else 1.0
            x = midpoint.copy()
            overshoot = span[axis] * (0.05 + 0.01 * index) + 1e-6
            x[axis] = (
                self.template_room.lower[axis] - overshoot
                if direction < 0.0
                else self.template_room.upper[axis] + overshoot
            )
            x += rng.normal(0.0, span * 0.01)
            room = self._clone_room(
                x=x,
                state="AdversarialBoundaryProbe",
                memory_updates={
                    "adversarial_profile": {
                        "axis": axis,
                        "direction": "lower" if direction < 0.0 else "upper",
                        "overshoot": float(overshoot),
                    }
                },
            )
            candidates.append(
                AdversarialCandidate(
                    room=room,
                    category=FailureCategory.BOUNDARY_FAILURE,
                    rationale=(
                        "Probe slight bound exceedances that can slip past "
                        "weak clipping or validation."
                    ),
                    stress_score=1.0 + index * 0.1,
                )
            )
        return candidates

    def generate_unstable_state(
        self, count: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate oscillatory, high-amplitude states likely to destabilize updates."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(count)
        span = self._span()
        candidates: list[AdversarialCandidate] = []
        base_pattern = np.where(np.arange(self.template_room.dim) % 2 == 0, 1.0, -1.0)
        for index in range(count):
            amplitude = 1.4 + 0.3 * index
            jitter = rng.normal(0.0, span * 0.03)
            x = self.template_room.x_star + amplitude * span * base_pattern + jitter
            room = self._clone_room(
                x=x,
                activation=np.roll(self.template_room.activation, index % self.template_room.dim),
                state="AdversarialUnstableProbe",
                memory_updates={"oscillation_frequency": 2 + index},
            )
            candidates.append(
                AdversarialCandidate(
                    room=room,
                    category=FailureCategory.UNSTABLE_STATE,
                    rationale="Probe oscillatory high-gain states that can amplify tiny perturbations.",
                    stress_score=1.4 + index * 0.2,
                )
            )
        return candidates

    def generate_numerical_singularity(
        self, count: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate near-zero denominator scenarios and other singular numeric states."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(count)
        midpoint = self._midpoint()
        candidates: list[AdversarialCandidate] = []
        for index in range(count):
            x = midpoint.copy()
            signed_epsilon = (1 if index % 2 == 0 else -1) * 10.0 ** (-(6 + index))
            x[0] = signed_epsilon
            if self.template_room.dim > 1:
                x[1:] += rng.normal(0.0, 0.01, size=self.template_room.dim - 1)
            equations = self._pathological_division_equation()
            room = self._clone_room(
                x=x,
                equations=equations,
                state="AdversarialSingularityProbe",
                gates={"near_zero_axis": 0},
                memory_updates={"singularity_epsilon": float(abs(signed_epsilon))},
            )
            candidates.append(
                AdversarialCandidate(
                    room=room,
                    category=FailureCategory.NUMERICAL_SINGULARITY,
                    rationale="Probe divisions, exponentials, or logs around near-singular values.",
                    stress_score=2.0 + index * 0.25,
                )
            )
        return candidates

    def generate_adversarial_input(
        self, count: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate deceptive but finite inputs that stress canonicalization logic."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(count)
        midpoint = self._midpoint()
        span = self._span()
        candidates: list[AdversarialCandidate] = []
        for index in range(count):
            noise = rng.normal(0.0, span * 0.005)
            pattern = np.where(np.arange(self.template_room.dim) % 2 == 0, 0.49, -0.49)
            x = midpoint + pattern * span + noise
            room = self._clone_room(
                x=x,
                activation=np.where(np.arange(self.template_room.dim) % 2 == 0, 1.0, 0.01),
                state="AdversarialInputProbe",
                memory_updates={
                    "conflicting_hints": ["maximize_gain", "minimize_gain"],
                    "duplicate_fields": {"priority": ["high", "low"]},
                },
            )
            candidates.append(
                AdversarialCandidate(
                    room=room,
                    category=FailureCategory.ADVERSARIAL_INPUT,
                    rationale="Probe ambiguous, conflicting inputs that remain numerically valid.",
                    stress_score=1.2 + index * 0.1,
                )
            )
        return candidates

    def generate_resource_exhaustion_probe(
        self, count: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate candidates that advertise unrealistic compute demands."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(count)
        midpoint = self._midpoint()
        span = self._span()
        candidates: list[AdversarialCandidate] = []
        for index in range(count):
            scale = 1.0 + 0.15 * index
            x = midpoint + rng.normal(0.0, span * 0.02)
            room = self._clone_room(
                x=x,
                compute={
                    **self.template_room.compute,
                    "compute": float(250_000.0 * scale),
                    "workers": float(512.0 * scale),
                    "memory_mb": float(131_072.0 * scale),
                },
                state="AdversarialResourceProbe",
                memory_updates={"synthetic_batch_size": int(50_000 * scale)},
            )
            candidates.append(
                AdversarialCandidate(
                    room=room,
                    category=FailureCategory.RESOURCE_EXHAUSTION,
                    rationale="Probe allocator and evaluator behavior under synthetic cost spikes.",
                    stress_score=2.5 + index * 0.3,
                )
            )
        return candidates

    def generate_pathological_equation(
        self, count: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate candidates carrying intentionally pathological numeric equations."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(count)
        midpoint = self._midpoint()
        candidates: list[AdversarialCandidate] = []
        for index in range(count):
            x = midpoint + rng.normal(0.0, 0.01, size=self.template_room.dim)
            equations = self._pathological_log_equation()
            room = self._clone_room(
                x=x,
                equations=equations,
                state="AdversarialEquationProbe",
                memory_updates={"pathological_pattern": "log(x0 - x0)"},
            )
            candidates.append(
                AdversarialCandidate(
                    room=room,
                    category=FailureCategory.PATHOLOGICAL_EQUATION,
                    rationale=(
                        "Probe evaluator handling of structurally valid but "
                        "numerically pathological equations."
                    ),
                    stress_score=1.8 + index * 0.15,
                )
            )
        return candidates

    def generate_unexpected_coupling(
        self, count: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate candidates whose coupling matrix amplifies small local violations."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(count)
        midpoint = self._midpoint()
        span = self._span()
        candidates: list[AdversarialCandidate] = []
        for index in range(count):
            x = midpoint.copy()
            axis = index % self.template_room.dim
            x[axis] = self.template_room.upper[axis] + span[axis] * 0.02
            x += rng.normal(0.0, span * 0.003)
            coupling = np.full((self.template_room.dim, self.template_room.dim), 1.5 + 0.5 * index)
            np.fill_diagonal(coupling, 1.0)
            room = self._clone_room(
                x=x,
                couplings=coupling,
                state="AdversarialCouplingProbe",
                memory_updates={"coupling_gain": float(1.5 + 0.5 * index)},
            )
            candidates.append(
                AdversarialCandidate(
                    room=room,
                    category=FailureCategory.UNEXPECTED_COUPLING,
                    rationale=(
                        "Probe cross-variable amplification that turns one local "
                        "issue into a system-wide stressor."
                    ),
                    stress_score=2.2 + index * 0.2,
                )
            )
        return candidates

    def generate_cascading_failure_scenario(
        self, count: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate multi-stage violation chains designed to cascade across dimensions."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(count)
        midpoint = self._midpoint()
        span = self._span()
        candidates: list[AdversarialCandidate] = []
        for index in range(count):
            x = midpoint.copy()
            breach_count = min(self.template_room.dim, max(2, 2 + index))
            for axis in range(breach_count):
                x[axis] = self.template_room.upper[axis] + span[axis] * (0.03 + 0.01 * axis)
            if breach_count < self.template_room.dim:
                x[breach_count:] += rng.normal(0.0, span[breach_count:] * 0.01)
            coupling = np.tril(
                np.ones((self.template_room.dim, self.template_room.dim))
            ) * (1.0 + 0.2 * index)
            room = self._clone_room(
                x=x,
                couplings=coupling,
                compute={**self.template_room.compute, "recovery_budget": float(8.0 - index)},
                state="AdversarialCascadeProbe",
                memory_updates={
                    "cascade_stages": [
                        "boundary breach",
                        "coupling amplification",
                        "recovery saturation",
                    ]
                },
            )
            candidates.append(
                AdversarialCandidate(
                    room=room,
                    category=FailureCategory.CASCADING_FAILURE,
                    rationale="Probe chained failures where one violation triggers broader collapse.",
                    stress_score=2.8 + index * 0.3,
                )
            )
        return candidates

    def generate_all(
        self, n_per_category: int = 1, seed: int | None = None
    ) -> list[AdversarialCandidate]:
        """Generate one batch spanning all eight failure categories."""
        rng = self._resolve_rng(seed)
        count = self._validate_count(n_per_category)
        candidates: list[AdversarialCandidate] = []
        generators = (
            self.generate_boundary_failure,
            self.generate_unstable_state,
            self.generate_numerical_singularity,
            self.generate_adversarial_input,
            self.generate_resource_exhaustion_probe,
            self.generate_pathological_equation,
            self.generate_unexpected_coupling,
            self.generate_cascading_failure_scenario,
        )
        for generator in generators:
            candidates.extend(generator(count=count, seed=int(rng.integers(0, 2**32 - 1))))
        return candidates

    def _resolve_rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(seed) if seed is not None else self.rng

    def _validate_count(self, count: int) -> int:
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("count must be an integer")
        if count < 1:
            raise ValueError("count must be >= 1")
        return count

    def _midpoint(self) -> np.ndarray:
        return (self.template_room.lower + self.template_room.upper) / 2.0

    def _span(self) -> np.ndarray:
        span = self.template_room.upper - self.template_room.lower
        return np.where(span == 0.0, 1.0, span)

    def _clone_room(self, **overrides: Any) -> Room:
        memory_updates = overrides.pop("memory_updates", None)
        room = self.template_room.clone(**overrides)
        if memory_updates:
            room.memory.update(memory_updates)
        return room

    def _pathological_division_equation(self) -> list:
        if EquationAST is None or Operator is None or Constant is None or Variable is None:
            return ["1 / x0"]
        equation = EquationAST(root=Operator("/", (Constant(1.0), Variable(0, "x0"))))
        return [equation]

    def _pathological_log_equation(self) -> list:
        if EquationAST is None or Operator is None or Variable is None:
            return ["log(x0 - x0)"]
        root = Operator("log", (Operator("-", (Variable(0, "x0"), Variable(0, "x0"))),))
        return [EquationAST(root=root)]


def run_red_team_campaign(
    candidates: Sequence[AdversarialCandidate],
    evaluate_fn: Callable[[Room], object],
    permission_gate: GenesisPermission,
) -> RedTeamReport:
    """Run candidates through permission + evaluation and aggregate results."""
    if not callable(evaluate_fn):
        raise TypeError("evaluate_fn must be callable")
    if not isinstance(permission_gate, GenesisPermission):
        raise TypeError("permission_gate must be a GenesisPermission or subclass")

    category_counts = {category: 0 for category in FailureCategory}
    category_failure_counts = {category: 0 for category in FailureCategory}
    analyses: list[FailureAnalysis] = []

    for candidate in candidates:
        category_counts[candidate.category] += 1
        permission_result = permission_gate.evaluate(
            candidate.room.x,
            candidate.room.lower,
            candidate.room.upper,
            w=np.maximum(np.abs(candidate.room.activation), _FLOAT_EPS),
            coupling=candidate.room.couplings,
        )
        evaluation_result: object | None = None
        exception: Exception | None = None
        try:
            evaluation_result = evaluate_fn(candidate.room)
        except Exception as exc:  # noqa: BLE001 - campaign must capture arbitrary failures
            exception = exc

        evaluation_passed = _evaluation_passed(evaluation_result, exception)
        failed = (not permission_result.admitted) or (not evaluation_passed)
        observed_category = (
            _infer_failure_category(candidate, permission_result, evaluation_result, exception)
            if failed
            else None
        )
        if failed:
            category_failure_counts[candidate.category] += 1
        failure_reason = _failure_reason(permission_result, evaluation_result, exception, failed)
        analyses.append(
            FailureAnalysis(
                candidate_id=candidate.room.id,
                category=candidate.category,
                observed_category=observed_category,
                failed=failed,
                permission_admitted=permission_result.admitted,
                evaluation_passed=evaluation_passed,
                severity=_severity(permission_result, evaluation_result, exception, failed),
                failure_reason=failure_reason,
                permission_result=permission_result,
                details={
                    "evaluation_summary": _summarize_result(evaluation_result),
                    "exception": None if exception is None else f"{type(exception).__name__}: {exception}",
                    "designed_category": candidate.category.value,
                },
            )
        )

    category_failure_rates = {
        category: (
            category_failure_counts[category] / category_counts[category]
            if category_counts[category]
            else 0.0
        )
        for category in FailureCategory
    }
    return RedTeamReport(
        analyses=analyses,
        category_counts=category_counts,
        category_failure_counts=category_failure_counts,
        category_failure_rates=category_failure_rates,
        suggested_improvements=_suggested_improvements(category_failure_rates, category_failure_counts),
    )


def _evaluation_passed(evaluation_result: object | None, exception: Exception | None) -> bool:
    if exception is not None:
        return False
    if isinstance(evaluation_result, dict) and "passed" in evaluation_result:
        return bool(evaluation_result["passed"])
    if hasattr(evaluation_result, "passed"):
        return bool(evaluation_result.passed)
    if isinstance(evaluation_result, (bool, np.bool_)):
        return bool(evaluation_result)
    if evaluation_result is None:
        return True
    if isinstance(evaluation_result, (int, float, np.floating, np.integer)):
        return bool(np.isfinite(float(evaluation_result)))
    if isinstance(evaluation_result, np.ndarray):
        return bool(np.all(np.isfinite(evaluation_result)))
    return True


def _infer_failure_category(
    candidate: AdversarialCandidate,
    permission_result: PermissionResult,
    evaluation_result: object | None,
    exception: Exception | None,
) -> FailureCategory:
    external = _external_failure_category(candidate, permission_result, evaluation_result, exception)
    if external is not None:
        return external

    if exception is not None:
        lowered = f"{type(exception).__name__}: {exception}".lower()
        if "memory" in lowered or "resource" in lowered:
            return FailureCategory.RESOURCE_EXHAUSTION
        if any(token in lowered for token in ("division", "zero", "overflow", "underflow", "nan", "inf")):
            return FailureCategory.NUMERICAL_SINGULARITY

    if _result_is_non_finite(evaluation_result):
        return FailureCategory.NUMERICAL_SINGULARITY
    if "cascade_stages" in candidate.room.memory:
        return FailureCategory.CASCADING_FAILURE
    if _numeric_compute_pressure(candidate.room) >= 100_000.0:
        return FailureCategory.RESOURCE_EXHAUSTION
    if _contains_pathological_equation(candidate.room):
        return FailureCategory.PATHOLOGICAL_EQUATION
    if _has_dense_coupling(candidate.room):
        return FailureCategory.UNEXPECTED_COUPLING
    if _looks_oscillatory(candidate.room.x):
        return FailureCategory.UNSTABLE_STATE
    if permission_result.phi > 0.0:
        return FailureCategory.BOUNDARY_FAILURE
    if candidate.category == FailureCategory.ADVERSARIAL_INPUT:
        return FailureCategory.ADVERSARIAL_INPUT
    return candidate.category


def _external_failure_category(
    candidate: AdversarialCandidate,
    permission_result: PermissionResult,
    evaluation_result: object | None,
    exception: Exception | None,
) -> FailureCategory | None:
    if _verification is None:
        return None
    function_names = (
        "classify_failure",
        "classify_red_team_failure",
        "classify_candidate_failure",
    )
    for name in function_names:
        fn = getattr(_verification, name, None)
        if not callable(fn):
            continue
        try:
            value = fn(
                candidate=candidate,
                room=candidate.room,
                permission_result=permission_result,
                evaluation_result=evaluation_result,
                exception=exception,
            )
        except TypeError:
            try:
                value = fn(candidate.room, evaluation_result)
            except Exception:  # pragma: no cover - defensive fallback only
                continue
        except Exception:  # pragma: no cover - defensive fallback only
            continue
        coerced = _coerce_category(value)
        if coerced is not None:
            return coerced
    return None


def _coerce_category(value: object) -> FailureCategory | None:
    if isinstance(value, FailureCategory):
        return value
    if isinstance(value, str):
        try:
            return FailureCategory(value)
        except ValueError:
            return None
    return None


def _result_is_non_finite(result: object | None) -> bool:
    if isinstance(result, np.ndarray):
        return not bool(np.all(np.isfinite(result)))
    if isinstance(result, (int, float, np.floating, np.integer)):
        return not bool(np.isfinite(float(result)))
    if isinstance(result, dict):
        value = result.get("value")
        if isinstance(value, (int, float, np.floating, np.integer)):
            return not bool(np.isfinite(float(value)))
    return False


def _numeric_compute_pressure(room: Room) -> float:
    values: list[float] = []
    for value in room.compute.values():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return max(values, default=0.0)


def _contains_pathological_equation(room: Room) -> bool:
    for equation in room.equations:
        if isinstance(equation, str) and ("log(" in equation or "/" in equation):
            return True
        root = getattr(equation, "root", None)
        if root is not None and ("log" in str(root) or "/" in str(root)):
            return True
    return False


def _has_dense_coupling(room: Room) -> bool:
    if room.couplings is None:
        return False
    off_diagonal = room.couplings - np.diag(np.diag(room.couplings))
    return bool(np.linalg.norm(off_diagonal, ord=1) > 0.0)


def _looks_oscillatory(x: np.ndarray) -> bool:
    nonzero_signs = np.sign(x[np.abs(x) > 1e-12])
    if nonzero_signs.size < 2:
        return False
    sign_changes = np.count_nonzero(np.diff(nonzero_signs) != 0)
    return bool(sign_changes >= max(1, nonzero_signs.size // 2))


def _failure_reason(
    permission_result: PermissionResult,
    evaluation_result: object | None,
    exception: Exception | None,
    failed: bool,
) -> str:
    if not failed:
        return "passed"
    if exception is not None:
        return f"{type(exception).__name__}: {exception}"
    if not permission_result.admitted:
        return (
            f"permission rejected: phi={permission_result.phi:.4g}, "
            f"cci={permission_result.cci:.4g}, margin={permission_result.margin:.4g}"
        )
    if isinstance(evaluation_result, dict) and "reason" in evaluation_result:
        return str(evaluation_result["reason"])
    if _result_is_non_finite(evaluation_result):
        return "evaluation produced a non-finite value"
    return "evaluation reported failure"


def _severity(
    permission_result: PermissionResult,
    evaluation_result: object | None,
    exception: Exception | None,
    failed: bool,
) -> float:
    if not failed:
        return 0.0
    severity = 1.0 + min(permission_result.phi, 5.0)
    severity += max(0.0, permission_result.cci - 1.0)
    if not permission_result.admitted:
        severity += 1.0
    if exception is not None or _result_is_non_finite(evaluation_result):
        severity += 2.0
    return float(min(severity, 10.0))


def _summarize_result(result: object | None) -> object:
    if isinstance(result, np.ndarray):
        return {
            "type": "ndarray",
            "shape": tuple(int(value) for value in result.shape),
            "finite": bool(np.all(np.isfinite(result))),
        }
    return result


def _suggested_improvements(
    category_failure_rates: dict[FailureCategory, float],
    category_failure_counts: dict[FailureCategory, int],
) -> list[str]:
    suggestions = {
        FailureCategory.BOUNDARY_FAILURE: (
            "Harden bound handling with tighter clipping and "
            "pre-normalization."
        ),
        FailureCategory.UNSTABLE_STATE: (
            "Add damping, step-size control, or stability regularizers for "
            "oscillatory states."
        ),
        FailureCategory.NUMERICAL_SINGULARITY: (
            "Guard divisions, logs, and exponentials with epsilon floors "
            "and finite-value checks."
        ),
        FailureCategory.ADVERSARIAL_INPUT: (
            "Canonicalize and validate conflicting candidate payloads "
            "before evaluation."
        ),
        FailureCategory.RESOURCE_EXHAUSTION: (
            "Enforce compute budgets and early-abort heuristics for "
            "high-cost candidates."
        ),
        FailureCategory.PATHOLOGICAL_EQUATION: (
            "Pre-screen equation structures for singular or log-invalid "
            "motifs before execution."
        ),
        FailureCategory.UNEXPECTED_COUPLING: (
            "Audit coupling matrices and cap cross-variable amplification "
            "before rollout."
        ),
        FailureCategory.CASCADING_FAILURE: (
            "Add staged circuit-breakers so one local breach cannot "
            "trigger system-wide collapse."
        ),
    }
    failing_categories = [
        category
        for category, count in category_failure_counts.items()
        if count > 0 and category_failure_rates[category] > 0.0
    ]
    failing_categories.sort(
        key=lambda category: (
            category_failure_rates[category],
            category_failure_counts[category],
            category.value,
        ),
        reverse=True,
    )
    return [suggestions[category] for category in failing_categories[:3]]
