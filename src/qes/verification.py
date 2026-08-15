"""Phase 15 -- Formal verification layer.

This module upgrades the candidate-evaluation story from a single scalar
score to a staged evidence package:

    Candidate
      -> numerical verification
      -> constraint verification
      -> safety verification
      -> stability verification
      -> invariant verification
      -> cross-domain verification
      -> Evidence(candidate)

The checks remain classical numerical/logical diagnostics over Python objects
and numpy arrays. They are not a theorem prover or a literal formal-methods
integration.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qes.closure import CrossDomainVerifier, DomainVerificationResult
from qes.invariants import InvariantEngine, InvariantReport
from qes.permission import AdaptivePermission, GenesisPermission, PermissionResult

try:  # Optional reuse path for equation-like candidates.
    from qes.equation_ast import (
        ASTNode,
        DimensionValidation,
        EquationAST,
        EquationASTForge,
        NumericalValidation,
        StabilityReport,
    )

    _EQUATION_AST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the module is absent.
    ASTNode = Any  # type: ignore[misc,assignment]
    DimensionValidation = Any  # type: ignore[misc,assignment]
    EquationAST = Any  # type: ignore[misc,assignment]
    EquationASTForge = Any  # type: ignore[misc,assignment]
    NumericalValidation = Any  # type: ignore[misc,assignment]
    StabilityReport = Any  # type: ignore[misc,assignment]
    _EQUATION_AST_AVAILABLE = False


PermissionGate = GenesisPermission | AdaptivePermission | object


def _safe_float(value: Any, *, name: str) -> float:
    """Return a finite float or raise a clear error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real-valued scalar") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _coerce_numeric_array(value: object, *, name: str, one_dimensional: bool = True) -> np.ndarray:
    """Return `value` as a numeric float array with basic dtype validation."""
    raw = np.asarray(value)
    # Explicitly reject boolean arrays with a clear error message expected by tests.
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must not be boolean")
    if raw.dtype.kind not in {"i", "u", "f", "c"}:
        raise TypeError(f"{name} must be numeric, got dtype {raw.dtype}")
    array = np.asarray(value, dtype=float)
    if one_dimensional and array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _candidate_state(candidate: object, candidate_state: object | None = None) -> np.ndarray:
    """Resolve a candidate's numerical state vector."""
    if candidate_state is not None:
        return _coerce_numeric_array(candidate_state, name="candidate_state")
    if hasattr(candidate, "x"):
        return _coerce_numeric_array(candidate.x, name="candidate.x")
    raise TypeError("candidate_state was not provided and candidate has no .x state vector")


def _candidate_bounds(
    candidate: object,
    lower: object | None,
    upper: object | None,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve lower/upper bounds for a candidate and broadcast scalars if needed."""
    lower_source = getattr(candidate, "lower", None) if lower is None else lower
    upper_source = getattr(candidate, "upper", None) if upper is None else upper
    if lower_source is None or upper_source is None:
        raise TypeError("lower and upper bounds are required for this verification stage")
    lower_array = np.asarray(lower_source, dtype=float)
    upper_array = np.asarray(upper_source, dtype=float)
    try:
        lower_array = np.broadcast_to(lower_array, shape).astype(float, copy=False)
        upper_array = np.broadcast_to(upper_array, shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"lower/upper bounds must be broadcastable to candidate_state shape {shape}"
        ) from exc
    return lower_array, upper_array


def _skip(stage: str, reason: str) -> VerificationStageResult:
    """Construct a skipped stage result."""
    return VerificationStageResult(stage=stage, passed=None, message=reason, skipped=True)


def _equation_root(candidate: object) -> ASTNode | None:
    """Return the underlying AST node when `candidate` looks equation-like."""
    if not _EQUATION_AST_AVAILABLE:
        return None
    if isinstance(candidate, EquationAST):
        return candidate.root
    if isinstance(candidate, ASTNode):
        return candidate
    return None


def _equation_forge() -> EquationASTForge:
    """Fresh helper forge used only for validation routines."""
    return EquationASTForge(rng=np.random.default_rng(0))


def _permission_result_to_stage(result: object) -> tuple[bool, dict[str, Any]]:
    """Interpret a permission-gate result object."""
    if isinstance(result, PermissionResult):
        details = {
            "phi": result.phi,
            "cci": result.cci,
            "margin": result.margin,
            "hard_permission": result.hard_permission,
            "soft_permission": result.soft_permission,
            "admitted": result.admitted,
        }
        return result.admitted, details

    for attr in ("admitted", "allowed", "passed", "ok", "is_safe"):
        if hasattr(result, attr):
            details = {attr: bool(getattr(result, attr))}
            return bool(getattr(result, attr)), details

    if isinstance(result, bool):
        return result, {"admitted": result}

    return bool(result), {"admitted": bool(result)}


def _report_details(report: InvariantReport) -> dict[str, Any]:
    """Convert an invariant report into plain serializable details."""
    violations = [
        {
            "scope": violation.scope,
            "check": violation.check,
            "message": violation.message,
            "object_id": violation.object_id,
        }
        for violation in report.violations
    ]
    return {"violations": violations, "violation_count": len(violations)}


@dataclass(frozen=True)
class VerificationStageResult:
    """Outcome of one verification stage."""

    stage: str
    passed: bool | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    @property
    def status(self) -> str:
        """Human-readable status label."""
        if self.skipped or self.passed is None:
            return "skipped"
        return "passed" if self.passed else "failed"


@dataclass(frozen=True)
class Evidence:
    """Ordered evidence package for one candidate."""

    candidate: object
    stages: tuple[VerificationStageResult, ...]
    passed: bool
    summary: str

    def failing_stages(self) -> tuple[VerificationStageResult, ...]:
        """Return every failed stage in pipeline order."""
        return tuple(stage for stage in self.stages if stage.passed is False)

    def __bool__(self) -> bool:
        return self.passed


def numerical_verification(
    candidate_state: object,
    *,
    samples: np.ndarray | None = None,
    max_magnitude: float = 1e6,
) -> VerificationStageResult:
    """Check that a candidate state or equation remains numerically sane.

    For equation-like candidates and a supplied `samples` batch, this reuses
    `EquationASTForge.validate_dimensions()` and
    `EquationASTForge.validate_numerically()`. Otherwise it performs generic
    finite-value and magnitude checks on a numeric state vector.
    """
    threshold = _safe_float(max_magnitude, name="max_magnitude")
    if threshold <= 0.0:
        raise ValueError("max_magnitude must be positive")

    root = _equation_root(candidate_state)
    if root is not None and samples is not None:
        sample_array = np.asarray(samples, dtype=float)
        if sample_array.ndim != 2:
            raise ValueError(f"samples must be a 2D array, got shape {sample_array.shape}")
        forge = _equation_forge()
        dim_report: DimensionValidation = forge.validate_dimensions(root, dim=sample_array.shape[1])
        if not dim_report.passed:
            return VerificationStageResult(
                stage="numerical",
                passed=False,
                message="equation references out-of-range dimensions",
                details={"invalid_indices": list(dim_report.invalid_indices)},
            )
        numerical_report: NumericalValidation = forge.validate_numerically(root, sample_array)
        if numerical_report.passed:
            return VerificationStageResult(
                stage="numerical",
                passed=True,
                message="equation numerical validation passed",
                details={"sample_count": int(sample_array.shape[0])},
            )
        return VerificationStageResult(
            stage="numerical",
            passed=False,
            message=numerical_report.message or "equation numerical validation failed",
            details={
                "failing_sample": numerical_report.failing_sample,
                "value": numerical_report.value,
            },
        )

    array = _coerce_numeric_array(candidate_state, name="candidate_state")
    if array.size == 0:
        raise ValueError("candidate_state must be non-empty")
    if not np.all(np.isfinite(array)):
        bad_indices = np.argwhere(~np.isfinite(array)).reshape(-1).tolist()
        return VerificationStageResult(
            stage="numerical",
            passed=False,
            message="candidate_state contains NaN or Inf",
            details={"non_finite_indices": bad_indices},
        )
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    if max_abs > threshold:
        return VerificationStageResult(
            stage="numerical",
            passed=False,
            message="candidate_state magnitude exceeds the configured limit",
            details={"max_abs": max_abs, "max_magnitude": threshold},
        )
    return VerificationStageResult(
        stage="numerical",
        passed=True,
        message="candidate_state is finite and within magnitude limits",
        details={"max_abs": max_abs, "shape": list(array.shape)},
    )


def constraint_verification(
    candidate_state: object,
    lower: object,
    upper: object,
) -> VerificationStageResult:
    """Check that a candidate state respects elementwise lower/upper bounds."""
    state = _coerce_numeric_array(candidate_state, name="candidate_state")
    if state.size == 0:
        raise ValueError("candidate_state must be non-empty")
    lower_array, upper_array = _candidate_bounds(
        candidate=object(),
        lower=lower,
        upper=upper,
        shape=state.shape,
    )
    if np.any(lower_array > upper_array):
        raise ValueError("lower must be <= upper elementwise")

    below = state < lower_array
    above = state > upper_array
    if bool(np.any(below) or np.any(above)):
        failing = np.argwhere(below | above).reshape(-1).tolist()
        return VerificationStageResult(
            stage="constraint",
            passed=False,
            message="candidate_state violates configured bounds",
            details={
                "failing_indices": failing,
                "lower": lower_array.tolist(),
                "upper": upper_array.tolist(),
            },
        )
    return VerificationStageResult(
        stage="constraint",
        passed=True,
        message="candidate_state satisfies configured bounds",
        details={"lower": lower_array.tolist(), "upper": upper_array.tolist()},
    )


def safety_verification(
    candidate: object,
    permission_gate: PermissionGate,
    *,
    candidate_state: object | None = None,
    lower: object | None = None,
    upper: object | None = None,
    weights: object | None = None,
    coupling: object | None = None,
) -> VerificationStageResult:
    """Run a candidate through a supplied permission gate."""
    if permission_gate is None:
        raise TypeError("permission_gate must not be None")

    evaluate_fn = getattr(permission_gate, "evaluate", None)
    if callable(evaluate_fn):
        state = _candidate_state(candidate, candidate_state)
        lower_array, upper_array = _candidate_bounds(candidate, lower, upper, state.shape)
        weight_array = (
            np.asarray(weights, dtype=float)
            if weights is not None
            else np.ones_like(state, dtype=float)
        )
        coupling_value = getattr(candidate, "couplings", None) if coupling is None else coupling
        result = evaluate_fn(state, lower_array, upper_array, weight_array, coupling_value)
        admitted, details = _permission_result_to_stage(result)
        message = "permission gate admitted candidate" if admitted else "permission gate rejected candidate"
        return VerificationStageResult(stage="safety", passed=admitted, message=message, details=details)

    check_fn = getattr(permission_gate, "check", None)
    if callable(check_fn):
        try:
            result = check_fn(candidate)
        except TypeError:
            state = _candidate_state(candidate, candidate_state)
            result = check_fn(state)
        admitted, details = _permission_result_to_stage(result)
        message = "permission gate admitted candidate" if admitted else "permission gate rejected candidate"
        return VerificationStageResult(stage="safety", passed=admitted, message=message, details=details)

    raise TypeError("permission_gate must expose a callable .evaluate(...) or .check(...) method")


def stability_verification(
    candidate_state: object,
    evaluate_fn: Callable[[np.ndarray], float] | None = None,
    *,
    probe_states: np.ndarray | None = None,
    epsilon: float = 1e-4,
    threshold: float = 100.0,
) -> VerificationStageResult:
    """Run a perturbation-based stability screen.

    For equation-like candidates and supplied `probe_states`, this reuses
    `EquationASTForge.analyze_stability()`. Otherwise it perturbs a single
    numeric state vector and verifies that the response remains finite and
    bounded by `threshold`.
    """
    epsilon_value = _safe_float(epsilon, name="epsilon")
    threshold_value = _safe_float(threshold, name="threshold")
    if epsilon_value <= 0.0:
        raise ValueError("epsilon must be positive")
    if threshold_value <= 0.0:
        raise ValueError("threshold must be positive")

    root = _equation_root(candidate_state)
    if root is not None and probe_states is not None:
        state_array = np.asarray(probe_states, dtype=float)
        if state_array.ndim != 2:
            raise ValueError(f"probe_states must be a 2D array, got shape {state_array.shape}")
        forge = _equation_forge()
        dim_report: DimensionValidation = forge.validate_dimensions(root, dim=state_array.shape[1])
        if not dim_report.passed:
            return VerificationStageResult(
                stage="stability",
                passed=False,
                message="equation references out-of-range dimensions",
                details={"invalid_indices": list(dim_report.invalid_indices)},
            )
        stability_report: StabilityReport = forge.analyze_stability(
            root,
            state_array,
            epsilon=epsilon_value,
            threshold=threshold_value,
        )
        details = {
            "max_sensitivity": stability_report.max_sensitivity,
            "threshold": stability_report.threshold,
            "failing_dimension": stability_report.failing_dimension,
        }
        message = (
            "equation stability validation passed"
            if stability_report.passed
            else "equation stability validation failed"
        )
        return VerificationStageResult(
            stage="stability", passed=stability_report.passed, message=message, details=details
        )

    if evaluate_fn is None or not callable(evaluate_fn):
        raise TypeError("evaluate_fn must be callable for generic stability verification")
    state = _coerce_numeric_array(candidate_state, name="candidate_state")
    if state.size == 0:
        raise ValueError("candidate_state must be non-empty")

    baseline = _safe_float(evaluate_fn(state.copy()), name="evaluate_fn(candidate_state)")
    max_ratio = 0.0
    failing_dimension: int | None = None
    for dimension in range(state.size):
        perturbed = np.array(state, copy=True)
        perturbed[dimension] += epsilon_value
        shifted = _safe_float(
            evaluate_fn(perturbed),
            name=f"evaluate_fn(candidate_state + perturbation[{dimension}])",
        )
        ratio = abs(shifted - baseline) / epsilon_value
        if ratio > max_ratio:
            max_ratio = ratio
            failing_dimension = dimension

    passed = max_ratio <= threshold_value
    return VerificationStageResult(
        stage="stability",
        passed=passed,
        message="stability probe passed" if passed else "stability probe exceeded sensitivity threshold",
        details={
            "baseline": baseline,
            "max_sensitivity": max_ratio,
            "threshold": threshold_value,
            "failing_dimension": None if passed else failing_dimension,
        },
    )


def invariant_verification(candidate: object, invariant_engine: InvariantEngine) -> VerificationStageResult:
    """Wrap `InvariantEngine.check(candidate)` as one verification stage."""
    if not isinstance(invariant_engine, InvariantEngine):
        raise TypeError("invariant_engine must be an InvariantEngine")
    report = invariant_engine.check(candidate)
    message = report.summary()
    return VerificationStageResult(
        stage="invariant",
        passed=report.passed,
        message=message,
        details=_report_details(report),
    )


def cross_domain_verification(
    candidate: object,
    domains: Sequence[Mapping[str, Any]],
    cross_domain_verifier: CrossDomainVerifier,
) -> VerificationStageResult:
    """Wrap `CrossDomainVerifier.verify()` across one or more domain payloads."""
    if not isinstance(cross_domain_verifier, CrossDomainVerifier):
        raise TypeError("cross_domain_verifier must be a CrossDomainVerifier")
    if not domains:
        raise ValueError("domains must be non-empty")

    per_domain: list[dict[str, Any]] = []
    passed = True
    for index, domain in enumerate(domains):
        if not isinstance(domain, Mapping):
            raise TypeError("each domain specification must be a mapping")
        observation = domain.get("observation", getattr(candidate, "x", candidate))
        if "action" not in domain:
            raise ValueError("each domain specification must provide an 'action'")
        action = domain["action"]
        disturbance = domain.get("disturbance")
        error_fn = domain.get("error_fn")
        threshold = _safe_float(domain.get("max_reconstruction_error", 1e-6), name="max_reconstruction_error")
        name = str(domain.get("name", f"domain-{index}"))

        kwargs: dict[str, Any] = {}
        if error_fn is not None:
            if not callable(error_fn):
                raise TypeError("domain error_fn must be callable")
            kwargs["error_fn"] = error_fn
        result: DomainVerificationResult = cross_domain_verifier.verify(
            observation, action, disturbance, **kwargs
        )
        domain_passed = (
            result.state_closed
            and result.constraint_closed
            and result.reconstruction_error <= threshold
        )
        per_domain.append(
            {
                "name": name,
                "reconstruction_error": result.reconstruction_error,
                "max_reconstruction_error": threshold,
                "state_closed": result.state_closed,
                "constraint_closed": result.constraint_closed,
                "passed": domain_passed,
            }
        )
        passed = passed and domain_passed

    message = "cross-domain verification passed" if passed else "cross-domain verification found conflicts"
    return VerificationStageResult(
        stage="cross-domain",
        passed=passed,
        message=message,
        details={"domains": per_domain, "domain_count": len(per_domain)},
    )


def _evidence_summary(stages: Sequence[VerificationStageResult]) -> str:
    """Build a compact evidence summary string."""
    passed = sum(stage.passed is True for stage in stages)
    failed = sum(stage.passed is False for stage in stages)
    skipped = sum(stage.skipped or stage.passed is None for stage in stages)
    return f"evidence package: passed={passed}, failed={failed}, skipped={skipped}"


class VerificationEngine:
    """Run the staged Phase 15 verification pipeline and produce evidence."""

    def __init__(
        self,
        *,
        lower: object | None = None,
        upper: object | None = None,
        permission_gate: PermissionGate | None = None,
        invariant_engine: InvariantEngine | None = None,
        cross_domain_verifier: CrossDomainVerifier | None = None,
        domains: Sequence[Mapping[str, Any]] | None = None,
        evaluate_fn: Callable[[np.ndarray], float] | None = None,
        numerical_samples: np.ndarray | None = None,
        stability_samples: np.ndarray | None = None,
        max_magnitude: float = 1e6,
        stability_epsilon: float = 1e-4,
        stability_threshold: float = 100.0,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.permission_gate = permission_gate
        self.invariant_engine = invariant_engine
        self.cross_domain_verifier = cross_domain_verifier
        self.domains = tuple(domains) if domains is not None else None
        self.evaluate_fn = evaluate_fn
        self.numerical_samples = numerical_samples
        self.stability_samples = stability_samples
        self.max_magnitude = max_magnitude
        self.stability_epsilon = stability_epsilon
        self.stability_threshold = stability_threshold

    def verify(
        self,
        candidate: object,
        *,
        candidate_state: object | None = None,
        lower: object | None = None,
        upper: object | None = None,
        permission_gate: PermissionGate | None = None,
        invariant_engine: InvariantEngine | None = None,
        cross_domain_verifier: CrossDomainVerifier | None = None,
        domains: Sequence[Mapping[str, Any]] | None = None,
        evaluate_fn: Callable[[np.ndarray], float] | None = None,
        numerical_samples: np.ndarray | None = None,
        stability_samples: np.ndarray | None = None,
        max_magnitude: float | None = None,
        stability_epsilon: float | None = None,
        stability_threshold: float | None = None,
        weights: object | None = None,
        coupling: object | None = None,
    ) -> Evidence:
        """Run the full verification chain and return `Evidence(candidate)`."""
        stages: list[VerificationStageResult] = []
        numeric_input = candidate if _equation_root(candidate) is not None else None
        samples = self.numerical_samples if numerical_samples is None else numerical_samples

        if numeric_input is not None and samples is not None:
            stages.append(
                numerical_verification(
                    numeric_input,
                    samples=samples,
                    max_magnitude=self.max_magnitude if max_magnitude is None else max_magnitude,
                )
            )
        else:
            try:
                state = _candidate_state(candidate, candidate_state)
            except TypeError:
                stages.append(_skip("numerical", "no numeric state or equation samples supplied"))
            else:
                stages.append(
                    numerical_verification(
                        state,
                        max_magnitude=self.max_magnitude if max_magnitude is None else max_magnitude,
                    )
                )

        try:
            state_for_bounds = _candidate_state(candidate, candidate_state)
            lower_value, upper_value = _candidate_bounds(
                candidate,
                self.lower if lower is None else lower,
                self.upper if upper is None else upper,
                state_for_bounds.shape,
            )
        except TypeError:
            stages.append(_skip("constraint", "no bounds supplied for constraint verification"))
        else:
            stages.append(constraint_verification(state_for_bounds, lower_value, upper_value))

        active_gate = self.permission_gate if permission_gate is None else permission_gate
        if active_gate is None:
            stages.append(_skip("safety", "no permission gate supplied"))
        else:
            stages.append(
                safety_verification(
                    candidate,
                    active_gate,
                    candidate_state=candidate_state,
                    lower=self.lower if lower is None else lower,
                    upper=self.upper if upper is None else upper,
                    weights=weights,
                    coupling=coupling,
                )
            )

        active_evaluate_fn = self.evaluate_fn if evaluate_fn is None else evaluate_fn
        active_probe_states = self.stability_samples if stability_samples is None else stability_samples
        if _equation_root(candidate) is not None and active_probe_states is not None:
            stages.append(
                stability_verification(
                    candidate,
                    probe_states=active_probe_states,
                    epsilon=(
                        self.stability_epsilon
                        if stability_epsilon is None
                        else stability_epsilon
                    ),
                    threshold=(
                        self.stability_threshold
                        if stability_threshold is None
                        else stability_threshold
                    ),
                )
            )
        elif active_evaluate_fn is None:
            stages.append(_skip("stability", "no evaluate_fn or equation probe states supplied"))
        else:
            state = _candidate_state(candidate, candidate_state)
            stages.append(
                stability_verification(
                    state,
                    active_evaluate_fn,
                    epsilon=(
                        self.stability_epsilon
                        if stability_epsilon is None
                        else stability_epsilon
                    ),
                    threshold=(
                        self.stability_threshold
                        if stability_threshold is None
                        else stability_threshold
                    ),
                )
            )

        active_invariant_engine = self.invariant_engine if invariant_engine is None else invariant_engine
        if active_invariant_engine is None:
            stages.append(_skip("invariant", "no invariant engine supplied"))
        else:
            stages.append(invariant_verification(candidate, active_invariant_engine))

        active_domains = self.domains if domains is None else tuple(domains)
        active_verifier = (
            self.cross_domain_verifier
            if cross_domain_verifier is None
            else cross_domain_verifier
        )
        if active_domains is None or active_verifier is None:
            stages.append(_skip("cross-domain", "no cross-domain verifier/domains supplied"))
        else:
            stages.append(cross_domain_verification(candidate, active_domains, active_verifier))

        overall_passed = all(stage.passed is not False for stage in stages)
        return Evidence(
            candidate=candidate,
            stages=tuple(stages),
            passed=overall_passed,
            summary=_evidence_summary(stages),
        )


__all__ = [
    "Evidence",
    "VerificationEngine",
    "VerificationStageResult",
    "constraint_verification",
    "cross_domain_verification",
    "invariant_verification",
    "numerical_verification",
    "safety_verification",
    "stability_verification",
]
