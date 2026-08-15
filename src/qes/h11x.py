"""H^11X: the six-layer engineering admissibility stack.

Source-derived architecture (see the PDF-extraction summary, section E,
"H^11X adds a separate six-layer engineering admissibility stack"):

    Layer 1  Necessity Field           N(x) >= 0 or formation is blocked
    Layer 2  Constraint Geometry       Need + Constraint -> Geometry
    Layer 3  Domain Gatekeeper         no domain may override another
    Layer 4  Failure Anticipation      push toward failure; deny unless a
                                       recovery path exists
    Layer 5  Self-Generative Correction  failure -> reintegration ->
                                       constraint relaxation -> reformation
    Layer 6  Export Barrier            only derived results leave H^11X;
                                       internal axioms/kernels/equations
                                       stay separated from output

The source states derived domain cores are produced as `C_i = F(H11X)`
while claiming `H11X = F^-1(C_i)` does not hold (non-reconstructability is
a design claim in the source, not a proof this module attempts to enforce
cryptographically -- `export_barrier()` simply never returns the internal
state alongside the derived result).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _validated_scalar(value: Any, *, name: str) -> float:
    """Return `value` as a finite float."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real-valued scalar") from exc
    if scalar != scalar or scalar in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return scalar


def _require_callable(fn: object, *, name: str) -> None:
    """Raise when a required callback is not callable."""
    if not callable(fn):
        raise TypeError(f"{name} must be callable")


def necessity_field(n: float) -> bool:
    """Layer 1: True iff N(x) >= 0, i.e. formation of the proposed entity
    remains admissible before any geometry or material choice is made."""
    return _validated_scalar(n, name="n") >= 0.0


@dataclass
class Geometry:
    """Layer 2 output: geometry emergent from need + constraint."""

    stress: float
    load_paths: float
    energy_flow: float
    temporal_stability: float

    def __post_init__(self) -> None:
        """Normalize geometry components to validated finite floats."""
        self.stress = _validated_scalar(self.stress, name="stress")
        self.load_paths = _validated_scalar(self.load_paths, name="load_paths")
        self.energy_flow = _validated_scalar(self.energy_flow, name="energy_flow")
        self.temporal_stability = _validated_scalar(
            self.temporal_stability,
            name="temporal_stability",
        )

    def severity(self) -> float:
        """A coarse scalar summary of how demanding this geometry is."""
        return self.stress + self.load_paths + self.energy_flow - self.temporal_stability


def constraint_geometry(
    stress: float, load_paths: float, energy_flow: float, temporal_stability: float
) -> Geometry:
    """Layer 2: Need + Constraint -> Geometry (not Aesthetic Shape -> Check Constraints)."""
    return Geometry(
        stress=stress,
        load_paths=load_paths,
        energy_flow=energy_flow,
        temporal_stability=temporal_stability,
    )


def domain_gatekeeper(domain_checks: dict[str, bool]) -> bool:
    """Layer 3: admissible only if every domain (material, chemical,
    mechanical, thermal, environmental, ...) passes; one domain cannot
    override another -- a single failing domain fails the whole gate."""
    if not isinstance(domain_checks, dict):
        raise TypeError("domain_checks must be a dict[str, bool]")
    if not domain_checks:
        return False
    for name, passed in domain_checks.items():
        if not isinstance(name, str):
            raise TypeError("domain_checks keys must be strings")
        if not isinstance(passed, bool):
            raise TypeError(f"domain_checks[{name!r}] must be a bool")
    return all(domain_checks.values())


def failure_anticipation(
    push_to_failure_fn: Callable[[], float], recovery_check_fn: Callable[[float], bool]
) -> bool:
    """Layer 4: create candidate -> push toward failure -> recovery path?
    Deny if no recovery path exists at the pushed (stressed) state; continue
    (admit) otherwise."""
    _require_callable(push_to_failure_fn, name="push_to_failure_fn")
    _require_callable(recovery_check_fn, name="recovery_check_fn")
    stressed_value = _validated_scalar(push_to_failure_fn(), name="push_to_failure_fn()")
    return bool(recovery_check_fn(stressed_value))


@dataclass
class CorrectionOutcome:
    """Layer 5 output: the regenerated state after correction."""

    reintegrated: Any
    relaxed: Any
    reformed: Any


def self_generative_correction(
    failed_state: Any,
    reintegrate_fn: Callable[[Any], Any],
    relax_fn: Callable[[Any], Any],
    reform_fn: Callable[[Any], Any],
) -> CorrectionOutcome:
    """Layer 5: Failure -> Reintegration -> Constraint relaxation -> Reformation.

    A regeneration architecture rather than simple rejection: each stage
    consumes the previous stage's output.
    """
    _require_callable(reintegrate_fn, name="reintegrate_fn")
    _require_callable(relax_fn, name="relax_fn")
    _require_callable(reform_fn, name="reform_fn")
    reintegrated = reintegrate_fn(failed_state)
    relaxed = relax_fn(reintegrated)
    reformed = reform_fn(relaxed)
    return CorrectionOutcome(reintegrated=reintegrated, relaxed=relaxed, reformed=reformed)


@dataclass
class ExportResult:
    """Layer 6 output: only `derived` ever leaves H^11X."""

    derived: Any


def export_barrier(internal_state: Any, derive_fn: Callable[[Any], Any]) -> ExportResult:
    """Layer 6: derive a result from internal axioms/kernels/equations, but
    return only the derived artifact -- `internal_state` itself never
    crosses the barrier."""
    _require_callable(derive_fn, name="derive_fn")
    derived = derive_fn(internal_state)
    return ExportResult(derived=derived)


@dataclass
class H11XResult:
    """Outcome of one full `H11X.evaluate()` pass through all six layers."""

    admitted: bool
    geometry: Geometry | None = None
    correction: CorrectionOutcome | None = None
    export: ExportResult | None = None
    denied_at_layer: int | None = None


class H11X:
    """Runs a candidate through the full six-layer engineering
    admissibility stack, denying at the first layer that fails and, on a
    detected failure, running the self-generative correction loop before
    re-checking the domain gate."""

    def evaluate(
        self,
        n: float,
        stress: float,
        load_paths: float,
        energy_flow: float,
        temporal_stability: float,
        domain_checks: dict[str, bool],
        push_to_failure_fn: Callable[[], float],
        recovery_check_fn: Callable[[float], bool],
        internal_state: Any,
        derive_fn: Callable[[Any], Any],
        reintegrate_fn: Callable[[Any], Any] | None = None,
        relax_fn: Callable[[Any], Any] | None = None,
        reform_fn: Callable[[Any], Any] | None = None,
    ) -> H11XResult:
        """Run one candidate through the full six-layer H^11X admissibility stack."""
        _require_callable(push_to_failure_fn, name="push_to_failure_fn")
        _require_callable(recovery_check_fn, name="recovery_check_fn")
        _require_callable(derive_fn, name="derive_fn")
        if not necessity_field(n):
            return H11XResult(admitted=False, denied_at_layer=1)

        geometry = constraint_geometry(stress, load_paths, energy_flow, temporal_stability)

        if not domain_gatekeeper(domain_checks):
            return H11XResult(admitted=False, geometry=geometry, denied_at_layer=3)

        correction = None
        if not failure_anticipation(push_to_failure_fn, recovery_check_fn):
            if reintegrate_fn is None or relax_fn is None or reform_fn is None:
                return H11XResult(admitted=False, geometry=geometry, denied_at_layer=4)
            _require_callable(reintegrate_fn, name="reintegrate_fn")
            _require_callable(relax_fn, name="relax_fn")
            _require_callable(reform_fn, name="reform_fn")
            correction = self_generative_correction(
                internal_state, reintegrate_fn, relax_fn, reform_fn
            )
            internal_state = correction.reformed

        export = export_barrier(internal_state, derive_fn)
        return H11XResult(admitted=True, geometry=geometry, correction=correction, export=export)
