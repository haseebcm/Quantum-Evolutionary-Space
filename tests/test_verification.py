import numpy as np
import pytest

import qes.verification as verification_module
from qes.closure import CrossDomainVerifier
from qes.equation_ast import Constant, EquationAST, Operator, Variable
from qes.invariants import InvariantEngine
from qes.permission import GenesisPermission
from qes.room import Room
from qes.verification import (
    Evidence,
    VerificationEngine,
    VerificationStageResult,
    constraint_verification,
    cross_domain_verification,
    invariant_verification,
    numerical_verification,
    safety_verification,
    stability_verification,
)


def make_room(**overrides) -> Room:
    fields = dict(
        x=np.array([0.25, -0.25], dtype=float),
        x_star=np.zeros(2, dtype=float),
        lower=-np.ones(2, dtype=float),
        upper=np.ones(2, dtype=float),
        activation=np.ones(2, dtype=float),
        compute={"cpu": 1.0},
    )
    fields.update(overrides)
    return Room(**fields)


def make_verifier() -> CrossDomainVerifier:
    return CrossDomainVerifier(
        encoder=lambda y: np.asarray(y, dtype=float),
        decoder=lambda x: x,
        transition_fn=lambda x, u, xi: x
        + np.asarray(u, dtype=float)
        + (np.zeros_like(x) if xi is None else np.asarray(xi, dtype=float)),
        state_space_check=lambda x: bool(np.all(np.abs(x) <= 1.5)),
        constraint_fn=lambda x, u: float(np.max(np.asarray(u, dtype=float)) - 0.5),
    )


class AllowCheckGate:
    def __init__(self, allow: bool) -> None:
        self.allow = allow

    def check(self, candidate: object) -> bool:
        return self.allow


class StateOnlyCheckGate:
    def check(self, state: object) -> bool:
        return bool(state[0] >= 0.0)  # type: ignore[index]


def test_stage_result_status_labels_cover_all_states():
    assert VerificationStageResult(stage="a", passed=True, message="ok").status == "passed"
    assert VerificationStageResult(stage="a", passed=False, message="bad").status == "failed"
    assert VerificationStageResult(stage="a", passed=None, message="skip", skipped=True).status == "skipped"


def test_evidence_bool_and_failing_stages_reflect_stage_outcomes():
    stages = (
        VerificationStageResult(stage="numerical", passed=True, message="ok"),
        VerificationStageResult(stage="stability", passed=False, message="bad"),
    )
    evidence = Evidence(candidate="c", stages=stages, passed=False, summary="summary")
    assert not evidence
    assert [stage.stage for stage in evidence.failing_stages()] == ["stability"]


def test_private_helper_validation_paths() -> None:
    with pytest.raises(TypeError, match="real-valued scalar"):
        verification_module._safe_float("bad", name="value")
    with pytest.raises(ValueError, match="finite"):
        verification_module._safe_float(float("inf"), name="value")
    with pytest.raises(ValueError, match="broadcastable"):
        verification_module._candidate_bounds(
            object(),
            lower=np.ones(3),
            upper=np.ones(3),
            shape=(2,),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        verification_module._coerce_numeric_array([[1.0], [2.0]], name="value")


def test_equation_root_respects_unavailable_module_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    equation = EquationAST(root=Variable(0))
    monkeypatch.setattr(verification_module, "_EQUATION_AST_AVAILABLE", False)
    assert verification_module._equation_root(equation) is None
    monkeypatch.setattr(verification_module, "_EQUATION_AST_AVAILABLE", True)
    assert verification_module._equation_root(Variable(0)) == Variable(0)


def test_permission_result_conversion_supports_attribute_and_truthy_fallbacks() -> None:
    class Allowed:
        allowed = True

    assert verification_module._permission_result_to_stage(Allowed()) == (True, {"allowed": True})
    assert verification_module._permission_result_to_stage(7) == (True, {"admitted": True})


def test_numerical_verification_passes_for_finite_state():
    result = numerical_verification(np.array([0.1, -0.2]))
    assert result.passed is True
    assert result.details["max_abs"] == pytest.approx(0.2)


def test_numerical_verification_fails_on_nan():
    result = numerical_verification(np.array([0.0, np.nan]))
    assert result.passed is False
    assert result.stage == "numerical"
    assert result.details["non_finite_indices"] == [1]


def test_numerical_verification_fails_on_large_magnitude():
    result = numerical_verification(np.array([2.0, 3.0]), max_magnitude=1.0)
    assert result.passed is False
    assert result.details["max_abs"] == pytest.approx(3.0)


def test_numerical_verification_rejects_non_numeric_state():
    with pytest.raises(TypeError):
        numerical_verification(["a", "b"])


def test_numerical_verification_rejects_invalid_thresholds_shapes_and_empty_states() -> None:
    with pytest.raises(ValueError, match="max_magnitude must be positive"):
        numerical_verification(np.array([1.0]), max_magnitude=0.0)
    with pytest.raises(ValueError, match="candidate_state must be non-empty"):
        numerical_verification(np.array([]))
    equation = EquationAST(root=Variable(0))
    with pytest.raises(ValueError, match="2D array"):
        numerical_verification(equation, samples=np.array([0.0, 1.0]))


def test_numerical_verification_uses_equation_ast_validation_for_good_equation():
    equation = EquationAST(root=Operator("+", (Variable(0), Constant(1.0))))
    samples = np.array([[0.0], [1.0], [2.0]], dtype=float)
    result = numerical_verification(equation, samples=samples)
    assert result.passed is True
    assert result.details["sample_count"] == 3


def test_numerical_verification_uses_equation_ast_dimension_validation():
    equation = EquationAST(root=Operator("+", (Variable(0), Variable(2))))
    samples = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    result = numerical_verification(equation, samples=samples)
    assert result.passed is False
    assert result.details["invalid_indices"] == [2]


def test_numerical_verification_uses_equation_ast_failure_report_message():
    equation = EquationAST(root=Operator("/", (Constant(1.0), Variable(0))))
    samples = np.array([[0.0], [1.0]], dtype=float)
    result = numerical_verification(equation, samples=samples)
    assert result.passed is False
    assert "failing_sample" in result.details


def test_constraint_verification_passes_within_bounds():
    result = constraint_verification(np.array([0.0, 0.5]), lower=-1.0, upper=1.0)
    assert result.passed is True


def test_constraint_verification_fails_outside_bounds():
    result = constraint_verification(np.array([0.0, 2.0]), lower=-1.0, upper=1.0)
    assert result.passed is False
    assert result.details["failing_indices"] == [1]


def test_constraint_verification_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        constraint_verification(np.array([0.0, 0.5]), lower=np.ones(2), upper=-np.ones(2))


def test_constraint_verification_rejects_empty_state():
    with pytest.raises(ValueError, match="non-empty"):
        constraint_verification(np.array([]), lower=-1.0, upper=1.0)


def test_safety_verification_passes_for_room_inside_permission_region():
    room = make_room()
    result = safety_verification(room, GenesisPermission(theta=1.0))
    assert result.passed is True
    assert result.details["admitted"] is True


def test_safety_verification_fails_for_room_outside_permission_region():
    room = make_room(x=np.array([5.0, 5.0]))
    result = safety_verification(room, GenesisPermission(theta=1.0))
    assert result.passed is False
    assert result.details["admitted"] is False


def test_safety_verification_supports_check_style_gate():
    result = safety_verification(object(), AllowCheckGate(True), candidate_state=np.array([0.0]))
    assert result.passed is True


def test_safety_verification_supports_state_only_check_gate_fallback():
    result = safety_verification(object(), StateOnlyCheckGate(), candidate_state=np.array([0.5]))
    assert result.passed is True


def test_safety_verification_rejects_none_permission_gate():
    with pytest.raises(TypeError, match="must not be None"):
        safety_verification(object(), None, candidate_state=np.array([0.0]))  # type: ignore[arg-type]


def test_safety_verification_rejects_unsupported_gate_interface():
    with pytest.raises(TypeError):
        safety_verification(object(), object(), candidate_state=np.array([0.0]))


def test_stability_verification_passes_for_generic_stable_function():
    result = stability_verification(
        np.array([0.25, -0.25]),
        lambda state: float(np.sum(state ** 2)),
        threshold=10.0,
    )
    assert result.passed is True
    assert result.details["max_sensitivity"] <= 10.0


def test_stability_verification_fails_for_generic_unstable_function():
    result = stability_verification(
        np.array([1e-6, 0.2]),
        lambda state: float(1.0 / state[0]),
        epsilon=1e-6,
        threshold=1e9,
    )
    assert result.passed is False
    assert result.details["max_sensitivity"] > result.details["threshold"]


def test_stability_verification_rejects_missing_generic_evaluate_fn():
    with pytest.raises(TypeError):
        stability_verification(np.array([0.1, 0.2]), None)


def test_stability_verification_rejects_invalid_parameters_and_empty_states() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        stability_verification(np.array([1.0]), lambda state: float(state[0]), epsilon=0.0)
    with pytest.raises(ValueError, match="threshold must be positive"):
        stability_verification(np.array([1.0]), lambda state: float(state[0]), threshold=0.0)
    with pytest.raises(ValueError, match="candidate_state must be non-empty"):
        stability_verification(np.array([]), lambda state: float(np.sum(state)))
    equation = EquationAST(root=Variable(1))
    with pytest.raises(ValueError, match="2D array"):
        stability_verification(equation, probe_states=np.array([0.0, 1.0]))


def test_stability_verification_uses_equation_ast_for_stable_expression():
    equation = EquationAST(root=Variable(0))
    probe_states = np.array([[0.1], [0.2], [0.3]], dtype=float)
    result = stability_verification(equation, probe_states=probe_states, threshold=10.0)
    assert result.passed is True
    assert result.details["max_sensitivity"] <= 10.0


def test_stability_verification_uses_equation_ast_for_unstable_expression():
    equation = EquationAST(root=Operator("/", (Constant(1.0), Variable(0))))
    probe_states = np.array([[1e-4], [2e-4], [5e-4]], dtype=float)
    result = stability_verification(equation, probe_states=probe_states, threshold=1e4)
    assert result.passed is False
    assert result.details["max_sensitivity"] > result.details["threshold"]


def test_stability_verification_uses_equation_ast_dimension_validation() -> None:
    equation = EquationAST(root=Variable(2))
    result = stability_verification(equation, probe_states=np.array([[0.0, 1.0]], dtype=float))
    assert result.passed is False
    assert result.details["invalid_indices"] == [2]


def test_invariant_verification_passes_for_healthy_room():
    result = invariant_verification(make_room(), InvariantEngine())
    assert result.passed is True
    assert result.details["violation_count"] == 0


def test_invariant_verification_fails_for_bad_room():
    room = make_room(x=np.array([0.0, np.nan]))
    result = invariant_verification(room, InvariantEngine())
    assert result.passed is False
    assert result.details["violation_count"] >= 1
    assert any(violation["check"] == "state_validity" for violation in result.details["violations"])


def test_invariant_verification_rejects_wrong_engine_type():
    with pytest.raises(TypeError):
        invariant_verification(make_room(), object())  # type: ignore[arg-type]


def test_cross_domain_verification_passes_across_multiple_domains():
    room = make_room()
    domains = [
        {
            "name": "engineering",
            "observation": room.x,
            "action": np.array([0.1, 0.0]),
            "disturbance": np.zeros(2),
        },
        {
            "name": "finance",
            "observation": room.x,
            "action": np.array([0.0, -0.1]),
            "disturbance": np.zeros(2),
        },
    ]
    result = cross_domain_verification(room, domains, make_verifier())
    assert result.passed is True
    assert result.details["domain_count"] == 2


def test_cross_domain_verification_fails_on_constraint_conflict():
    room = make_room()
    domains = [
        {
            "name": "conflict",
            "observation": room.x,
            "action": np.array([0.6, 0.0]),
            "disturbance": np.zeros(2),
        }
    ]
    result = cross_domain_verification(room, domains, make_verifier())
    assert result.passed is False
    assert result.details["domains"][0]["constraint_closed"] is False


def test_cross_domain_verification_fails_on_reconstruction_error_threshold():
    verifier = CrossDomainVerifier(
        encoder=lambda y: np.asarray(y, dtype=float),
        decoder=lambda x: x + 1.0,
        transition_fn=lambda x, u, xi: x,
        state_space_check=lambda x: True,
        constraint_fn=lambda x, u: 0.0,
    )
    domains = [{"observation": np.array([0.0]), "action": np.array([0.0]), "max_reconstruction_error": 0.1}]
    result = cross_domain_verification(make_room(), domains, verifier)
    assert result.passed is False
    assert result.details["domains"][0]["reconstruction_error"] > 0.1


def test_cross_domain_verification_validates_domain_inputs() -> None:
    with pytest.raises(TypeError, match="CrossDomainVerifier"):
        cross_domain_verification(make_room(), [{"action": np.array([0.0])}], object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        cross_domain_verification(make_room(), [], make_verifier())
    with pytest.raises(TypeError, match="mapping"):
        cross_domain_verification(make_room(), ["bad-domain"], make_verifier())  # type: ignore[list-item]
    with pytest.raises(TypeError, match="callable"):
        cross_domain_verification(
            make_room(),
            [{"action": np.array([0.0]), "error_fn": "bad"}],
            make_verifier(),
        )


def test_cross_domain_verification_passes_callable_error_fn_through() -> None:
    called_with: list[tuple[np.ndarray, np.ndarray]] = []

    def error_fn(expected: np.ndarray, observed: np.ndarray) -> float:
        called_with.append((expected, observed))
        return float(np.max(np.abs(expected - observed)))

    result = cross_domain_verification(
        make_room(),
        [{"action": np.array([0.0, 0.0]), "error_fn": error_fn}],
        make_verifier(),
    )
    assert result.passed is True
    assert len(called_with) == 1


def test_cross_domain_verification_rejects_missing_action():
    with pytest.raises(ValueError):
        cross_domain_verification(make_room(), [{"observation": np.array([0.0])}], make_verifier())


def test_verification_engine_all_passes_return_positive_evidence():
    room = make_room()
    engine = VerificationEngine(
        permission_gate=GenesisPermission(theta=1.0),
        invariant_engine=InvariantEngine(),
        cross_domain_verifier=make_verifier(),
        domains=[{"name": "eng", "action": np.array([0.1, 0.0]), "disturbance": np.zeros(2)}],
        evaluate_fn=lambda state: float(np.sum(state ** 2)),
        lower=-1.0,
        upper=1.0,
        stability_threshold=10.0,
    )
    evidence = engine.verify(room)
    assert evidence.passed is True
    assert len(evidence.stages) == 6
    assert all(stage.passed is not False for stage in evidence.stages)


def test_verification_engine_identifies_failing_stability_stage():
    room = make_room(x=np.array([1e-6, 0.2]))
    engine = VerificationEngine(
        permission_gate=GenesisPermission(theta=1.0),
        invariant_engine=InvariantEngine(),
        cross_domain_verifier=make_verifier(),
        domains=[{"name": "eng", "action": np.array([0.1, 0.0]), "disturbance": np.zeros(2)}],
        evaluate_fn=lambda state: float(1.0 / state[0]),
        lower=-1.0,
        upper=1.0,
        stability_epsilon=1e-6,
        stability_threshold=1e9,
    )
    evidence = engine.verify(room)
    assert evidence.passed is False
    assert [stage.stage for stage in evidence.failing_stages()] == ["stability"]


def test_verification_engine_skips_missing_optional_dependencies_gracefully():
    engine = VerificationEngine()
    evidence = engine.verify(make_room())
    statuses = {stage.stage: stage.status for stage in evidence.stages}
    assert statuses["safety"] == "skipped"
    assert statuses["invariant"] == "skipped"
    assert statuses["cross-domain"] == "skipped"
    assert evidence.passed is True


def test_verification_engine_skips_numerical_when_candidate_has_no_state() -> None:
    engine = VerificationEngine()
    evidence = engine.verify(object())
    stage_map = {stage.stage: stage for stage in evidence.stages}
    assert stage_map["numerical"].status == "skipped"
    assert stage_map["constraint"].status == "skipped"
    assert stage_map["stability"].status == "skipped"


def test_verification_engine_raises_when_permission_gate_lacks_bounds_context():
    engine = VerificationEngine(permission_gate=GenesisPermission(theta=1.0))
    with pytest.raises(TypeError):
        engine.verify(object(), candidate_state=np.array([0.0, 0.1]))


def test_verification_engine_supports_equation_candidates_with_equation_ast_reuse():
    equation = EquationAST(root=Operator("+", (Variable(0), Constant(1.0))))
    samples = np.array([[0.0], [1.0], [2.0]], dtype=float)
    engine = VerificationEngine(
        numerical_samples=samples,
        stability_samples=samples,
        stability_threshold=10.0,
    )
    evidence = engine.verify(equation)
    stage_map = {stage.stage: stage for stage in evidence.stages}
    assert stage_map["numerical"].passed is True
    assert stage_map["stability"].passed is True
    assert stage_map["constraint"].status == "skipped"
