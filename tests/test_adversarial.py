import numpy as np
import pytest

from qes.adversarial import (
    AdversarialCandidate,
    AdversarialGenerator,
    FailureCategory,
    run_red_team_campaign,
)
from qes.permission import GenesisPermission
from qes.room import Room


def make_template_room(dim: int = 3) -> Room:
    return Room(
        x=np.zeros(dim),
        x_star=np.zeros(dim),
        lower=-np.ones(dim),
        upper=np.ones(dim),
        activation=np.ones(dim),
        compute={"compute": 1.0},
    )


def make_candidate(
    category: FailureCategory,
    *,
    x: np.ndarray | None = None,
    couplings: np.ndarray | None = None,
    compute: dict | None = None,
    memory: dict | None = None,
    equations: list | None = None,
) -> AdversarialCandidate:
    room = make_template_room()
    if x is not None:
        room = room.clone(x=np.asarray(x, dtype=float))
    if couplings is not None:
        room = room.clone(couplings=np.asarray(couplings, dtype=float))
    if compute is not None:
        room = room.clone(compute=dict(compute))
    if equations is not None:
        room = room.clone(equations=list(equations))
    if memory:
        room.memory.update(memory)
    return AdversarialCandidate(room=room, category=category, rationale=f"probe {category.value}")


def test_failure_category_members_cover_spec():
    assert {category.value for category in FailureCategory} == {
        "boundary_failure",
        "unstable_state",
        "numerical_singularity",
        "adversarial_input",
        "resource_exhaustion",
        "pathological_equation",
        "unexpected_coupling",
        "cascading_failure",
    }


def test_adversarial_candidate_requires_non_empty_rationale():
    with pytest.raises(ValueError, match="non-empty"):
        AdversarialCandidate(
            room=make_template_room(),
            category=FailureCategory.BOUNDARY_FAILURE,
            rationale="",
        )


def test_generate_boundary_failure_tags_category():
    generator = AdversarialGenerator(make_template_room(), seed=1)
    candidate = generator.generate_boundary_failure()[0]
    assert candidate.category is FailureCategory.BOUNDARY_FAILURE
    assert isinstance(candidate.room, Room)


def test_generate_unstable_state_tags_category():
    generator = AdversarialGenerator(make_template_room(), seed=1)
    candidate = generator.generate_unstable_state()[0]
    assert candidate.category is FailureCategory.UNSTABLE_STATE


def test_generate_numerical_singularity_tags_category():
    generator = AdversarialGenerator(make_template_room(), seed=1)
    candidate = generator.generate_numerical_singularity()[0]
    assert candidate.category is FailureCategory.NUMERICAL_SINGULARITY


def test_generate_adversarial_input_tags_category():
    generator = AdversarialGenerator(make_template_room(), seed=1)
    candidate = generator.generate_adversarial_input()[0]
    assert candidate.category is FailureCategory.ADVERSARIAL_INPUT


def test_generate_resource_exhaustion_tags_category():
    generator = AdversarialGenerator(make_template_room(), seed=1)
    candidate = generator.generate_resource_exhaustion_probe()[0]
    assert candidate.category is FailureCategory.RESOURCE_EXHAUSTION


def test_generate_pathological_equation_tags_category():
    generator = AdversarialGenerator(make_template_room(), seed=1)
    candidate = generator.generate_pathological_equation()[0]
    assert candidate.category is FailureCategory.PATHOLOGICAL_EQUATION


def test_generate_unexpected_coupling_tags_category():
    generator = AdversarialGenerator(make_template_room(), seed=1)
    candidate = generator.generate_unexpected_coupling()[0]
    assert candidate.category is FailureCategory.UNEXPECTED_COUPLING


def test_generate_cascading_failure_tags_category():
    generator = AdversarialGenerator(make_template_room(), seed=1)
    candidate = generator.generate_cascading_failure_scenario()[0]
    assert candidate.category is FailureCategory.CASCADING_FAILURE


def test_boundary_generator_pushes_state_outside_bounds():
    generator = AdversarialGenerator(make_template_room(), seed=2)
    candidate = generator.generate_boundary_failure()[0]
    assert np.any(candidate.room.x < candidate.room.lower) or np.any(candidate.room.x > candidate.room.upper)


def test_unstable_generator_creates_alternating_sign_pattern():
    generator = AdversarialGenerator(make_template_room(dim=4), seed=2)
    candidate = generator.generate_unstable_state()[0]
    assert np.count_nonzero(np.diff(np.sign(candidate.room.x)) != 0) >= 2


def test_numerical_generator_places_first_dimension_near_zero():
    generator = AdversarialGenerator(make_template_room(), seed=2)
    candidate = generator.generate_numerical_singularity()[0]
    assert abs(candidate.room.x[0]) < 1e-5


def test_adversarial_input_generator_sets_conflicting_metadata():
    generator = AdversarialGenerator(make_template_room(), seed=2)
    candidate = generator.generate_adversarial_input()[0]
    assert "conflicting_hints" in candidate.room.memory
    assert "duplicate_fields" in candidate.room.memory


def test_resource_generator_requests_large_compute_budget():
    generator = AdversarialGenerator(make_template_room(), seed=2)
    candidate = generator.generate_resource_exhaustion_probe()[0]
    assert candidate.room.compute["compute"] >= 250_000.0


def test_pathological_generator_attaches_equation_or_marker():
    generator = AdversarialGenerator(make_template_room(), seed=2)
    candidate = generator.generate_pathological_equation()[0]
    assert candidate.room.equations or candidate.room.memory["pathological_pattern"]


def test_unexpected_coupling_generator_creates_non_diagonal_coupling():
    generator = AdversarialGenerator(make_template_room(), seed=2)
    candidate = generator.generate_unexpected_coupling()[0]
    off_diagonal = candidate.room.couplings - np.diag(np.diag(candidate.room.couplings))
    assert np.any(off_diagonal != 0.0)


def test_cascading_generator_records_cascade_stages():
    generator = AdversarialGenerator(make_template_room(), seed=2)
    candidate = generator.generate_cascading_failure_scenario()[0]
    assert candidate.room.memory["cascade_stages"] == [
        "boundary breach",
        "coupling amplification",
        "recovery saturation",
    ]


def test_generate_all_is_deterministic_for_same_seed():
    template = make_template_room()
    generator = AdversarialGenerator(template, seed=0)
    first = generator.generate_all(n_per_category=2, seed=11)
    second = generator.generate_all(n_per_category=2, seed=11)
    assert [candidate.category for candidate in first] == [candidate.category for candidate in second]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.room.x, right.room.x)
        np.testing.assert_allclose(left.room.couplings, right.room.couplings)
        assert left.room.compute == right.room.compute
        assert left.room.memory == right.room.memory


def test_generate_all_changes_with_different_seed():
    template = make_template_room()
    generator = AdversarialGenerator(template, seed=0)
    first = generator.generate_all(seed=11)
    second = generator.generate_all(seed=12)
    assert any(
        not np.allclose(a.room.x, b.room.x)
        for a, b in zip(first, second, strict=True)
    )


def test_generate_all_returns_expected_total_count():
    generator = AdversarialGenerator(make_template_room(), seed=0)
    candidates = generator.generate_all(n_per_category=3, seed=9)
    assert len(candidates) == 24


def test_generate_all_rejects_invalid_count():
    generator = AdversarialGenerator(make_template_room(), seed=0)
    with pytest.raises(ValueError, match=">= 1"):
        generator.generate_all(n_per_category=0)


def test_category_generator_rejects_invalid_count():
    generator = AdversarialGenerator(make_template_room(), seed=0)
    with pytest.raises(ValueError, match=">= 1"):
        generator.generate_boundary_failure(count=0)


def test_run_red_team_campaign_handles_empty_candidate_list():
    report = run_red_team_campaign([], lambda room: True, GenesisPermission(theta=10.0))
    assert report.total_candidates == 0
    assert report.total_failures == 0
    assert all(rate == 0.0 for rate in report.category_failure_rates.values())
    assert report.suggested_improvements == []


def test_run_red_team_campaign_classifies_exceptions_as_failures():
    candidate = make_candidate(FailureCategory.NUMERICAL_SINGULARITY, x=np.array([1e-8, 0.0, 0.0]))

    def evaluate(_: Room) -> float:
        raise ZeroDivisionError("division by zero")

    report = run_red_team_campaign([candidate], evaluate, GenesisPermission(theta=10.0))
    analysis = report.analyses[0]
    assert analysis.failed
    assert analysis.observed_category is FailureCategory.NUMERICAL_SINGULARITY
    assert "ZeroDivisionError" in analysis.failure_reason


def test_run_red_team_campaign_honors_dict_passed_flag():
    candidate = make_candidate(FailureCategory.ADVERSARIAL_INPUT)
    report = run_red_team_campaign(
        [candidate],
        lambda room: {"passed": False, "reason": "policy mismatch"},
        GenesisPermission(theta=10.0),
    )
    analysis = report.analyses[0]
    assert analysis.failed
    assert analysis.failure_reason == "policy mismatch"


def test_run_red_team_campaign_honors_boolean_false_results():
    candidate = make_candidate(FailureCategory.ADVERSARIAL_INPUT)
    report = run_red_team_campaign([candidate], lambda room: False, GenesisPermission(theta=10.0))
    assert report.analyses[0].failed


def test_run_red_team_campaign_treats_non_finite_arrays_as_failures():
    candidate = make_candidate(FailureCategory.NUMERICAL_SINGULARITY)
    report = run_red_team_campaign(
        [candidate],
        lambda room: np.array([np.nan, 1.0]),
        GenesisPermission(theta=10.0),
    )
    analysis = report.analyses[0]
    assert analysis.failed
    assert analysis.observed_category is FailureCategory.NUMERICAL_SINGULARITY


def test_run_red_team_campaign_allows_clean_candidate_to_pass():
    candidate = make_candidate(FailureCategory.ADVERSARIAL_INPUT, x=np.array([0.1, 0.2, -0.1]))
    report = run_red_team_campaign(
        [candidate], lambda room: float(np.sum(room.x)), GenesisPermission(theta=10.0)
    )
    analysis = report.analyses[0]
    assert not analysis.failed
    assert report.suggested_improvements == []


def test_run_red_team_campaign_permission_rejection_causes_failure():
    candidate = make_candidate(FailureCategory.BOUNDARY_FAILURE, x=np.array([3.0, 0.0, 0.0]))
    report = run_red_team_campaign([candidate], lambda room: True, GenesisPermission(theta=10.0))
    analysis = report.analyses[0]
    assert analysis.failed
    assert not analysis.permission_admitted
    assert analysis.observed_category is FailureCategory.BOUNDARY_FAILURE


def test_red_team_report_aggregation_counts_and_rates_are_correct():
    candidates = [
        make_candidate(FailureCategory.BOUNDARY_FAILURE, x=np.array([2.0, 0.0, 0.0])),
        make_candidate(FailureCategory.BOUNDARY_FAILURE, x=np.array([0.0, 0.0, 0.0])),
        make_candidate(FailureCategory.ADVERSARIAL_INPUT, x=np.array([0.0, 0.0, 0.0])),
    ]
    results = iter([True, True, False])
    report = run_red_team_campaign(candidates, lambda room: next(results), GenesisPermission(theta=10.0))
    assert report.category_counts[FailureCategory.BOUNDARY_FAILURE] == 2
    assert report.category_failure_counts[FailureCategory.BOUNDARY_FAILURE] == 1
    assert report.category_failure_counts[FailureCategory.ADVERSARIAL_INPUT] == 1
    assert report.category_failure_rates[FailureCategory.BOUNDARY_FAILURE] == pytest.approx(0.5)
    assert report.category_failure_rates[FailureCategory.ADVERSARIAL_INPUT] == pytest.approx(1.0)


def test_suggested_improvements_populate_when_failures_occur():
    candidates = [make_candidate(FailureCategory.RESOURCE_EXHAUSTION, compute={"compute": 500_000.0})]
    report = run_red_team_campaign(
        candidates,
        lambda room: (_ for _ in ()).throw(MemoryError("resource exhaustion")),
        GenesisPermission(theta=10.0),
    )
    assert report.suggested_improvements


def test_dense_coupling_failure_is_classified_as_unexpected_coupling():
    couplings = np.full((3, 3), 2.0)
    np.fill_diagonal(couplings, 1.0)
    candidate = make_candidate(
        FailureCategory.UNEXPECTED_COUPLING,
        x=np.array([1.05, 0.0, 0.0]),
        couplings=couplings,
    )
    report = run_red_team_campaign([candidate], lambda room: True, GenesisPermission(theta=0.1))
    assert report.analyses[0].observed_category is FailureCategory.UNEXPECTED_COUPLING


def test_cascade_failure_is_classified_from_metadata():
    candidate = make_candidate(
        FailureCategory.CASCADING_FAILURE,
        x=np.array([1.1, 1.05, 0.0]),
        memory={"cascade_stages": ["breach", "amplify", "collapse"]},
    )
    report = run_red_team_campaign([candidate], lambda room: False, GenesisPermission(theta=10.0))
    assert report.analyses[0].observed_category is FailureCategory.CASCADING_FAILURE


def test_resource_failure_is_classified_from_compute_pressure():
    candidate = make_candidate(
        FailureCategory.RESOURCE_EXHAUSTION,
        compute={"compute": 500_000.0, "memory_mb": 128_000.0},
    )
    report = run_red_team_campaign([candidate], lambda room: False, GenesisPermission(theta=10.0))
    assert report.analyses[0].observed_category is FailureCategory.RESOURCE_EXHAUSTION


def test_pathological_equation_failure_is_classified_from_equation_payload():
    candidate = make_candidate(
        FailureCategory.PATHOLOGICAL_EQUATION,
        equations=["log(x0 - x0)"],
    )
    report = run_red_team_campaign([candidate], lambda room: False, GenesisPermission(theta=10.0))
    assert report.analyses[0].observed_category is FailureCategory.PATHOLOGICAL_EQUATION


def test_unstable_failure_is_classified_from_oscillatory_state():
    candidate = make_candidate(FailureCategory.UNSTABLE_STATE, x=np.array([2.0, -2.0, 2.0]))
    report = run_red_team_campaign([candidate], lambda room: False, GenesisPermission(theta=100.0))
    assert report.analyses[0].observed_category is FailureCategory.UNSTABLE_STATE


def test_adversarial_candidate_validates_all_input_types() -> None:
    with pytest.raises(TypeError, match="room must be a Room"):
        AdversarialCandidate(  # type: ignore[arg-type]
            room=object(),
            category=FailureCategory.BOUNDARY_FAILURE,
            rationale="ok",
        )
    with pytest.raises(TypeError, match="FailureCategory"):
        AdversarialCandidate(  # type: ignore[arg-type]
            room=make_template_room(),
            category="boundary_failure",
            rationale="ok",
        )
    with pytest.raises(TypeError, match="string"):
        AdversarialCandidate(  # type: ignore[arg-type]
            room=make_template_room(),
            category=FailureCategory.BOUNDARY_FAILURE,
            rationale=123,
        )
    with pytest.raises(ValueError, match="finite"):
        AdversarialCandidate(
            room=make_template_room(),
            category=FailureCategory.BOUNDARY_FAILURE,
            rationale="ok",
            stress_score=float("nan"),
        )
    with pytest.raises(ValueError, match=">= 0"):
        AdversarialCandidate(
            room=make_template_room(),
            category=FailureCategory.BOUNDARY_FAILURE,
            rationale="ok",
            stress_score=-1.0,
        )


def test_report_overall_failure_rate_and_generator_constructor_validation() -> None:
    empty = run_red_team_campaign([], lambda room: True, GenesisPermission(theta=10.0))
    assert empty.overall_failure_rate == pytest.approx(0.0)

    failing = run_red_team_campaign(
        [make_candidate(FailureCategory.ADVERSARIAL_INPUT)],
        lambda room: False,
        GenesisPermission(theta=10.0),
    )
    assert failing.overall_failure_rate == pytest.approx(1.0)

    with pytest.raises(ValueError, match="pass rng or seed"):
        AdversarialGenerator(make_template_room(), rng=np.random.default_rng(0), seed=1)
    with pytest.raises(TypeError, match="template_room must be a Room"):
        AdversarialGenerator(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one dimension"):
        AdversarialGenerator(make_template_room(dim=0))


def test_numerical_generator_single_dimension_and_count_validation() -> None:
    generator = AdversarialGenerator(make_template_room(dim=1), seed=3)
    candidate = generator.generate_numerical_singularity()[0]
    assert candidate.room.x.shape == (1,)
    with pytest.raises(TypeError, match="count must be an integer"):
        generator.generate_boundary_failure(count=True)  # type: ignore[arg-type]


def test_pathological_equation_helpers_cover_clone_and_string_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    import qes.adversarial as adversarial_module

    generator = AdversarialGenerator(make_template_room(), seed=0)
    room = generator._clone_room()
    assert room.memory == generator.template_room.memory

    monkeypatch.setattr(adversarial_module, "EquationAST", None)
    monkeypatch.setattr(adversarial_module, "Operator", None)
    monkeypatch.setattr(adversarial_module, "Constant", None)
    monkeypatch.setattr(adversarial_module, "Variable", None)
    assert generator._pathological_division_equation() == ["1 / x0"]
    assert generator._pathological_log_equation() == ["log(x0 - x0)"]


def test_run_red_team_campaign_validates_inputs() -> None:
    with pytest.raises(TypeError, match="evaluate_fn must be callable"):
        run_red_team_campaign([], object(), GenesisPermission(theta=1.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GenesisPermission"):
        run_red_team_campaign([], lambda room: True, object())  # type: ignore[arg-type]


def test_private_red_team_helpers_cover_remaining_classification_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qes.adversarial as adversarial_module

    class PassedObject:
        passed = True

    assert adversarial_module._evaluation_passed(PassedObject(), None) is True
    assert adversarial_module._evaluation_passed(None, None) is True
    assert adversarial_module._evaluation_passed(object(), None) is True

    candidate = make_candidate(FailureCategory.BOUNDARY_FAILURE)
    permission = GenesisPermission(theta=10.0).evaluate(
        candidate.room.x,
        candidate.room.lower,
        candidate.room.upper,
        w=np.maximum(np.abs(candidate.room.activation), np.finfo(float).eps),
        coupling=candidate.room.couplings,
    )
    assert adversarial_module._infer_failure_category(candidate, permission, {"value": 1.0}, RuntimeError("plain")) is FailureCategory.BOUNDARY_FAILURE
    assert adversarial_module._infer_failure_category(candidate, permission, {"value": np.nan}, None) is FailureCategory.NUMERICAL_SINGULARITY
    assert adversarial_module._coerce_category(FailureCategory.ADVERSARIAL_INPUT) is FailureCategory.ADVERSARIAL_INPUT
    assert adversarial_module._coerce_category("adversarial_input") is FailureCategory.ADVERSARIAL_INPUT
    assert adversarial_module._coerce_category("unknown") is None
    assert adversarial_module._coerce_category(1) is None
    assert adversarial_module._result_is_non_finite({"value": np.nan}) is True
    assert adversarial_module._result_is_non_finite({"value": "x"}) is False
    assert adversarial_module._result_is_non_finite(object()) is False
    assert adversarial_module._numeric_compute_pressure(make_template_room().clone(compute={"cpu": "bad", "mem": np.inf, "ok": 2.0})) == pytest.approx(2.0)

    class RootEquation:
        def __init__(self) -> None:
            self.root = "log(x0)"

    assert adversarial_module._contains_pathological_equation(
        make_template_room().clone(equations=[RootEquation()])
    ) is True
    assert adversarial_module._has_dense_coupling(make_template_room().clone(couplings=None)) is False

    monkeypatch.setattr(adversarial_module, "_verification", None)
    assert adversarial_module._external_failure_category(candidate, permission, None, None) is None

    class VerificationModule:
        @staticmethod
        def classify_failure(**kwargs: object) -> str:
            return "resource_exhaustion"

        @staticmethod
        def classify_red_team_failure(*args: object) -> str:
            return "unstable_state"

        @staticmethod
        def classify_candidate_failure(**kwargs: object) -> object:
            raise RuntimeError("ignore")

    monkeypatch.setattr(adversarial_module, "_verification", VerificationModule)
    assert (
        adversarial_module._external_failure_category(candidate, permission, None, None)
        is FailureCategory.RESOURCE_EXHAUSTION
    )
    assert (
        adversarial_module._infer_failure_category(candidate, permission, None, None)
        is FailureCategory.RESOURCE_EXHAUSTION
    )

    class LegacyOnlyVerification:
        @staticmethod
        def classify_failure(**kwargs: object) -> object:
            raise TypeError("legacy signature")

        @staticmethod
        def classify_red_team_failure(room: Room, evaluation_result: object) -> str:
            return "cascading_failure"

    monkeypatch.setattr(adversarial_module, "_verification", LegacyOnlyVerification)
    assert (
        adversarial_module._external_failure_category(candidate, permission, None, None)
        is FailureCategory.CASCADING_FAILURE
    )

    class InvalidThenValidVerification:
        @staticmethod
        def classify_failure(**kwargs: object) -> str:
            return "not-a-category"

        @staticmethod
        def classify_red_team_failure(**kwargs: object) -> str:
            return "unstable_state"

    monkeypatch.setattr(adversarial_module, "_verification", InvalidThenValidVerification)
    assert (
        adversarial_module._external_failure_category(candidate, permission, None, None)
        is FailureCategory.UNSTABLE_STATE
    )


def test_default_infer_failure_category_falls_back_to_candidate_category() -> None:
    import qes.adversarial as adversarial_module

    candidate = make_candidate(FailureCategory.PATHOLOGICAL_EQUATION, equations=["safe"])
    permission = GenesisPermission(theta=10.0).evaluate(
        candidate.room.x,
        candidate.room.lower,
        candidate.room.upper,
        w=np.maximum(np.abs(candidate.room.activation), np.finfo(float).eps),
        coupling=candidate.room.couplings,
    )
    assert (
        adversarial_module._infer_failure_category(candidate, permission, {"value": 1.0}, RuntimeError("plain"))
        is FailureCategory.PATHOLOGICAL_EQUATION
    )
    room = make_template_room()
    room.couplings = None
    assert adversarial_module._has_dense_coupling(room) is False
