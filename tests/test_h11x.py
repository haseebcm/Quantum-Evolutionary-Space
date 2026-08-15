import pytest

from qes.h11x import (
    H11X,
    CorrectionOutcome,
    Geometry,
    constraint_geometry,
    domain_gatekeeper,
    export_barrier,
    failure_anticipation,
    necessity_field,
    self_generative_correction,
)


def test_necessity_field():
    assert necessity_field(0.0) is True
    assert necessity_field(1.0) is True
    assert necessity_field(-0.1) is False


def test_constraint_geometry_and_severity():
    geometry = constraint_geometry(1.0, 2.0, 3.0, 0.5)
    assert isinstance(geometry, Geometry)
    assert geometry.severity() == pytest.approx(1.0 + 2.0 + 3.0 - 0.5)


def test_domain_gatekeeper_all_pass():
    assert domain_gatekeeper({"material": True, "thermal": True}) is True


def test_domain_gatekeeper_one_fails():
    assert domain_gatekeeper({"material": True, "thermal": False}) is False


def test_domain_gatekeeper_empty_is_not_admissible():
    assert domain_gatekeeper({}) is False


def test_failure_anticipation_recovery_exists():
    assert failure_anticipation(lambda: 10.0, lambda v: v < 100.0) is True


def test_failure_anticipation_no_recovery():
    assert failure_anticipation(lambda: 200.0, lambda v: v < 100.0) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.5", True),
        (float("inf"), ValueError),
        ("not-a-number", TypeError),
    ],
)
def test_necessity_field_validates_scalar_inputs(value, expected):
    if expected is True:
        assert necessity_field(value) is True
    else:
        with pytest.raises(expected):
            necessity_field(value)


def test_domain_gatekeeper_validates_mapping_contents():
    with pytest.raises(TypeError, match="dict\\[str, bool\\]"):
        domain_gatekeeper([("material", True)])
    with pytest.raises(TypeError, match="keys must be strings"):
        domain_gatekeeper({1: True})
    with pytest.raises(TypeError, match="must be a bool"):
        domain_gatekeeper({"material": 1})


def test_callbacks_must_be_callable():
    with pytest.raises(TypeError, match="push_to_failure_fn must be callable"):
        failure_anticipation("boom", lambda value: True)
    with pytest.raises(TypeError, match="derive_fn must be callable"):
        export_barrier({"x": 1}, derive_fn="nope")
    with pytest.raises(TypeError, match="relax_fn must be callable"):
        self_generative_correction(
            failed_state=1,
            reintegrate_fn=lambda s: s + 1,
            relax_fn=None,
            reform_fn=lambda s: s,
        )


def test_self_generative_correction_chains_stages():
    outcome = self_generative_correction(
        failed_state=1,
        reintegrate_fn=lambda s: s + 1,
        relax_fn=lambda s: s * 2,
        reform_fn=lambda s: s - 1,
    )
    assert isinstance(outcome, CorrectionOutcome)
    assert outcome.reintegrated == 2
    assert outcome.relaxed == 4
    assert outcome.reformed == 3


def test_export_barrier_never_exposes_internal_state():
    result = export_barrier({"secret": 42}, derive_fn=lambda s: s["secret"] * 2)
    assert result.derived == 84
    assert not hasattr(result, "internal_state")


class TestH11X:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            n=1.0,
            stress=1.0,
            load_paths=1.0,
            energy_flow=1.0,
            temporal_stability=1.0,
            domain_checks={"material": True},
            push_to_failure_fn=lambda: 10.0,
            recovery_check_fn=lambda v: v < 100.0,
            internal_state={"x": 1},
            derive_fn=lambda s: s["x"] * 10,
        )
        kwargs.update(overrides)
        return kwargs

    def test_denied_at_layer_1(self):
        result = H11X().evaluate(**self._base_kwargs(n=-1.0))
        assert result.admitted is False
        assert result.denied_at_layer == 1
        assert result.geometry is None

    def test_denied_at_layer_3(self):
        result = H11X().evaluate(**self._base_kwargs(domain_checks={"material": False}))
        assert result.admitted is False
        assert result.denied_at_layer == 3
        assert result.geometry is not None

    def test_denied_at_layer_4_without_correction_callbacks(self):
        result = H11X().evaluate(
            **self._base_kwargs(push_to_failure_fn=lambda: 200.0, recovery_check_fn=lambda v: v < 100.0)
        )
        assert result.admitted is False
        assert result.denied_at_layer == 4
        assert result.correction is None

    def test_layer_4_failure_recovers_via_correction_then_admits(self):
        result = H11X().evaluate(
            **self._base_kwargs(
                push_to_failure_fn=lambda: 200.0,
                recovery_check_fn=lambda v: v < 100.0,
                reintegrate_fn=lambda s: {"x": s["x"] + 1},
                relax_fn=lambda s: {"x": s["x"] * 2},
                reform_fn=lambda s: {"x": s["x"] - 1},
            )
        )
        assert result.admitted is True
        assert result.correction is not None
        assert result.export is not None
        assert result.export.derived == (((1 + 1) * 2) - 1) * 10

    def test_full_pass_admits_without_correction(self):
        result = H11X().evaluate(**self._base_kwargs())
        assert result.admitted is True
        assert result.correction is None
        assert result.export.derived == 10

    def test_evaluate_validates_required_callbacks_before_layer_checks(self):
        with pytest.raises(TypeError, match="derive_fn must be callable"):
            H11X().evaluate(**self._base_kwargs(derive_fn=None))
