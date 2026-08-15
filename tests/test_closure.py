import numpy as np
import pytest

from qes.closure import (
    CrossDomainVerifier,
    PermissionClosure,
    action_selection,
    collapse_proximity,
    divergence,
    divergence_gradient,
    safe_exploration_closure,
)


def test_divergence_zero_when_x_in_viable_set():
    viable = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    assert divergence(np.array([1.0, 1.0]), viable) == pytest.approx(0.0)


def test_divergence_positive_distance():
    viable = [np.array([0.0, 0.0])]
    assert divergence(np.array([3.0, 4.0]), viable) == pytest.approx(5.0)


def test_divergence_raises_on_empty_viable_set():
    with pytest.raises(ValueError):
        divergence(np.array([0.0]), [])


def test_divergence_rejects_mismatched_viable_point_shape():
    with pytest.raises(ValueError, match="does not match x shape"):
        divergence(np.array([0.0, 1.0]), [np.array([0.0])])


def test_divergence_gradient_shape_and_sign():
    viable = [np.array([0.0, 0.0])]
    grad = divergence_gradient(np.array([3.0, 4.0]), viable)
    assert grad.shape == (2,)
    # Gradient should point away from the viable point (positive components).
    assert grad[0] > 0
    assert grad[1] > 0


def test_collapse_proximity_additive_combination():
    assert collapse_proximity(1.0, 2.0, 3.0) == pytest.approx(6.0)


class TestPermissionClosure:
    def test_evaluate_filters_admissible_actions(self):
        closure = PermissionClosure(
            candidate_actions=[1, 2, 3, 4],
            constraint_fn=lambda x, u: u - 2,  # admissible iff u <= 2
        )
        result = closure.evaluate(x=None)
        assert result.admissible_actions == [1, 2]
        assert result.is_safe is True

    def test_evaluate_empty_admissible_set_is_unsafe(self):
        closure = PermissionClosure(
            candidate_actions=[5, 6],
            constraint_fn=lambda x, u: u,  # never <= 0
        )
        result = closure.evaluate(x=None)
        assert result.admissible_actions == []
        assert result.is_safe is False


def test_action_selection_picks_argmin():
    chosen = action_selection(
        x=0,
        actions=[1, 2, 3],
        j_fn=lambda x, u: u,
        s_fn=lambda x: 0.0,
        transition_fn=lambda x, u: x + u,
        lam=1.0,
    )
    assert chosen == 1


def test_action_selection_raises_on_empty_actions():
    with pytest.raises(ValueError):
        action_selection(
            x=0,
            actions=[],
            j_fn=lambda x, u: u,
            s_fn=lambda x: 0.0,
            transition_fn=lambda x, u: x,
        )


def test_safe_exploration_closure_true_when_mapping_stays_admissible():
    result = safe_exploration_closure(
        policies=[1, 2, 3],
        search_operator=lambda p: p + 10,
        is_admissible_fn=lambda p: p < 20,
    )
    assert result is True


def test_safe_exploration_closure_false_when_mapping_escapes_admissible():
    result = safe_exploration_closure(
        policies=[1, 15],
        search_operator=lambda p: p + 10,
        is_admissible_fn=lambda p: p < 20,
    )
    assert result is False


class TestCrossDomainVerifier:
    def _make(self):
        return CrossDomainVerifier(
            encoder=lambda y: np.asarray(y, dtype=float),
            decoder=lambda x: x,
            transition_fn=lambda x, u, xi: x + u,
            state_space_check=lambda x: bool(np.all(x < 100)),
            constraint_fn=lambda x, u: u - 5,
        )

    def test_verify_uses_default_error_fn(self):
        verifier = self._make()
        result = verifier.verify(y=[1.0, 2.0], u=1, xi=None)
        assert result.reconstruction_error == pytest.approx(0.0)
        assert result.state_closed is True
        assert result.constraint_closed is True

    def test_verify_constraint_closed_reflects_meaningful_check(self):
        verifier = self._make()
        result = verifier.verify(y=[1.0, 2.0], u=10, xi=None)
        assert result.constraint_closed is False

    def test_verify_state_closed_false_when_out_of_bounds(self):
        verifier = self._make()
        result = verifier.verify(y=[50.0, 60.0], u=60, xi=None)
        assert result.state_closed is False

    def test_verify_custom_error_fn(self):
        verifier = self._make()
        result = verifier.verify(
            y=[1.0, 2.0], u=1, xi=None, error_fn=lambda a, b: 42.0
        )
        assert result.reconstruction_error == 42.0

    def test_risk_monotonicity_true_for_nondecreasing(self):
        assert CrossDomainVerifier.risk_monotonicity([0.1, 0.2, 0.5, 0.5, 1.0]) is True

    def test_risk_monotonicity_false_for_decrease(self):
        assert CrossDomainVerifier.risk_monotonicity([0.5, 0.4, 0.6]) is False

    def test_risk_monotonicity_single_element_true(self):
        assert CrossDomainVerifier.risk_monotonicity([0.5]) is True


def test_internal_vector_and_scalar_validators_reject_bad_inputs():
    import qes.closure as closure_module

    with pytest.raises(TypeError, match="one-dimensional float array"):
        closure_module._as_finite_vector(object(), name="x")
    with pytest.raises(ValueError, match="one-dimensional"):
        closure_module._as_finite_vector([[1.0, 2.0]], name="x")
    with pytest.raises(ValueError, match="finite values"):
        closure_module._as_finite_vector([1.0, np.nan], name="x")
    with pytest.raises(TypeError, match="real-valued scalar"):
        closure_module._as_finite_scalar(object(), name="value")
    with pytest.raises(ValueError, match="finite"):
        closure_module._as_finite_scalar(float("inf"), name="value")
    with pytest.raises(ValueError, match=">= 0.0"):
        closure_module._as_finite_scalar(-1.0, name="value", minimum=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        closure_module._as_finite_scalar(2.0, name="value", maximum=1.0)


def test_divergence_gradient_covers_zero_eps_and_empty_vector_cases():
    with pytest.raises(ValueError, match="eps must be > 0"):
        divergence_gradient(np.array([1.0]), [np.array([0.0])], eps=0.0)
    np.testing.assert_allclose(
        divergence_gradient(np.array([]), [np.array([])], eps=1e-3),
        np.array([]),
    )


def test_permission_closure_requires_callable_constraint():
    with pytest.raises(TypeError, match="constraint_fn must be callable"):
        PermissionClosure(candidate_actions=[1], constraint_fn=object())  # type: ignore[arg-type]


def test_action_selection_validates_callbacks_and_defensive_empty_iteration():
    class TruthyEmptySequence:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> object:
            raise IndexError

    with pytest.raises(TypeError, match="j_fn must be callable"):
        action_selection(0, [1], object(), lambda x: 0.0, lambda x, u: x)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="s_fn must be callable"):
        action_selection(0, [1], lambda x, u: 0.0, object(), lambda x, u: x)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="transition_fn must be callable"):
        action_selection(0, [1], lambda x, u: 0.0, lambda x: 0.0, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be non-empty"):
        action_selection(
            x=0,
            actions=TruthyEmptySequence(),  # type: ignore[arg-type]
            j_fn=lambda x, u: 0.0,
            s_fn=lambda x: 0.0,
            transition_fn=lambda x, u: x,
        )


def test_safe_exploration_closure_requires_callables():
    with pytest.raises(TypeError, match="search_operator must be callable"):
        safe_exploration_closure([1], object(), lambda x: True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="is_admissible_fn must be callable"):
        safe_exploration_closure([1], lambda x: x, object())  # type: ignore[arg-type]


def test_default_error_fn_and_verifier_error_paths():
    import qes.closure as closure_module

    with pytest.raises(ValueError, match="does not match"):
        closure_module._default_error_fn([1.0], [1.0, 2.0])

    verifier = CrossDomainVerifier(
        encoder=lambda y: np.asarray(y, dtype=float),
        decoder=lambda x: x,
        transition_fn=lambda x, u, xi: np.asarray([1.0, 2.0, 3.0]),
        state_space_check=lambda x: True,
        constraint_fn=lambda x, u: 0.0,
    )
    with pytest.raises(TypeError, match="error_fn must be callable"):
        verifier.verify([1.0, 2.0], 1, None, error_fn=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match"):
        verifier.verify([1.0, 2.0], 1, None)


def test_cross_domain_verifier_constructor_and_risk_monotonicity_validation():
    with pytest.raises(TypeError, match="encoder must be callable"):
        CrossDomainVerifier(  # type: ignore[arg-type]
            encoder=object(),
            decoder=lambda x: x,
            transition_fn=lambda x, u, xi: x,
            state_space_check=lambda x: True,
            constraint_fn=lambda x, u: 0.0,
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        CrossDomainVerifier.risk_monotonicity([[0.1, 0.2]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="finite values"):
        CrossDomainVerifier.risk_monotonicity([0.1, np.nan])
