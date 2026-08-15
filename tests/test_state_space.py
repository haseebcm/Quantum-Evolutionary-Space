import numpy as np
import pytest

from qes.state_space import MCCStateSpace


def test_dim_matches_vector_length():
    space = MCCStateSpace(lower=[-1, -1, -1], upper=[1, 1, 1], reference=[0, 0, 0])
    assert space.dim == 3


def test_within_envelope_elementwise_mask():
    space = MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0])
    mask = space.within_envelope(np.array([0.5, 2.0]))
    np.testing.assert_array_equal(mask, [True, False])


def test_within_envelope_inclusive_at_boundaries():
    space = MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0])
    mask = space.within_envelope(np.array([-1.0, 1.0]))
    np.testing.assert_array_equal(mask, [True, True])


def test_is_admissible_true_when_all_within_bounds():
    space = MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0])
    assert space.is_admissible(np.array([0.0, 0.5]))


def test_is_admissible_false_when_any_component_out_of_bounds():
    space = MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0])
    assert not space.is_admissible(np.array([0.0, 1.5]))


def test_clip_projects_state_back_into_envelope():
    space = MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0])
    clipped = space.clip(np.array([5.0, -5.0]))
    np.testing.assert_allclose(clipped, [1.0, -1.0])


def test_clip_leaves_admissible_state_unchanged():
    space = MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0])
    x = np.array([0.2, -0.3])
    np.testing.assert_allclose(space.clip(x), x)


def test_constructor_rejects_mismatched_lower_upper_shapes():
    with pytest.raises(ValueError):
        MCCStateSpace(lower=[-1, -1], upper=[1], reference=[0, 0])


def test_constructor_rejects_mismatched_reference_shape():
    with pytest.raises(ValueError):
        MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0, 0])


def test_rooms_may_have_independent_dimensionality():
    space_a = MCCStateSpace(lower=[-1], upper=[1], reference=[0])
    space_b = MCCStateSpace(lower=[-1, -1, -1], upper=[1, 1, 1], reference=[0, 0, 0])
    assert space_a.dim != space_b.dim


def test_volume_computes_bounding_box_measure():
    space = MCCStateSpace(lower=[-1, -2], upper=[1, 2], reference=[0, 0])
    assert space.volume() == pytest.approx(2.0 * 4.0)


def test_named_dimensions_index_and_labeled_lookup():
    space = MCCStateSpace(
        lower=[-1, -1], upper=[1, 1], reference=[0, 0], names=["position", "velocity"]
    )
    assert space.index_of("velocity") == 1
    labeled = space.labeled(np.array([0.5, -0.5]))
    assert labeled == {"position": 0.5, "velocity": -0.5}


def test_named_dimensions_constructor_rejects_mismatched_length():
    with pytest.raises(ValueError):
        MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0], names=["only_one"])


def test_index_of_without_names_raises():
    space = MCCStateSpace(lower=[-1], upper=[1], reference=[0])
    with pytest.raises(ValueError):
        space.index_of("anything")


def test_labeled_without_names_raises():
    space = MCCStateSpace(lower=[-1], upper=[1], reference=[0])
    with pytest.raises(ValueError):
        space.labeled(np.array([0.0]))
