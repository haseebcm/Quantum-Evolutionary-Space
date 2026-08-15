import numpy as np
import pytest

from qes.reality_operators import (
    RealityFamily,
    RealityFamilyBuilder,
    crossover,
    dimensional_transformation,
    equation_substitution,
    extrapolate,
    interpolate,
    inversion,
    mutation,
    parameter_transformation,
    perturbation,
    topology_transformation,
)
from qes.room import Room


def make_room(dim: int = 3, **overrides) -> Room:
    fields = dict(
        x=np.zeros(dim),
        x_star=np.ones(dim) * 0.5,
        lower=-np.ones(dim),
        upper=np.ones(dim),
        activation=np.ones(dim),
    )
    fields.update(overrides)
    return Room(**fields)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Individual operators
# ---------------------------------------------------------------------------


def test_mutation_perturbs_state_and_preserves_lineage(rng):
    room = make_room()
    child = mutation(room, rng, scale=0.1)
    assert child.dim == room.dim
    assert not np.array_equal(child.x, room.x)
    assert child.lineage == [room.id]


def test_mutation_rejects_negative_scale(rng):
    with pytest.raises(ValueError):
        mutation(make_room(), rng, scale=-0.1)


def test_crossover_blends_parent_states(rng):
    a = make_room(x=np.zeros(3))
    b = make_room(x=np.ones(3))
    child = crossover(a, b, rng, alpha=0.25)
    np.testing.assert_allclose(child.x, np.full(3, 0.75))
    assert a.id in child.lineage and b.id in child.lineage


def test_crossover_requires_matching_dimensionality(rng):
    a = make_room(dim=2)
    b = make_room(dim=3)
    with pytest.raises(ValueError):
        crossover(a, b, rng)


def test_crossover_rejects_alpha_out_of_range(rng):
    a, b = make_room(), make_room()
    with pytest.raises(ValueError):
        crossover(a, b, rng, alpha=1.5)


def test_interpolate_endpoints_match_parents():
    a = make_room(x=np.zeros(3))
    b = make_room(x=np.ones(3))
    np.testing.assert_allclose(interpolate(a, b, 0.0).x, a.x)
    np.testing.assert_allclose(interpolate(a, b, 1.0).x, b.x)
    np.testing.assert_allclose(interpolate(a, b, 0.5).x, np.full(3, 0.5))


def test_interpolate_rejects_t_outside_unit_interval():
    a, b = make_room(), make_room()
    with pytest.raises(ValueError):
        interpolate(a, b, 1.5)


def test_extrapolate_projects_beyond_endpoints():
    a = make_room(x=np.zeros(3))
    b = make_room(x=np.ones(3))
    np.testing.assert_allclose(extrapolate(a, b, 2.0).x, np.full(3, 2.0))
    np.testing.assert_allclose(extrapolate(a, b, -1.0).x, np.full(3, -1.0))


def test_inversion_reflects_through_center():
    room = make_room(x=np.zeros(3), x_star=np.ones(3))
    reflected = inversion(room)
    np.testing.assert_allclose(reflected.x, np.full(3, 2.0))
    reflected_custom = inversion(room, center=np.zeros(3))
    np.testing.assert_allclose(reflected_custom.x, np.zeros(3))


def test_perturbation_falls_back_to_mutation_without_directions(rng):
    room = make_room()
    child = perturbation(room, rng, scale=0.1)
    assert not np.array_equal(child.x, room.x)


def test_perturbation_uses_structured_directions(rng):
    room = make_room(x=np.zeros(3))
    directions = np.eye(3)
    child = perturbation(room, rng, scale=0.1, directions=directions)
    assert child.dim == 3


def test_perturbation_rejects_wrong_direction_shape(rng):
    room = make_room(dim=3)
    with pytest.raises(ValueError):
        perturbation(room, rng, directions=np.eye(2))


def test_dimensional_transformation_scales_state_and_bounds():
    room = make_room(x=np.ones(2), x_star=np.ones(2) * 0.5, lower=-np.ones(2), upper=np.ones(2))
    child = dimensional_transformation(room, np.eye(2) * 2.0)
    np.testing.assert_allclose(child.x, np.full(2, 2.0))
    np.testing.assert_allclose(child.lower, np.full(2, -2.0))
    np.testing.assert_allclose(child.upper, np.full(2, 2.0))


def test_dimensional_transformation_rejects_wrong_shape():
    room = make_room(dim=3)
    with pytest.raises(ValueError):
        dimensional_transformation(room, np.eye(2))


def test_topology_transformation_permutes_dimensions():
    room = make_room(dim=3, x=np.array([1.0, 2.0, 3.0]))
    child = topology_transformation(room, [2, 0, 1])
    np.testing.assert_allclose(child.x, np.array([3.0, 1.0, 2.0]))


def test_topology_transformation_rejects_invalid_permutation():
    room = make_room(dim=3)
    with pytest.raises(ValueError):
        topology_transformation(room, [0, 0, 1])


def test_equation_substitution_replaces_equations():
    room = make_room(equations=["old"])
    child = equation_substitution(room, ["new1", "new2"])
    assert child.equations == ["new1", "new2"]
    assert room.equations == ["old"]  # original untouched


def test_parameter_transformation_applies_theta_fn():
    room = make_room(theta={"a": 1.0})
    child = parameter_transformation(room, lambda t: {**t, "b": 2.0})
    assert child.theta == {"a": 1.0, "b": 2.0}
    assert room.theta == {"a": 1.0}  # original untouched


def test_parameter_transformation_rejects_non_dict_result():
    room = make_room()
    with pytest.raises(TypeError):
        parameter_transformation(room, lambda t: "not a dict")


# ---------------------------------------------------------------------------
# RealityFamily / RealityFamilyBuilder
# ---------------------------------------------------------------------------


def test_reality_family_statistics():
    members = [make_room(x=np.zeros(3)), make_room(x=np.ones(3))]
    family = RealityFamily(members=members, parent_id="R-parent")
    np.testing.assert_allclose(family.mean_state(), np.full(3, 0.5))
    assert family.diversity() > 0.0
    assert len(family) == 2
    assert list(family) == members


def test_reality_family_best_uses_score_fn():
    members = [make_room(x=np.array([5.0, 5.0, 5.0])), make_room(x=np.zeros(3))]
    family = RealityFamily(members=members)
    best = family.best(score_fn=lambda r: float(np.sum(np.abs(r.x))))
    np.testing.assert_allclose(best.x, np.zeros(3))


def test_reality_family_empty_raises():
    family = RealityFamily(members=[])
    with pytest.raises(ValueError):
        family.mean_state()
    with pytest.raises(ValueError):
        family.best(score_fn=lambda r: 0.0)
    assert family.diversity() == 0.0


def test_reality_family_builder_mutation_only(rng):
    parent = make_room()
    builder = RealityFamilyBuilder(rng=rng)
    family = builder.build(parent, count=5)
    assert len(family) == 5
    assert family.parent_id == parent.id
    assert all(m.state == "Active" for m in family)


def test_reality_family_builder_with_second_parent(rng):
    parent = make_room(x=np.zeros(3))
    other = make_room(x=np.ones(3))
    builder = RealityFamilyBuilder(rng=rng)
    family = builder.build(parent, count=10, second_parent=other)
    assert len(family) == 10
