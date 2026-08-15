import numpy as np

from qes.reality_generator import RealityGenerator
from qes.room import Room


def make_room(n=2):
    return Room(
        x=np.zeros(n),
        x_star=np.zeros(n),
        lower=-np.ones(n),
        upper=np.ones(n),
        activation=np.ones(n),
    )


def test_branch_creates_requested_number_of_children():
    gen = RealityGenerator(rng=np.random.default_rng(0))
    parent = make_room()
    children = gen.branch(parent, count=5)
    assert len(children) == 5
    assert all(c.state == "Active" for c in children)


def test_branch_children_have_parent_in_lineage():
    gen = RealityGenerator(rng=np.random.default_rng(0))
    parent = make_room()
    children = gen.branch(parent, count=3)
    for child in children:
        assert parent.id in child.lineage


def test_branch_perturbs_state_away_from_parent():
    gen = RealityGenerator(rng=np.random.default_rng(0))
    parent = make_room()
    children = gen.branch(parent, count=3, scale=0.1)
    assert any(not np.allclose(c.x, parent.x) for c in children)


def test_branch_cycles_through_provided_equations():
    gen = RealityGenerator(rng=np.random.default_rng(0))
    parent = make_room()
    equations = [["eqA"], ["eqB"]]
    children = gen.branch(parent, count=4, equations=equations)
    assert children[0].equations == ["eqA"]
    assert children[1].equations == ["eqB"]
    assert children[2].equations == ["eqA"]


def test_branch_cycles_through_provided_activations():
    gen = RealityGenerator(rng=np.random.default_rng(0))
    parent = make_room()
    activations = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    children = gen.branch(parent, count=3, activations=activations)
    np.testing.assert_allclose(children[0].activation, [1.0, 0.0])
    np.testing.assert_allclose(children[1].activation, [0.0, 1.0])
    np.testing.assert_allclose(children[2].activation, [1.0, 0.0])


def test_branch_uses_custom_perturb_fn():
    gen = RealityGenerator()
    parent = make_room()
    children = gen.branch(parent, count=2, perturb_fn=lambda r: np.array([1.0, 1.0]))
    for child in children:
        np.testing.assert_allclose(child.x, [1.0, 1.0])


def test_latin_hypercube_samples_cover_each_stratum_per_axis():
    gen = RealityGenerator(rng=np.random.default_rng(0))
    lower = np.array([0.0, 0.0])
    upper = np.array([10.0, 10.0])
    samples = gen.latin_hypercube_samples(dim=2, count=5, lower=lower, upper=upper)
    assert samples.shape == (5, 2)
    for d in range(2):
        strata = np.floor(samples[:, d] / 2.0).astype(int)
        assert sorted(strata.tolist()) == [0, 1, 2, 3, 4]
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)


def test_branch_latin_hypercube_creates_active_children_within_room_bounds():
    gen = RealityGenerator(rng=np.random.default_rng(0))
    parent = make_room()
    children = gen.branch_latin_hypercube(parent, count=4)
    assert len(children) == 4
    assert all(c.state == "Active" for c in children)
    for c in children:
        assert np.all(c.x >= parent.lower)
        assert np.all(c.x <= parent.upper)
        assert parent.id in c.lineage
