import numpy as np
import pytest

from qes.room import Room
from qes.state_space import MCCStateSpace


def make_room(n=3):
    return Room(
        x=np.zeros(n),
        x_star=np.zeros(n),
        lower=-np.ones(n),
        upper=np.ones(n),
        activation=np.ones(n),
    )


def test_room_effective_state_applies_activation():
    room = make_room(3)
    room.x = np.array([1.0, 2.0, 3.0])
    room.activation = np.array([1.0, 0.0, 0.5])
    np.testing.assert_allclose(room.effective_state(), [1.0, 0.0, 1.5])


def test_room_clone_extends_lineage_and_copies_arrays():
    parent = make_room(2)
    parent.x = np.array([0.1, 0.2])
    child = parent.clone(x=np.array([0.3, 0.4]))
    assert child.lineage == [parent.id]
    assert child.id != parent.id
    # mutating child's arrays must not affect the parent
    child.x[0] = 99.0
    assert parent.x[0] != 99.0


def test_room_dim_matches_state_length():
    room = make_room(5)
    assert room.dim == 5


def test_room_generation_reflects_lineage_depth():
    parent = make_room(2)
    assert parent.generation == 0
    child = parent.clone()
    assert child.generation == 1
    grandchild = child.clone()
    assert grandchild.generation == 2


def test_room_tag_and_get_tag_roundtrip():
    room = make_room(2)
    room.tag("family", "control-search")
    assert room.get_tag("family") == "control-search"
    assert room.get_tag("missing", default="none") == "none"


def test_mcc_state_space_bounds():
    space = MCCStateSpace(lower=[-1, -1], upper=[1, 1], reference=[0, 0])
    assert space.is_admissible(np.array([0.5, -0.5]))
    assert not space.is_admissible(np.array([2.0, 0.0]))
    clipped = space.clip(np.array([5.0, -5.0]))
    np.testing.assert_allclose(clipped, [1.0, -1.0])


def test_mcc_state_space_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        MCCStateSpace(lower=[-1, -1], upper=[1], reference=[0, 0])
