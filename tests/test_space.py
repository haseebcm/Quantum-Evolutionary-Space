import numpy as np

from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace


def make_seed_room(n=2):
    return Room(
        x=np.zeros(n),
        x_star=np.zeros(n),
        lower=-np.ones(n),
        upper=np.ones(n),
        activation=np.ones(n),
    )


def test_add_room_marks_seed_as_active():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    room = make_seed_room()
    space.add_room(room)
    assert room.state == "Active"
    assert room.id in space.rooms
    assert space.total_generated == 1


def test_spawn_adds_multiple_rooms():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    parent = make_seed_room()
    gen = RealityGenerator(rng=np.random.default_rng(0))
    children = gen.branch(parent, count=5, scale=0.01)
    space.spawn(children)
    assert len(space.active_rooms()) == 5


def test_execute_advances_active_room_state():
    space = QESSpace(
        permission_gate=GenesisPermission(theta=1.0),
        step_fn=lambda room, t, dt: room.x + 0.1,
    )
    room = make_seed_room()
    space.add_room(room)
    space.execute()
    np.testing.assert_allclose(room.x, [0.1, 0.1])


def test_execute_with_max_workers_parallelizes_step_fn():
    space = QESSpace(
        permission_gate=GenesisPermission(theta=1.0),
        step_fn=lambda room, t, dt: room.x + 1.0,
        max_workers=4,
    )
    gen = RealityGenerator(rng=np.random.default_rng(0))
    parent = make_seed_room()
    children = gen.branch(parent, count=6, scale=0.0)
    space.spawn(children)
    space.execute()
    for room in space.active_rooms():
        np.testing.assert_allclose(room.x, [1.0, 1.0])


def test_check_permission_collapses_out_of_bounds_room():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    room = make_seed_room()
    room.x = np.array([5.0, 5.0])  # well outside [-1, 1]
    space.add_room(room)
    space.check_permission()
    assert room.state == "Collapsed"
    assert space.total_collapsed == 1


def test_check_permission_keeps_admissible_room_active():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    room = make_seed_room()
    space.add_room(room)
    space.check_permission()
    assert room.state == "Active"
    assert room.weight == 1.0


def test_full_step_runs_execute_divergence_permission_and_convergence():
    space = QESSpace(
        permission_gate=GenesisPermission(theta=1.0),
        step_fn=lambda room, t, dt: room.x,  # stay at origin: fully admissible
    )
    parent = make_seed_room()
    gen = RealityGenerator(rng=np.random.default_rng(1))
    children = gen.branch(parent, count=4, scale=0.01)
    space.spawn(children)

    telemetry = space.step()

    assert telemetry.active == 4
    assert telemetry.collapsed == 0
    assert telemetry.dominant_room_id is not None
    assert 0.0 <= telemetry.convergence <= 1.0


def test_run_executes_requested_number_of_steps():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    space.add_room(make_seed_room())
    history = space.run(3)
    assert len(history) == 3
    assert space.time == 3.0


def test_select_with_no_active_rooms_is_a_no_op():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    space.select(survivors_per_kind=1, signature_fn=lambda r: r.dim)


def test_select_shadows_losers_per_signature_kind():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    winner = make_seed_room()
    winner.weight = 2.0
    loser = make_seed_room()
    loser.weight = 1.0
    space.add_room(winner)
    space.add_room(loser)
    space.select(survivors_per_kind=1, signature_fn=lambda r: r.dim)
    assert winner.state == "Active"
    assert loser.state == "Shadow"


def test_convergence_and_dominant_room_are_defaults_when_empty():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    assert space.convergence() == 0.0
    assert space.dominant_room() is None


def test_collapsed_rooms_are_excluded_from_dominant_room():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    good = make_seed_room()
    bad = make_seed_room()
    bad.x = np.array([10.0, 10.0])
    space.add_room(good)
    space.add_room(bad)
    space.check_permission()
    dominant = space.dominant_room()
    assert dominant.id == good.id


def test_clone_is_independent_copy():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    space.add_room(make_seed_room())
    clone = space.clone()

    assert clone is not space
    clone.step()
    assert clone.time == 1.0
    assert space.time == 0.0
    # Mutating a cloned room must not affect the original.
    clone_room = next(iter(clone.rooms.values()))
    clone_room.x[0] = 99.0
    original_room = next(iter(space.rooms.values()))
    assert original_room.x[0] != 99.0


def test_snapshot_and_restore_roundtrip():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    space.add_room(make_seed_room())
    snapshot = space.snapshot()

    space.step()
    space.step()
    assert space.time == 2.0

    space.restore(snapshot)
    assert space.time == 0.0
    assert space.total_generated == 1
    assert len(space.history) == 0
