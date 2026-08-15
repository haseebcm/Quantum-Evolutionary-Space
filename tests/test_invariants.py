import numpy as np
import pytest

from qes.invariants import InvariantEngine, InvariantReport, InvariantViolation
from qes.permission import GenesisPermission
from qes.room import Room
from qes.space import QESSpace
from qes.universe import Universe
from qes.world import World


def make_space() -> QESSpace:
    return QESSpace(permission_gate=GenesisPermission(theta=2.0))


def make_room(**overrides) -> Room:
    fields = dict(
        x=np.zeros(2),
        x_star=np.ones(2) * 0.5,
        lower=-np.ones(2),
        upper=np.ones(2),
        activation=np.ones(2),
    )
    fields.update(overrides)
    return Room(**fields)


def test_healthy_room_passes_all_invariants():
    engine = InvariantEngine()
    report = engine.check_room(make_room())
    assert report.passed
    assert report.violations == []
    assert bool(report) is True


def test_non_finite_state_fails_state_validity():
    room = make_room(x=np.array([0.0, np.nan]))
    report = InvariantEngine().check_room(room)
    assert not report.passed
    assert any(v.check == "state_validity" for v in report.violations)


def test_inconsistent_shapes_fail_dimensional_consistency():
    room = make_room(x=np.zeros(3))
    report = InvariantEngine().check_room(room)
    assert not report.passed
    assert any(v.check == "dimensional_consistency" for v in report.violations)


def test_inverted_bounds_fail():
    room = make_room(lower=np.ones(2), upper=-np.ones(2))
    report = InvariantEngine().check_room(room)
    assert not report.passed
    assert any(v.check == "bounds" for v in report.violations)


def test_self_referential_lineage_fails():
    room = make_room()
    room.lineage = [room.id]
    report = InvariantEngine().check_room(room)
    assert not report.passed
    assert any(v.check == "lineage_integrity" for v in report.violations)


def test_duplicate_lineage_fails():
    room = make_room()
    room.lineage = ["R-00001", "R-00001"]
    report = InvariantEngine().check_room(room)
    assert not report.passed
    assert any(v.check == "lineage_integrity" for v in report.violations)


def test_negative_compute_fails_resource_consistency():
    room = make_room(compute={"cpu": -1.0})
    report = InvariantEngine().check_room(room)
    assert not report.passed
    assert any(v.check == "resource_consistency" for v in report.violations)


def test_negative_weight_fails_probability_normalization():
    room = make_room(weight=-0.5)
    report = InvariantEngine().check_room(room)
    assert not report.passed
    assert any(v.check == "probability_normalization" for v in report.violations)


def test_unrecognized_lifecycle_state_fails():
    room = make_room()
    room.state = "NotARealState"
    report = InvariantEngine().check_room(room)
    assert not report.passed
    assert any(v.check == "lifecycle_consistency" for v in report.violations)


def test_check_space_aggregates_every_room():
    space = make_space()
    space.spawn([make_room(), make_room(x=np.array([0.0, np.nan]))])
    report = InvariantEngine().check_space(space)
    assert not report.passed
    assert len(report.violations) == 1


def test_check_world_recurses_into_space_and_nested_universe():
    space = make_space()
    space.spawn([make_room(x=np.array([np.inf, 0.0]))])
    world = World(name="w", space=space)

    nested_universe = Universe()
    nested_space = make_space()
    nested_space.spawn([make_room(lower=np.ones(2), upper=-np.ones(2))])
    nested_world = World(name="inner", space=nested_space)
    nested_universe.add_world(nested_world)
    world.set_nested_universe(nested_universe)

    report = InvariantEngine().check_world(world)
    assert not report.passed
    checks = {v.check for v in report.violations}
    assert "state_validity" in checks
    assert "bounds" in checks


def test_check_universe_recurses_into_all_worlds():
    universe = Universe()
    space1 = make_space()
    space1.spawn([make_room()])
    universe.add_world(World(name="a", space=space1))

    space2 = make_space()
    space2.spawn([make_room(x=np.array([0.0, np.nan]))])
    universe.add_world(World(name="b", space=space2))

    report = InvariantEngine().check_universe(universe)
    assert not report.passed
    assert len(report.violations) == 1


def test_dispatch_check_routes_by_type():
    engine = InvariantEngine()
    assert engine.check(make_room()).passed
    space = make_space()
    space.spawn([make_room()])
    assert engine.check(space).passed
    assert engine.check(World(name="w", space=space)).passed
    universe = Universe()
    universe.add_world(World(name="w", space=space))
    assert engine.check(universe).passed


def test_check_rejects_unsupported_type():
    with pytest.raises(TypeError):
        InvariantEngine().check(object())


def test_report_summary_reflects_pass_and_fail():
    passing = InvariantReport(passed=True)
    assert "OK" in passing.summary()
    failing = InvariantReport(
        passed=False,
        violations=[InvariantViolation(scope="room", check="bounds", message="bad")],
    )
    assert "FAILED" in failing.summary()
    assert "bounds=1" in failing.summary()


def test_report_merge_combines_violations():
    v1 = InvariantViolation(scope="room", check="a", message="m1")
    v2 = InvariantViolation(scope="room", check="b", message="m2")
    r1 = InvariantReport(passed=False, violations=[v1])
    r2 = InvariantReport(passed=False, violations=[v2])
    merged = r1.merge(r2)
    assert merged.violations == [v1, v2]
    assert not merged.passed


def test_couplings_none_skips_shape_check_but_bad_shape_fails():
    room = make_room()
    room.couplings = None
    assert InvariantEngine().check_room(room).passed

    room = make_room()
    room.couplings = np.zeros((1, 1))
    report = InvariantEngine().check_room(room)
    assert any("couplings shape" in violation.message for violation in report.violations)


def test_resource_consistency_and_weight_require_numeric_finite_values():
    non_numeric = make_room(compute={"cpu": object()})
    non_finite = make_room(compute={"cpu": float("inf")})
    bad_weight = make_room(weight=float("nan"))

    non_numeric_report = InvariantEngine().check_room(non_numeric)
    non_finite_report = InvariantEngine().check_room(non_finite)
    bad_weight_report = InvariantEngine().check_room(bad_weight)

    assert any("not numeric" in violation.message for violation in non_numeric_report.violations)
    assert any("not finite" in violation.message for violation in non_finite_report.violations)
    assert any("weight=" in violation.message for violation in bad_weight_report.violations)


def test_non_negative_finite_compute_allocation_is_accepted():
    report = InvariantEngine().check_room(make_room(compute={"cpu": 1.0}))
    assert report.passed


def test_check_world_handles_nested_universe_without_space():
    nested_universe = Universe()
    nested_space = make_space()
    nested_space.spawn([make_room(weight=float("nan"))])
    nested_universe.add_world(World(name="inner", space=nested_space))

    world = World(name="outer", space=None)
    world.set_nested_universe(nested_universe)

    report = InvariantEngine().check_world(world)

    assert not report.passed
    assert any(v.check == "probability_normalization" for v in report.violations)
