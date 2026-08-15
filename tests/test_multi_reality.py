import numpy as np
import pytest

from qes.multi_reality import (
    VQCE,
    PatternReplicationLayer,
    SimulationEngine,
    VirtualNode,
    VirtualNodeProjection,
)
from qes.patterns import Pattern, PatternMemory
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


def test_simulation_engine_evolves_independent_copies():
    room = make_room(2)
    engine = SimulationEngine()

    def step_a(r, t, dt):
        return r.x + 1.0

    def step_b(r, t, dt):
        return r.x + 2.0

    results = engine.simulate(room, [step_a, step_b], steps=3, dt=1.0)
    np.testing.assert_allclose(results[0].x, [3.0, 3.0])
    np.testing.assert_allclose(results[1].x, [6.0, 6.0])
    # source room untouched
    np.testing.assert_allclose(room.x, [0.0, 0.0])


def test_virtual_node_projection_creates_unique_nodes():
    room = make_room(2)
    projection = VirtualNodeProjection(generator=RealityGenerator(rng=np.random.default_rng(0)))
    nodes = projection.project(room, count=4, scale=0.1)
    assert len(nodes) == 4
    ids = {node.id for node in nodes}
    assert len(ids) == 4
    for node in nodes:
        assert node.room.dim == 2


def test_pattern_replication_layer_seeds_all_nodes():
    room = make_room(2)
    projection = VirtualNodeProjection()
    nodes = projection.project(room, count=3)
    memory = PatternMemory()
    layer = PatternReplicationLayer(memory=memory)
    pattern = Pattern(intent="explore", context={}, payload={"theta": 1.0})

    count = layer.replicate(pattern, nodes)
    assert count == 3
    for node in nodes:
        assert pattern.id in node.room.memory["patterns"]
    assert memory.all_patterns("explore") == [pattern]


def test_pattern_replication_layer_best_for():
    memory = PatternMemory()
    layer = PatternReplicationLayer(memory=memory)
    pattern = Pattern(intent="explore", context={}, payload=1)
    memory.store(pattern)
    assert layer.best_for("explore") is not None
    assert layer.best_for("missing") is None


def test_vqce_rejects_non_positive_h11():
    with pytest.raises(ValueError):
        VQCE(h11=0.0)


def test_vqce_allocate_and_compute_stable():
    vqce = VQCE(h11=10.0, stability_tolerance=1.0)
    key = vqce.allocate_state(dim=2)
    result = vqce.compute(key, np.array([1.0, 1.0]), compute_fn=lambda x: x * 2)
    assert result.validated is True
    assert result.stabilized is True
    np.testing.assert_allclose(result.state, [2.0, 2.0])


def test_vqce_compute_stabilizes_out_of_bounds_state():
    vqce = VQCE(h11=1.0, stability_tolerance=1.0)
    key = vqce.allocate_state(dim=2)
    result = vqce.compute(key, np.array([100.0, 100.0]), compute_fn=lambda x: x)
    assert result.stabilized is False
    assert np.linalg.norm(result.state) <= 1.0 + 1e-6


def test_vqce_compute_unknown_state_raises():
    vqce = VQCE()
    with pytest.raises(KeyError):
        vqce.compute("missing", np.zeros(2), compute_fn=lambda x: x)


def test_vqce_compute_with_failing_validate_fn():
    vqce = VQCE(h11=10.0)
    key = vqce.allocate_state(dim=2)
    result = vqce.compute(
        key, np.array([1.0, 1.0]), compute_fn=lambda x: x, validate_fn=lambda a, b: False
    )
    assert result.validated is False


def test_simulation_engine_validates_inputs():
    engine = SimulationEngine()
    room = make_room(2)
    with pytest.raises(TypeError, match="room must be a Room"):
        engine.simulate("not-a-room", [], steps=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="steps must be an integer"):
        engine.simulate(room, [], steps=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps must be >= 0"):
        engine.simulate(room, [], steps=-1)
    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        engine.simulate(room, [], steps=1, dt=0.0)
    with pytest.raises(TypeError, match="step_fns must contain callables"):
        engine.simulate(room, [123], steps=1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="step result shape must match"):
        engine.simulate(room, [lambda r, t, dt: np.zeros(3)], steps=1)
    with pytest.raises(ValueError, match="step result must be a one-dimensional array"):
        engine.simulate(room, [lambda r, t, dt: np.zeros((2, 1))], steps=1)
    with pytest.raises(ValueError, match="step result must contain only finite values"):
        engine.simulate(room, [lambda r, t, dt: np.array([0.0, np.inf])], steps=1)


def test_virtual_node_projection_and_replication_validate_inputs():
    room = make_room(2)
    projection = VirtualNodeProjection()
    with pytest.raises(TypeError, match="room must be a Room"):
        projection.project("not-a-room", count=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="count must be an integer"):
        projection.project(room, count=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scale must be finite and >= 0"):
        projection.project(room, count=1, scale=-0.1)
    assert projection.project(room, count=0) == []
    with pytest.raises(TypeError, match="room must be a Room"):
        VirtualNode(room="not-a-room")  # type: ignore[arg-type]

    layer = PatternReplicationLayer(memory=PatternMemory())
    pattern = Pattern(intent="explore", context={}, payload={"theta": 1.0})
    node = projection.project(room, count=1)[0]
    with pytest.raises(TypeError, match="pattern must be a Pattern"):
        layer.replicate("not-a-pattern", [node])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="VirtualNode instances"):
        layer.replicate(pattern, ["not-a-node"])  # type: ignore[list-item]
    node.room.memory["patterns"] = "bad"
    with pytest.raises(TypeError, match="must be a list"):
        layer.replicate(pattern, [node])


def test_pattern_replication_layer_does_not_duplicate_pattern_ids():
    room = make_room(2)
    node = VirtualNodeProjection().project(room, count=1)[0]
    pattern = Pattern(intent="explore", context={}, payload={"theta": 1.0})
    node.room.memory["patterns"] = [pattern.id]
    layer = PatternReplicationLayer()
    assert layer.replicate(pattern, [node]) == 1
    assert node.room.memory["patterns"] == [pattern.id]


def test_vqce_validates_allocation_and_compute_inputs():
    with pytest.raises(ValueError, match="stability_tolerance must be finite and positive"):
        VQCE(stability_tolerance=0.0)

    vqce = VQCE()
    with pytest.raises(TypeError, match="dim must be an integer"):
        vqce.allocate_state(dim=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name must be a string"):
        vqce.allocate_state(dim=1, name=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name must be non-empty"):
        vqce.allocate_state(dim=1, name="")
    key = vqce.allocate_state(dim=2, name="named")
    with pytest.raises(ValueError, match="already allocated"):
        vqce.allocate_state(dim=2, name="named")
    assert key == "named"

    with pytest.raises(TypeError, match="state_key must be a string"):
        vqce.compute(1, np.zeros(2), compute_fn=lambda x: x)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="compute_fn must be callable"):
        vqce.compute(key, np.zeros(2), compute_fn=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="validate_fn must be callable"):
        vqce.compute(key, np.zeros(2), compute_fn=lambda x: x, validate_fn=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not match allocated state shape"):
        vqce.compute(key, np.zeros(3), compute_fn=lambda x: x)
    with pytest.raises(ValueError, match="must preserve the allocated state shape"):
        vqce.compute(key, np.zeros(2), compute_fn=lambda x: np.zeros(3))
