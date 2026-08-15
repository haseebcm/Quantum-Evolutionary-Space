import pytest

from qes.expansion import DomainBirthKernel, InfinityRouter, MetaExpansionEngine
from qes.universe import Universe
from qes.world import World


def test_domain_birth_kernel_spawns_unique_worlds():
    kernel = DomainBirthKernel()
    world_a = kernel.spawn()
    world_b = kernel.spawn()
    assert world_a.id != world_b.id
    assert len(kernel.births) == 2


def test_domain_birth_kernel_spawn_from_inherits_fields():
    kernel = DomainBirthKernel()
    parent = World(name="parent")
    parent.fields["temperature"] = 42
    child = kernel.spawn_from(parent)
    assert child.fields == {"temperature": 42}
    assert len(kernel.births) == 2
    assert kernel.births[-1].parent_name == "parent"


def test_infinity_router_returns_qualifying_world():
    kernel = DomainBirthKernel()
    world_low = kernel.spawn(name="low")
    world_high = kernel.spawn(name="high")
    router = InfinityRouter(kernel=kernel)

    def has_capacity(world):
        return world.name == "high"

    chosen = router.route("event", [world_low, world_high], capacity_fn=has_capacity)
    assert chosen is world_high


def test_infinity_router_births_new_world_when_none_qualify():
    kernel = DomainBirthKernel()
    world = kernel.spawn(name="full")
    router = InfinityRouter(kernel=kernel)

    chosen = router.route("event", [world], capacity_fn=lambda w: False)
    assert chosen is not world
    assert len(kernel.births) == 2  # 1 initial + 1 from router


def test_meta_expansion_engine_no_expand_below_threshold():
    universe = Universe()
    world = World(name="w1")
    universe.add_world(world)
    engine = MetaExpansionEngine()

    result = engine.expand(universe, load_fn=lambda w: 0.1, threshold=1.0)
    assert result is None
    assert engine.expansions == 0


def test_meta_expansion_engine_expands_when_all_worlds_loaded():
    universe = Universe()
    world = World(name="w1")
    universe.add_world(world)
    engine = MetaExpansionEngine()

    new_world = engine.expand(universe, load_fn=lambda w: 2.0, threshold=1.0, name="w2")
    assert new_world is not None
    assert new_world.id in universe.worlds
    assert engine.expansions == 1


def test_meta_expansion_engine_should_expand_false_for_empty_universe():
    engine = MetaExpansionEngine()
    assert engine.should_expand([], load_fn=lambda w: 1.0, threshold=0.5) is False


def test_domain_birth_kernel_and_router_validate_inputs():
    kernel = DomainBirthKernel()

    with pytest.raises(TypeError, match="string"):
        kernel.spawn(name=123)
    with pytest.raises(ValueError, match="non-empty"):
        kernel.spawn(name="   ")
    with pytest.raises(TypeError, match="parent must be a World"):
        kernel.spawn_from(parent="not-a-world")

    router = InfinityRouter(kernel=kernel)
    with pytest.raises(TypeError, match="capacity_fn must be callable"):
        router.route("event", [], capacity_fn=None)
    with pytest.raises(TypeError, match="worlds\\[0\\] must be a World"):
        router.route("event", ["not-a-world"], capacity_fn=lambda world: True)


def test_meta_expansion_engine_validates_load_threshold_and_universe_contract():
    universe = Universe()
    world = World(name="w1")
    universe.add_world(world)
    engine = MetaExpansionEngine()

    with pytest.raises(TypeError, match="load_fn must be callable"):
        engine.should_expand([world], load_fn=None, threshold=1.0)
    with pytest.raises(TypeError, match="threshold must be a real-valued scalar"):
        engine.should_expand([world], load_fn=lambda _: 1.0, threshold="bad")
    with pytest.raises(ValueError, match="threshold must be finite"):
        engine.should_expand([world], load_fn=lambda _: 1.0, threshold=float("inf"))
    with pytest.raises(TypeError, match="load_fn must return a real-valued scalar"):
        engine.should_expand([world], load_fn=lambda _: object(), threshold=1.0)
    with pytest.raises(ValueError, match="load_fn must return a finite value"):
        engine.should_expand([world], load_fn=lambda _: float("nan"), threshold=1.0)
    with pytest.raises(TypeError, match="must provide 'worlds' and 'add_world'"):
        engine.expand(object(), load_fn=lambda _: 1.0, threshold=1.0)
