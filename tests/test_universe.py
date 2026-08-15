from qes.agent import Agent
from qes.equation_forge import EquationForge
from qes.universe import Universe, UniverseTelemetry
from qes.world import World


def test_universe_starts_empty_at_time_zero():
    universe = Universe()
    telemetry = universe.telemetry()
    assert isinstance(telemetry, UniverseTelemetry)
    assert telemetry.time == 0.0
    assert telemetry.world_count == 0
    assert telemetry.agent_count == 0


def test_universe_add_world_and_add_agent():
    universe = Universe()
    world = World()
    agent = Agent()
    universe.add_world(world)
    universe.add_agent(agent)
    assert world.id in universe.worlds
    assert agent.id in universe.agents


def test_universe_add_equation_registers_it():
    universe = Universe()
    eq = EquationForge().seed(theta={"a": 1.0})
    universe.add_equation(eq)
    assert universe.equations[eq.id] is eq


def test_universe_all_agents_includes_free_standing_and_in_world():
    universe = Universe()
    free_agent = Agent(name="free")
    world = World()
    world_agent = Agent(name="scoped")
    world.add_agent(world_agent)
    universe.add_agent(free_agent)
    universe.add_world(world)

    all_agents = universe.all_agents()
    assert free_agent in all_agents
    assert world_agent in all_agents
    assert len(all_agents) == 2


def test_universe_step_advances_time_and_records_event():
    universe = Universe(dt=0.5)
    telemetry = universe.step(event="observation")
    assert telemetry.time == 0.5
    assert telemetry.event_count == 1
    assert universe.event_log == [(0.0, "observation")]


def test_universe_step_with_no_event_does_not_log():
    universe = Universe()
    universe.step()
    assert universe.event_log == []


def test_universe_run_executes_requested_steps():
    universe = Universe()
    telemetry_list = universe.run(steps=4)
    assert len(telemetry_list) == 4
    assert telemetry_list[-1].time == 4.0


def test_universe_run_feeds_events_per_step():
    universe = Universe()
    universe.run(steps=3, events=["a", "b"])
    assert universe.event_log == [(0.0, "a"), (1.0, "b")]


def test_universe_step_invokes_free_standing_agents():
    universe = Universe()
    seen = []

    def policy(agent, observation, t):
        seen.append(t)
        return None

    universe.add_agent(Agent(policy=policy))
    universe.step()
    assert seen == [0.0]


def test_spawn_nested_universe_sets_world_nested_universe():
    universe = Universe()
    world = World()
    universe.add_world(world)
    nested = Universe()
    universe.spawn_nested_universe(world.id, nested)
    assert world.nested_universe is nested


def test_universe_step_recursively_steps_nested_universe():
    universe = Universe()
    world = World()
    universe.add_world(world)
    nested = Universe()
    universe.spawn_nested_universe(world.id, nested)

    universe.step()
    # The nested universe should have advanced by its own dt (default 1.0).
    assert nested.time == 1.0


def test_universe_repr_reflects_world_and_agent_counts():
    universe = Universe()
    universe.add_world(World())
    universe.add_agent(Agent())
    text = repr(universe)
    assert "worlds=1" in text
    assert "agents=1" in text


def test_universe_clone_is_independent_and_has_new_id():
    universe = Universe()
    universe.add_agent(Agent(name="scout"))
    clone = universe.clone()

    assert clone.id != universe.id
    assert clone is not universe
    clone.step()
    assert clone.time == 1.0
    assert universe.time == 0.0  # original untouched


def test_universe_simulate_returns_evolved_clone_without_mutating_original():
    universe = Universe()
    branch = universe.simulate(steps=3)
    assert branch.time == 3.0
    assert universe.time == 0.0


def test_universe_compare_reports_telemetry_deltas():
    universe_a = Universe()
    universe_b = Universe()
    universe_b.add_world(World())
    universe_b.step()

    comparison = universe_a.compare(universe_b)
    assert comparison.world_count_delta == 1
    assert comparison.time_delta == 1.0


def test_universe_merge_combines_worlds_and_agents_without_mutating_sources():
    universe_a = Universe()
    universe_a.add_agent(Agent(name="a"))
    universe_a.add_equation(EquationForge().seed(theta={"a": 1.0}))
    universe_b = Universe()
    universe_b.add_world(World(name="b-world"))

    merged = universe_a.merge(universe_b)

    assert len(merged.agents) == 1
    assert len(merged.worlds) == 1
    assert len(merged.equations) == 1
    assert len(universe_a.worlds) == 0  # original untouched
    assert len(universe_b.agents) == 0  # original untouched


def test_universe_collapse_nested_absorbs_inhabitants_and_removes_nesting():
    universe = Universe()
    host_world = World()
    universe.add_world(host_world)
    nested = Universe()
    nested_world = World(name="inner")
    nested.add_world(nested_world)
    nested.add_agent(Agent(name="inner-agent"))
    nested.add_equation(EquationForge().seed(theta={"a": 1.0}))
    universe.spawn_nested_universe(host_world.id, nested)

    collapsed = universe.collapse_nested(host_world.id)

    assert collapsed is nested
    assert nested_world.id in universe.worlds
    assert host_world.nested_universe is None
    assert len(universe.all_agents()) == 1
    assert len(universe.equations) == 1


def test_universe_collapse_nested_returns_none_when_no_nested_universe():
    universe = Universe()
    world = World()
    universe.add_world(world)
    assert universe.collapse_nested(world.id) is None


def test_universe_restore_skips_worlds_removed_since_snapshot():
    universe = Universe()
    world = World()
    universe.add_world(world)
    snapshot = universe.snapshot()

    del universe.worlds[world.id]
    # Restoring after the world was removed should not error or resurrect it.
    universe.restore(snapshot)
    assert world.id not in universe.worlds


def test_universe_snapshot_and_restore_roundtrip():
    universe = Universe()
    world = World()
    agent = Agent()
    world.add_agent(agent)
    universe.add_world(world)
    universe.memory["seed"] = 1
    agent.memory["value"] = "before"

    snapshot = universe.snapshot()
    universe.step(event="drift")
    agent.memory["value"] = "after"

    universe.restore(snapshot)
    assert universe.time == 0.0
    assert universe.event_log == []
    assert universe.memory["seed"] == 1
    assert agent.memory["value"] == "before"
