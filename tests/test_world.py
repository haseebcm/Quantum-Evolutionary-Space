import numpy as np

from qes.agent import Agent
from qes.permission import GenesisPermission
from qes.room import Room
from qes.space import QESSpace
from qes.world import World


def make_space_with_room():
    space = QESSpace(permission_gate=GenesisPermission(theta=1.0))
    room = Room(
        x=np.zeros(2),
        x_star=np.zeros(2),
        lower=-np.ones(2),
        upper=np.ones(2),
        activation=np.ones(2),
    )
    space.add_room(room)
    return space


def test_world_has_unique_id():
    w1 = World()
    w2 = World()
    assert w1.id != w2.id
    assert w1.id.startswith("W-")


def test_world_add_and_remove_agent():
    world = World()
    agent = Agent()
    world.add_agent(agent)
    assert agent.id in world.agents
    world.remove_agent(agent.id)
    assert agent.id not in world.agents


def test_world_step_advances_contained_space_and_returns_telemetry():
    space = make_space_with_room()
    world = World(space=space)
    result = world.step(t=0.0, dt=1.0)
    assert "telemetry" in result
    assert result["telemetry"].time == 1.0


def test_world_step_invokes_agent_actions():
    world = World()
    calls = []

    def policy(agent, observation, t):
        calls.append((observation, t))
        return "moved"

    agent = Agent(policy=policy)
    world.add_agent(agent)
    result = world.step(t=2.0, dt=1.0, observation={"foo": "bar"})
    assert result["agent_actions"][agent.id] == "moved"
    assert calls == [({"foo": "bar"}, 2.0)]


def test_world_step_with_no_space_or_agents_returns_empty_actions():
    world = World()
    result = world.step(t=0.0, dt=1.0)
    assert result == {"agent_actions": {}}


def test_world_set_nested_universe_and_step_recurses():
    from qes.universe import Universe

    world = World()
    nested = Universe()
    world.set_nested_universe(nested)
    assert world.nested_universe is nested

    result = world.step(t=0.0, dt=1.0)
    assert "nested" in result
    assert result["nested"].time == 1.0


def test_world_repr_reflects_agent_count_and_nesting():
    from qes.universe import Universe

    world = World()
    world.add_agent(Agent())
    world.set_nested_universe(Universe())
    text = repr(world)
    assert "agents=1" in text
    assert "nested=True" in text


def test_world_set_field_enriches_agent_observation():
    world = World()
    seen = []

    def policy(agent, observation, t):
        seen.append(observation.get("temperature"))
        return None

    world.add_agent(Agent(policy=policy))
    world.set_field("temperature", 42)
    world.step(t=0.0, dt=1.0)
    assert seen == [42]


def test_world_broadcast_delivers_to_all_agents():
    world = World()
    a1, a2 = Agent(), Agent()
    world.add_agent(a1)
    world.add_agent(a2)
    delivered = world.broadcast({"alert": "collapse"}, t=1.0)
    assert delivered == 2
    assert a1.inbox[0].payload == {"alert": "collapse"}
    assert a2.inbox[0].payload == {"alert": "collapse"}


def test_world_restore_skips_agents_removed_since_snapshot():
    world = World()
    agent = Agent()
    world.add_agent(agent)
    snapshot = world.snapshot()

    world.remove_agent(agent.id)
    # Restoring after the agent was removed should not error, and should not
    # resurrect the removed agent.
    world.restore(snapshot)
    assert agent.id not in world.agents
    world = World()
    agent = Agent()
    world.add_agent(agent)
    world.memory["counter"] = 1
    world.set_field("signal", 1.0)
    agent.memory["seen"] = True

    snapshot = world.snapshot()

    world.memory["counter"] = 99
    world.set_field("signal", 99.0)
    agent.memory["seen"] = False

    world.restore(snapshot)
    assert world.memory["counter"] == 1
    assert world.fields["signal"] == 1.0
    assert agent.memory["seen"] is True
