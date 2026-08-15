import pytest

from qes.agent import Agent
from qes.multi_agent import AutonomousNodeRouter, CognitiveAgentSpawner, LayeredRoleEngine


def test_cognitive_agent_spawner_creates_population_with_state_fn():
    spawner = CognitiveAgentSpawner()
    population = spawner.spawn_population(3, name_prefix="worker", state_fn=lambda i: {"idx": i})
    assert len(population) == 3
    assert [a.state["idx"] for a in population] == [0, 1, 2]
    assert spawner.spawned == population


def test_cognitive_agent_spawner_default_empty_state():
    spawner = CognitiveAgentSpawner()
    population = spawner.spawn_population(2)
    assert all(a.state == {} for a in population)


def test_layered_role_engine_requires_nonempty_layers():
    with pytest.raises(ValueError):
        LayeredRoleEngine(layers=[])


def test_layered_role_engine_assigns_roles_across_layers():
    agents = [Agent(name=f"a{i}") for i in range(4)]
    engine = LayeredRoleEngine(layers=[["coordinator"], ["worker"]])
    assignments = engine.assign(agents)
    assert len(assignments) == 4
    coordinators = engine.agents_with_role("coordinator")
    workers = engine.agents_with_role("worker")
    assert len(coordinators) + len(workers) == 4
    for agent in agents:
        assert agent.memory["role"] in {"coordinator", "worker"}


def test_layered_role_engine_handles_empty_agent_list():
    engine = LayeredRoleEngine(layers=[["role_a"]])
    assignments = engine.assign([])
    assert assignments == {}


def test_autonomous_node_router_returns_none_for_empty_agents():
    router = AutonomousNodeRouter()
    result = router.route("task", [], fitness_fn=lambda a, t: 0.0)
    assert result is None


def test_autonomous_node_router_selects_max_fitness():
    agents = [Agent(name="a"), Agent(name="b")]
    router = AutonomousNodeRouter()
    chosen = router.route(
        "task", agents, fitness_fn=lambda a, t: 1.0 if a.name == "b" else 0.0
    )
    assert chosen.name == "b"


def test_autonomous_node_router_broadcasts_task_to_all_inboxes():
    agents = [Agent(name="a"), Agent(name="b")]
    router = AutonomousNodeRouter()
    count = router.broadcast_task("payload", agents, t=1.0)
    assert count == 2
    for agent in agents:
        messages = agent.receive()
        assert len(messages) == 1
        assert messages[0].payload == "payload"


@pytest.mark.parametrize(
    ("size", "error_type", "match"),
    [
        (True, TypeError, "size must be an integer"),
        (-1, ValueError, "size must be >= 0"),
    ],
)
def test_cognitive_agent_spawner_rejects_invalid_population_size(
    size: int, error_type: type[Exception], match: str
):
    with pytest.raises(error_type, match=match):
        CognitiveAgentSpawner().spawn_population(size)


def test_cognitive_agent_spawner_validates_other_inputs():
    spawner = CognitiveAgentSpawner()
    with pytest.raises(TypeError, match="name_prefix must be a string"):
        spawner.spawn_population(1, name_prefix=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy must be callable"):
        spawner.spawn_population(1, policy=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state_fn must be callable"):
        spawner.spawn_population(1, state_fn=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state_fn must return a dict"):
        spawner.spawn_population(1, state_fn=lambda i: [])  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("layers", "error_type", "match"),
    [
        ("not-a-list", TypeError, "layers must be a list"),
        ([("role",)], TypeError, "layer 0 must be a list"),
        ([[]], ValueError, "layer 0 must contain at least one role"),
        ([[1]], TypeError, "role names must be strings"),
        ([[""]], ValueError, "role names must be non-empty"),
    ],
)
def test_layered_role_engine_rejects_invalid_layer_shapes(
    layers: object, error_type: type[Exception], match: str
):
    with pytest.raises(error_type, match=match):
        LayeredRoleEngine(layers=layers)  # type: ignore[arg-type]


def test_layered_role_engine_stops_when_no_agents_remain():
    agent = Agent(name="solo")
    engine = LayeredRoleEngine(layers=[["lead"], ["worker"]])
    assignments = engine.assign([agent])
    assert assignments[agent.id].layer == 0
    assert engine.agents_with_role("worker") == []


def test_layered_role_engine_validates_agents_and_role_queries():
    dup_a = Agent(name="a", id="dup")
    dup_b = Agent(name="b", id="dup")
    engine = LayeredRoleEngine(layers=[["role"]])
    with pytest.raises(TypeError, match="Agent instances"):
        engine.assign([dup_a, "not-agent"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unique ids"):
        engine.assign([dup_a, dup_b])
    with pytest.raises(TypeError, match="role must be a string"):
        engine.agents_with_role(1)  # type: ignore[arg-type]


def test_autonomous_node_router_validates_fitness_function_and_scores():
    router = AutonomousNodeRouter()
    agents = [Agent(name="a"), Agent(name="b")]
    with pytest.raises(TypeError, match="fitness_fn must be callable"):
        router.route("task", agents, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite score"):
        router.route("task", agents, fitness_fn=lambda agent, task: float("nan"))
    chosen = router.route(
        "task",
        agents,
        fitness_fn=lambda agent, task: 2.0 if agent.name == "a" else 1.0,
    )
    assert chosen is agents[0]


def test_autonomous_node_router_broadcast_rejects_non_finite_timestamp():
    router = AutonomousNodeRouter()
    with pytest.raises(ValueError, match="t must be finite"):
        router.broadcast_task("payload", [Agent(name="a")], t=float("nan"))
