from qes.agent import Agent


def test_agent_default_is_inert_and_has_unique_id():
    a1 = Agent()
    a2 = Agent()
    assert a1.id != a2.id
    assert a1.id.startswith("A-")


def test_agent_perceive_merges_observation_into_memory():
    agent = Agent()
    agent.perceive({"x": 1})
    agent.perceive({"y": 2})
    assert agent.memory == {"x": 1, "y": 2}


def test_agent_act_without_policy_returns_none_but_still_perceives():
    agent = Agent()
    result = agent.act({"seen": True}, t=0.0)
    assert result is None
    assert agent.memory["seen"] is True


def test_agent_act_with_policy_invokes_policy_with_self_observation_time():
    calls = []

    def policy(agent, observation, t):
        calls.append((agent.id, observation, t))
        return "action"

    agent = Agent(policy=policy)
    result = agent.act({"obs": 1}, t=3.5)
    assert result == "action"
    assert calls == [(agent.id, {"obs": 1}, 3.5)]


def test_agent_repr_contains_id_and_name():
    agent = Agent(name="scout")
    text = repr(agent)
    assert agent.id in text
    assert "scout" in text


def test_agent_memory_capacity_evicts_oldest():
    agent = Agent(memory_capacity=2)
    agent.perceive({"a": 1})
    agent.perceive({"b": 2})
    agent.perceive({"c": 3})
    assert list(agent.memory.keys()) == ["b", "c"]


def test_agent_memory_decay_shrinks_numeric_values():
    agent = Agent(memory_decay=0.5)
    agent.memory["level"] = 8.0
    agent.decay_memory()
    assert agent.memory["level"] == 4.0


def test_agent_memory_decay_skips_non_numeric_values():
    agent = Agent(memory_decay=0.5)
    agent.memory["label"] = "unchanged"
    agent.memory["active"] = True
    agent.decay_memory()
    assert agent.memory["label"] == "unchanged"
    assert agent.memory["active"] is True


def test_set_goal_without_metric_keeps_default_metric():
    agent = Agent()
    default_metric = agent.goal_metric
    agent.set_goal("reach target")
    assert agent.goal == "reach target"
    assert agent.goal_metric is default_metric


def test_agent_act_applies_decay_after_perceiving():
    agent = Agent(memory_decay=0.5)
    agent.act({"level": 10.0}, t=0.0)
    assert agent.memory["level"] == 5.0


def test_agent_history_is_bounded_and_records_actions():
    agent = Agent(history_capacity=2)
    agent.act({"i": 1}, t=0.0)
    agent.act({"i": 2}, t=1.0)
    agent.act({"i": 3}, t=2.0)
    assert len(agent.history) == 2
    assert agent.history[-1]["t"] == 2.0


def test_agent_goal_progress_and_satisfaction():
    agent = Agent()
    agent.set_goal("reach target", goal_metric=lambda a: a.memory.get("progress", 0.0))
    assert agent.goal_progress() == 0.0
    assert not agent.is_goal_satisfied()
    agent.memory["progress"] = 1.0
    assert agent.goal_progress() == 1.0
    assert agent.is_goal_satisfied()


def test_agent_without_goal_reports_zero_progress():
    agent = Agent()
    assert agent.goal_progress() == 0.0
    assert not agent.is_goal_satisfied()


def test_agent_send_and_receive_messages():
    sender = Agent(name="sender")
    receiver = Agent(name="receiver")
    sender.send(receiver, payload={"hello": "world"}, t=1.0)
    messages = receiver.receive()
    assert len(messages) == 1
    assert messages[0].sender == sender.id
    assert messages[0].payload == {"hello": "world"}
    assert receiver.inbox == []
