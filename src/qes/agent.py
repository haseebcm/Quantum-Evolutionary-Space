"""Intelligent agent (docs/QES-architecture.md, sections 33-36).

An agent is an autonomous inhabitant of a QES universe: it perceives an
observation, decides on an action via a policy function, and carries its
own memory across ticks. Agents are one of the entity kinds a `World` may
host alongside rooms (candidate realities) and digital twins.

Advanced capabilities layered on top of the base perceive/act loop:

- Bounded, decaying memory: `memory` is capped at `memory_capacity` entries
  and each existing value's influence decays by `memory_decay` per tick
  (numeric values only), so agents forget stale observations gracefully
  instead of accumulating unbounded state.
- Goal-directed behaviour: an optional `goal` payload plus `goal_progress()`
  and `is_goal_satisfied()`, driven by a pluggable `goal_metric`.
- Inter-agent messaging: a bounded `inbox` and `send()`/`receive()` so
  agents inhabiting the same `World`/`Universe` can communicate, enabling
  swarm/multi-agent coordination without changing the base act() contract.
- Action history: a bounded `history` log of every action taken, useful for
  introspection, debugging, and pattern-mining across ticks.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_id_counter = itertools.count(1)


def _next_id() -> str:
    return f"A-{next(_id_counter):05d}"


PolicyFn = Callable[["Agent", dict, float], Any]
GoalMetricFn = Callable[["Agent"], float]


@dataclass
class Message:
    """A single inter-agent message delivered via `Agent.send`/`receive`."""

    sender: str
    payload: Any
    t: float = 0.0


@dataclass
class Agent:
    """An intelligent entity A_j inhabiting a world/universe.

    Attributes:
        name: human-readable label.
        state: arbitrary agent state payload.
        memory: persistent memory/knowledge carried across ticks (bounded
            and decaying, see `memory_capacity`/`memory_decay`).
        policy: callable(agent, observation, t) -> action; if None, `act()`
            simply updates memory and returns None (a passive/inert agent).
        goal: arbitrary goal payload the agent is pursuing (None = no goal).
        goal_metric: callable(agent) -> float in [0, 1] measuring progress
            toward `goal`; defaults to a metric that always reports 0.0
            (unknown progress) when a goal is set but no metric is given.
        memory_capacity: maximum number of keys retained in `memory`; the
            oldest entries (by insertion order) are evicted once exceeded.
        memory_decay: multiplicative decay applied to numeric memory values
            every tick via `decay_memory()` (1.0 = no decay).
        history_capacity: maximum number of entries retained in `history`.
        id: unique agent identifier.
    """

    name: str = "agent"
    state: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    policy: PolicyFn | None = None
    goal: Any = None
    goal_metric: GoalMetricFn | None = None
    memory_capacity: int = 256
    memory_decay: float = 1.0
    history_capacity: int = 256
    inbox: list = field(default_factory=list)
    history: list = field(default_factory=list)
    id: str = field(default_factory=_next_id)

    def perceive(self, observation: dict) -> None:
        """Fold an observation into memory (bounded, insertion-ordered)."""
        for key, value in observation.items():
            self.memory[key] = value
        while len(self.memory) > self.memory_capacity:
            oldest_key = next(iter(self.memory))
            del self.memory[oldest_key]

    def decay_memory(self) -> None:
        """Shrink numeric memory values toward zero by `memory_decay` (<= 1.0)."""
        if self.memory_decay >= 1.0:
            return
        for key, value in list(self.memory.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.memory[key] = value * self.memory_decay

    def act(self, observation: dict, t: float) -> Any:
        """Perceive an observation, decide an action via the policy, and log it."""
        self.perceive(observation)
        self.decay_memory()
        action = None if self.policy is None else self.policy(self, observation, t)
        self.history.append({"t": t, "observation": dict(observation), "action": action})
        if len(self.history) > self.history_capacity:
            del self.history[0]
        return action

    # ------------------------------------------------------------------
    # Goal-directed behaviour
    # ------------------------------------------------------------------
    def set_goal(self, goal: Any, goal_metric: GoalMetricFn | None = None) -> None:
        """Assign a goal and (optionally) the metric used to score progress toward it."""
        self.goal = goal
        if goal_metric is not None:
            self.goal_metric = goal_metric

    def goal_progress(self) -> float:
        """Progress toward `goal` in [0, 1] via `goal_metric`; 0.0 if no goal/metric."""
        if self.goal is None or self.goal_metric is None:
            return 0.0
        return float(self.goal_metric(self))

    def is_goal_satisfied(self, threshold: float = 1.0) -> bool:
        """True iff `goal_progress()` has reached `threshold` (default: fully complete)."""
        return self.goal is not None and self.goal_progress() >= threshold

    # ------------------------------------------------------------------
    # Inter-agent messaging
    # ------------------------------------------------------------------
    def send(self, other: Agent, payload: Any, t: float = 0.0) -> None:
        """Deliver a message to another agent's inbox."""
        other.inbox.append(Message(sender=self.id, payload=payload, t=t))

    def receive(self) -> list:
        """Drain and return all pending inbox messages (FIFO)."""
        messages = list(self.inbox)
        self.inbox.clear()
        return messages

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Agent(id={self.id!r}, name={self.name!r})"
