"""Multi-Agent Domain.

Source-derived components (see the PDF-extraction summary, "Multi-Agent
Domain"):

    Cognitive Agent Spawner  -- births a population of `Agent`s from a
                                template, each an independent cognitive
                                inhabitant of a `World`.
    Layered Role Engine       -- assigns each agent a role from a layered
                                hierarchy (e.g. explorer/worker/coordinator)
                                and exposes role-scoped policies.
    Autonomous Node Router    -- routes an incoming task/message to the
                                agent best suited to handle it, without a
                                central scheduler making per-tick decisions
                                for every agent.

Builds directly on `qes.agent.Agent`; distinct from that module in that it
governs *populations* of agents (spawning, role assignment, routing)
rather than a single agent's perceive/act loop.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from qes.agent import Agent, Message, PolicyFn

StateFactory = Callable[[int], dict[str, Any]]
FitnessFn = Callable[[Agent, Any], float]


def _validate_population_size(size: int) -> int:
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("size must be an integer")
    if size < 0:
        raise ValueError("size must be >= 0")
    return size


def _validate_agents(agents: Sequence[Agent]) -> list[Agent]:
    validated = list(agents)
    for agent in validated:
        if not isinstance(agent, Agent):
            raise TypeError("agents must contain Agent instances")
    ids = [agent.id for agent in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("agents must have unique ids")
    return validated


@dataclass
class RoleAssignment:
    """Records which role an agent was assigned, and by which layer."""

    agent_id: str
    role: str
    layer: int


class CognitiveAgentSpawner:
    """Births a population of agents from a template policy/state."""

    def __init__(self) -> None:
        self.spawned: list[Agent] = []

    def spawn_population(
        self,
        size: int,
        name_prefix: str = "agent",
        policy: PolicyFn | None = None,
        state_fn: StateFactory | None = None,
    ) -> list[Agent]:
        """Create `size` independent agents sharing `policy` but each with
        its own (optionally index-derived, via `state_fn`) initial state.

        Args:
            size: number of agents to create; must be >= 0.
            name_prefix: prefix used when naming each created agent.
            policy: optional shared policy callable for every agent.
            state_fn: optional callable returning the initial state for agent i.
        """
        _validate_population_size(size)
        if not isinstance(name_prefix, str):
            raise TypeError("name_prefix must be a string")
        if policy is not None and not callable(policy):
            raise TypeError("policy must be callable")
        if state_fn is not None and not callable(state_fn):
            raise TypeError("state_fn must be callable")

        population: list[Agent] = []
        for i in range(size):
            raw_state = state_fn(i) if state_fn is not None else {}
            if not isinstance(raw_state, dict):
                raise TypeError("state_fn must return a dict")
            agent = Agent(name=f"{name_prefix}-{i}", state=dict(raw_state), policy=policy)
            population.append(agent)
        self.spawned.extend(population)
        return population


class LayeredRoleEngine:
    """Assigns agents to roles drawn from an ordered list of layers (e.g.
    layer 0 = "coordinator", layer 1 = "worker", ...), distributing agents
    round-robin across roles within each layer so populations naturally
    stratify into a role hierarchy."""

    def __init__(self, layers: list[list[str]]):
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of role lists")
        if not layers:
            raise ValueError("layers must be non-empty")

        normalized_layers: list[list[str]] = []
        for layer_index, roles in enumerate(layers):
            if not isinstance(roles, list):
                raise TypeError(f"layer {layer_index} must be a list of role names")
            if not roles:
                raise ValueError(f"layer {layer_index} must contain at least one role")
            normalized_roles: list[str] = []
            for role in roles:
                if not isinstance(role, str):
                    raise TypeError("role names must be strings")
                if not role:
                    raise ValueError("role names must be non-empty")
                normalized_roles.append(role)
            normalized_layers.append(normalized_roles)

        self.layers = normalized_layers
        self.assignments: dict[str, RoleAssignment] = {}

    def assign(self, agents: list[Agent]) -> dict[str, RoleAssignment]:
        """Distribute `agents` across the configured layers/roles, filling
        each layer's roles round-robin before moving to the next layer."""
        validated_agents = _validate_agents(agents)
        if not validated_agents:
            return {}

        remaining = list(validated_agents)
        total_layers = len(self.layers)
        updated_assignments = dict(self.assignments)

        for layer_idx, roles in enumerate(self.layers):
            if not remaining:
                break
            layers_left = total_layers - layer_idx
            layer_size = max(1, int(np.ceil(len(remaining) / layers_left)))
            layer_agents = remaining[:layer_size]
            remaining = remaining[layer_size:]
            role_count = len(roles)
            for i, agent in enumerate(layer_agents):
                role = roles[i % role_count]
                agent.memory["role"] = role
                updated_assignments[agent.id] = RoleAssignment(
                    agent_id=agent.id,
                    role=role,
                    layer=layer_idx,
                )

        self.assignments = updated_assignments
        return dict(self.assignments)

    def agents_with_role(self, role: str) -> list[str]:
        """Return the ids of agents currently assigned `role`."""
        if not isinstance(role, str):
            raise TypeError("role must be a string")
        return [a_id for a_id, rec in self.assignments.items() if rec.role == role]


class AutonomousNodeRouter:
    """Routes an incoming task to the agent best able to handle it, per a
    supplied fitness function, without a central per-tick scheduler."""

    def route(self, task: Any, agents: list[Agent], fitness_fn: FitnessFn) -> Agent | None:
        """Return the agent maximizing `fitness_fn(agent, task)`, or None
        if `agents` is empty."""
        validated_agents = _validate_agents(agents)
        if not validated_agents:
            return None
        if not callable(fitness_fn):
            raise TypeError("fitness_fn must be callable")

        best_agent: Agent | None = None
        best_score = -np.inf
        for agent in validated_agents:
            score = float(fitness_fn(agent, task))
            if not np.isfinite(score):
                raise ValueError("fitness_fn must return a finite score")
            if best_agent is None or score > best_score:
                best_agent = agent
                best_score = score
        return best_agent

    def broadcast_task(self, task: Any, agents: list[Agent], t: float = 0.0) -> int:
        """Deliver `task` as a message to every agent's inbox; returns the
        count of agents reached."""
        validated_agents = _validate_agents(agents)
        timestamp = float(t)
        if not np.isfinite(timestamp):
            raise ValueError("t must be finite")

        for agent in validated_agents:
            agent.inbox.append(Message(sender="router", payload=task, t=timestamp))
        return len(validated_agents)
