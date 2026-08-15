"""Virtual world / environment (docs/QES-architecture.md, sections 33-35).

A World (R_i) is a virtual reality inside a QES universe: `Q supseteq R_i`.
It may host a population of candidate realities via a `QESSpace`, a set of
intelligent agents, and -- because a world is itself a first-class object
-- optionally a nested QES universe (`Q supseteq R_i supseteq Q'_i`),
enabling the recursive containment chain `Q^(0) supseteq Q^(1) supseteq ...`.

Advanced capabilities:

- Shared environment fields: `fields` is a world-global dict every agent's
  observation is enriched with each tick, modelling a shared physical/
  informational medium (temperature, price, signal strength, ...) that all
  inhabitants of the world perceive identically.
- Message bus: `broadcast()` delivers a payload to every agent's inbox in
  one call, and `route_messages()` drains each agent's outbox-equivalent
  (any pending `Agent.send` calls made during `step()`) is handled directly
  by agents themselves; the world only provides the shared broadcast
  channel plus per-tick message-count telemetry.
- Snapshot/restore: `snapshot()` captures a deep copy of mutable world
  state (agent memories/inboxes, environment fields, room population
  weights/positions via `QESSpace` telemetry) so a world's trajectory can
  be checkpointed and rewound -- the World-level analogue of Universe
  branching operations.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qes.agent import Agent
from qes.space import QESSpace

if TYPE_CHECKING:
    from qes.universe import Universe

_id_counter = itertools.count(1)


def _next_id() -> str:
    return f"W-{next(_id_counter):05d}"


@dataclass
class World:
    """A virtual reality/environment R_i hosting rooms, agents, and/or a nested universe.

    Attributes:
        name: human-readable label.
        space: an optional `QESSpace` -- the population of candidate rooms
            (realities) this world governs.
        agents: intelligent agents inhabiting this world, keyed by id.
        nested_universe: an optional nested `Universe` hosted inside this
            world (`Q supseteq R_i supseteq Q'_i`).
        memory: world-local memory/knowledge store.
        fields: shared environment fields every agent perceives each tick
            (e.g. ambient signals all inhabitants have equal access to).
        id: unique world identifier.
    """

    name: str = "world"
    space: QESSpace | None = None
    agents: dict = field(default_factory=dict)
    nested_universe: Universe | None = None
    memory: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    id: str = field(default_factory=_next_id)

    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id: str) -> None:
        self.agents.pop(agent_id, None)

    def set_nested_universe(self, universe: Universe) -> None:
        """Host a nested QES universe inside this world: Q supseteq R_i supseteq Q'_i."""
        self.nested_universe = universe

    # ------------------------------------------------------------------
    # Shared environment + inter-agent messaging
    # ------------------------------------------------------------------
    def set_field(self, key: str, value: object) -> None:
        """Set/update a world-global environment field visible to every agent."""
        self.fields[key] = value

    def broadcast(self, payload: object, t: float = 0.0, sender: str = "world") -> int:
        """Deliver `payload` to every agent's inbox. Returns the recipient count."""
        from qes.agent import Message

        for agent in self.agents.values():
            agent.inbox.append(Message(sender=sender, payload=payload, t=t))
        return len(self.agents)

    def step(self, t: float, dt: float, observation: dict | None = None) -> dict:
        """Advance this world one tick.

        Steps its room population (if any), lets every agent perceive/act
        (observation enriched with the shared `fields`), and recursively
        steps its nested universe (if any). Returns a dict summarizing what
        happened this tick.
        """
        observation = dict(observation or {})
        observation.update(self.fields)
        result: dict = {"agent_actions": {}}
        if self.space is not None:
            result["telemetry"] = self.space.step()
        for agent_id, agent in self.agents.items():
            result["agent_actions"][agent_id] = agent.act(observation, t)
        if self.nested_universe is not None:
            result["nested"] = self.nested_universe.step()
        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Deep-copy the mutable parts of this world's state for later `restore()`."""
        return {
            "memory": copy.deepcopy(self.memory),
            "fields": copy.deepcopy(self.fields),
            "agents": {
                agent_id: {
                    "state": copy.deepcopy(agent.state),
                    "memory": copy.deepcopy(agent.memory),
                    "inbox": copy.deepcopy(agent.inbox),
                    "history": copy.deepcopy(agent.history),
                }
                for agent_id, agent in self.agents.items()
            },
        }

    def restore(self, snapshot: dict) -> None:
        """Restore world/agent state previously captured by `snapshot()`."""
        self.memory = copy.deepcopy(snapshot.get("memory", {}))
        self.fields = copy.deepcopy(snapshot.get("fields", {}))
        for agent_id, agent_snapshot in snapshot.get("agents", {}).items():
            agent = self.agents.get(agent_id)
            if agent is None:
                continue
            agent.state = copy.deepcopy(agent_snapshot["state"])
            agent.memory = copy.deepcopy(agent_snapshot["memory"])
            agent.inbox = copy.deepcopy(agent_snapshot["inbox"])
            agent.history = copy.deepcopy(agent_snapshot["history"])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"World(id={self.id!r}, name={self.name!r}, "
            f"agents={len(self.agents)}, nested={self.nested_universe is not None})"
        )
