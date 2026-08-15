"""QES Universe (docs/QES-architecture.md, sections 33-36).

    Q = { S, E, A, R, M, C, T, G }

The universe is the top-level, persistent digital space in which worlds,
agents, equations, memory, compute, time, and governance rules exist as
first-class objects. Unlike `QESSpace` (a single governed population of
rooms), a `Universe` may host many worlds simultaneously, each of which may
in turn host its own room population and/or a further nested universe,
producing the recursive containment chain:

    Q^(0) supseteq Q^(1) supseteq Q^(2) supseteq ...

Evolution follows the event-driven master transition:

    Q(t + dt) = F( Q(t), I(t) )

where `I(t)` is any new input, event, observation, or generated change
delivered to the universe at time `t`.

Reality branching (docs section 35, "Reality Branching" hierarchy: Clone,
Mutate, Simulate, Compare, Merge, Collapse) is realized here as first-class
operations on the universe itself, not just on individual rooms:

- `clone()`      -- Clone: an independent deep copy of the whole universe,
                    free to diverge (`Q_a`, `Q_b` share no mutable state).
- `snapshot()`/`restore()` -- lightweight checkpoint/rewind of universe state.
- `compare()`    -- Compare: a telemetry-level diff between two universes.
- `merge()`      -- Merge: combine two universes' worlds/agents/equations
                    into a new universe.
- `collapse_nested()` -- Collapse: pull a nested universe's inhabitants up
                    into the parent, removing one level of nesting.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any

from qes.agent import Agent
from qes.compute_allocator import ComputeAllocator
from qes.equation_forge import Equation
from qes.permission import GenesisPermission
from qes.world import World

InputEvent = Any

_id_counter = itertools.count(1)


def _next_id() -> str:
    return f"Q-{next(_id_counter):05d}"


@dataclass
class UniverseTelemetry:
    """Snapshot of Q(t) at the universe level."""

    time: float
    world_count: int
    agent_count: int
    equation_count: int
    event_count: int


@dataclass
class UniverseComparison:
    """Result of `Universe.compare()`: a telemetry-level diff between two universes."""

    time_delta: float
    world_count_delta: int
    agent_count_delta: int
    equation_count_delta: int
    event_count_delta: int


class Universe:
    """Q: the persistent intelligent virtual universe.

    Contains worlds (R), agents (A), equations (E), memory (M), compute (C),
    virtual time (T), and governance (G) as first-class objects. Rooms,
    simulations, models, and agents are only ever inhabitants of a Universe
    -- never the universe itself.
    """

    def __init__(
        self,
        governance: GenesisPermission | None = None,
        compute: ComputeAllocator | None = None,
        dt: float = 1.0,
        name: str = "universe",
    ):
        self.id = _next_id()
        self.name = name
        self.worlds: dict = {}  # R: id -> World
        self.agents: dict = {}  # A: free-standing (world-less) id -> Agent
        self.equations: dict = {}  # E: id -> Equation
        self.memory: dict = {}  # M: arbitrary universe-level knowledge store
        self.compute = compute  # C: finite compute budget allocator
        self.governance = governance  # G: universe-level governance/admission rule
        self.dt = dt  # part of T
        self.time: float = 0.0  # T: virtual clock
        self.event_log: list = []  # history of I(t) inputs

    # ------------------------------------------------------------------
    # Population management (S, E, A, R)
    # ------------------------------------------------------------------
    def add_world(self, world: World) -> None:
        """Instantiate a virtual world/environment inside this universe."""
        self.worlds[world.id] = world

    def add_agent(self, agent: Agent) -> None:
        """Instantiate a free-standing agent (not scoped to any one world)."""
        self.agents[agent.id] = agent

    def add_equation(self, equation: Equation) -> None:
        """Register an equation/model in the universe's equation population."""
        self.equations[equation.id] = equation

    def spawn_nested_universe(self, world_id: str, universe: Universe) -> None:
        """Host a nested universe inside one of this universe's worlds.

        Realizes `Q supseteq R_i supseteq Q'_i` -- the world identified by
        `world_id` now itself contains an independently governed universe.
        """
        world = self.worlds[world_id]
        world.set_nested_universe(universe)

    # ------------------------------------------------------------------
    # Evolution: Q(t+dt) = F(Q(t), I(t))
    # ------------------------------------------------------------------
    def step(self, event: InputEvent | None = None) -> UniverseTelemetry:
        """Advance the universe one tick.

        Steps every world (which in turn steps its rooms, agents, and any
        nested universe), then every free-standing agent, folding in any
        external input event `I(t)`.
        """
        if event is not None:
            self.event_log.append((self.time, event))
        observation = {"event": event} if event is not None else {}
        for world in self.worlds.values():
            world.step(self.time, self.dt, observation)
        for agent in self.agents.values():
            agent.act(observation, self.time)
        self.time += self.dt
        return self.telemetry()

    def run(self, steps: int, events: list | None = None) -> list:
        """Advance the universe `steps` ticks, optionally feeding one input
        event per tick from `events` (padded with `None` if shorter)."""
        events = events or []
        return [
            self.step(events[i] if i < len(events) else None) for i in range(steps)
        ]

    # ------------------------------------------------------------------
    # Reality branching: Clone, Simulate, Compare, Merge, Collapse
    # ------------------------------------------------------------------
    def clone(self, name: str | None = None) -> Universe:
        """Clone: an independent deep copy of this universe, free to diverge.

        The clone shares no mutable state with the original (worlds, agents,
        equations, and memory are deep-copied) but keeps its own fresh id.
        This is the universe-level analogue of `Room.clone()` / reality
        branching `B(R_i) = {R_i1, ..., R_im}`, promoted to whole realities.
        """
        clone = copy.deepcopy(self)
        clone.id = _next_id()
        clone.name = name or f"{self.name}-clone"
        return clone

    def simulate(self, steps: int, events: list | None = None) -> Universe:
        """Simulate: clone this universe and run the clone forward `steps`
        ticks, leaving the original untouched. Returns the evolved clone."""
        branch = self.clone()
        branch.run(steps, events)
        return branch

    def compare(self, other: Universe) -> UniverseComparison:
        """Compare: a telemetry-level diff between this universe and `other`."""
        mine = self.telemetry()
        theirs = other.telemetry()
        return UniverseComparison(
            time_delta=theirs.time - mine.time,
            world_count_delta=theirs.world_count - mine.world_count,
            agent_count_delta=theirs.agent_count - mine.agent_count,
            equation_count_delta=theirs.equation_count - mine.equation_count,
            event_count_delta=theirs.event_count - mine.event_count,
        )

    def merge(self, other: Universe, name: str | None = None) -> Universe:
        """Merge: combine this universe and `other` into a new universe.

        Worlds, free-standing agents, equations, and memory from both
        universes are folded into a fresh `Universe` (deep-copied, so
        neither source universe is mutated). Governance/compute/dt are
        inherited from `self`; `other`'s memory keys win on conflict.
        """
        merged = Universe(governance=self.governance, compute=self.compute, dt=self.dt)
        merged.name = name or f"{self.name}+{other.name}"
        merged.time = max(self.time, other.time)
        for source in (self, other):
            for world in source.worlds.values():
                merged.add_world(copy.deepcopy(world))
            for agent in source.agents.values():
                merged.add_agent(copy.deepcopy(agent))
            for equation in source.equations.values():
                merged.add_equation(copy.deepcopy(equation))
            merged.memory.update(copy.deepcopy(source.memory))
        merged.event_log = list(self.event_log) + list(other.event_log)
        return merged

    def collapse_nested(self, world_id: str) -> Universe | None:
        """Collapse: pull a nested universe's inhabitants up into `self`.

        The nested universe hosted inside world `world_id` has its worlds,
        free-standing agents, and equations absorbed directly into this
        universe, and the nesting link is removed (`Q supseteq R supseteq Q'`
        becomes `Q supseteq R`, with `Q'`'s inhabitants now direct children
        of `Q`). Returns the collapsed (now detached) nested universe, or
        None if the target world had no nested universe.
        """
        world = self.worlds[world_id]
        nested = world.nested_universe
        if nested is None:
            return None
        for w in nested.worlds.values():
            self.add_world(w)
        for a in nested.agents.values():
            self.add_agent(a)
        for e in nested.equations.values():
            self.add_equation(e)
        self.memory.update(nested.memory)
        world.nested_universe = None
        return nested

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Deep-copy the full universe state for later `restore()`."""
        return {
            "time": self.time,
            "memory": copy.deepcopy(self.memory),
            "event_log": copy.deepcopy(self.event_log),
            "worlds": {w_id: w.snapshot() for w_id, w in self.worlds.items()},
        }

    def restore(self, snapshot: dict) -> None:
        """Restore universe/world/agent state previously captured by `snapshot()`."""
        self.time = snapshot["time"]
        self.memory = copy.deepcopy(snapshot["memory"])
        self.event_log = copy.deepcopy(snapshot["event_log"])
        for world_id, world_snapshot in snapshot.get("worlds", {}).items():
            world = self.worlds.get(world_id)
            if world is not None:
                world.restore(world_snapshot)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def all_agents(self) -> list:
        """Every agent reachable from this universe: free-standing + in-world."""
        result = list(self.agents.values())
        for world in self.worlds.values():
            result.extend(world.agents.values())
        return result

    def telemetry(self) -> UniverseTelemetry:
        return UniverseTelemetry(
            time=self.time,
            world_count=len(self.worlds),
            agent_count=len(self.all_agents()),
            equation_count=len(self.equations),
            event_count=len(self.event_log),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Universe(id={self.id!r}, t={self.time:.2f}, worlds={len(self.worlds)}, "
            f"agents={len(self.all_agents())})"
        )
