"""Universal / Cosmic Expansion Domain.

Source-derived components (see the PDF-extraction summary, "Universal
Expansion Domain" and "EC-Loop"):

    Meta-Expansion Engine   -- grows the reachable universe (new worlds,
                               new domains) rather than the state inside a
                               fixed universe.
    Domain Birth Kernel     -- instantiates brand-new domains/worlds on
                               demand, seeded from a parent.
    Infinity Router         -- routes an incoming event/request to the
                               (possibly newly-born) world/domain best
                               suited to handle it, growing capacity
                               without bound as load increases.

This complements `qes.hypervisor.CosmicVP.expand()` (which grows raw node
*capacity*) by growing the *inhabited* universe itself -- new `World`
objects attached to a `Universe`, each optionally seeded with its own
nested sub-universe, forming the recursive containment chain
`Q^(0) supseteq Q^(1) supseteq Q^(2) supseteq ...` described in
`qes.universe`.
"""
from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qes.world import World

if TYPE_CHECKING:
    from qes.universe import Universe

_domain_id_counter = itertools.count(1)


def _next_domain_id() -> str:
    return f"D-{next(_domain_id_counter):05d}"


def _validate_optional_name(name: str | None) -> str | None:
    """Validate an optional world/domain name."""
    if name is None:
        return None
    if not isinstance(name, str):
        raise TypeError("name must be a string when provided")
    if not name.strip():
        raise ValueError("name must be non-empty when provided")
    return name


def _validate_threshold(threshold: float) -> float:
    """Validate an expansion threshold."""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError("threshold must be a real-valued scalar") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("threshold must be finite")
    return value


def _validate_load(value: Any, *, name: str) -> float:
    """Validate a load/capacity scalar."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must return a real-valued scalar") from exc
    if scalar != scalar or scalar in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must return a finite value")
    return scalar


@dataclass
class DomainBirthRecord:
    """Record of one domain-birth event."""

    domain_id: str
    world: World
    parent_name: str | None = None


class DomainBirthKernel:
    """Instantiates brand-new worlds/domains on demand.

    `spawn()` creates a fresh, empty `World`; `spawn_from(parent)` clones a
    parent world's shared `fields` (its ambient environment) into the new
    child so newly-born domains inherit context rather than starting from
    a completely blank universe.
    """

    def __init__(self) -> None:
        self.births: list[DomainBirthRecord] = []

    def spawn(self, name: str | None = None) -> World:
        """Create a fresh world/domain and record the birth event."""
        validated_name = _validate_optional_name(name)
        domain_id = _next_domain_id()
        world = World(name=validated_name or domain_id)
        self.births.append(DomainBirthRecord(domain_id=domain_id, world=world))
        return world

    def spawn_from(self, parent: World, name: str | None = None) -> World:
        """Create a child world seeded from `parent`'s shared environment fields."""
        if not isinstance(parent, World):
            raise TypeError("parent must be a World")
        child = self.spawn(name=name)
        child.fields = copy.deepcopy(parent.fields)
        self.births.append(
            DomainBirthRecord(
                domain_id=self.births[-1].domain_id,
                world=child,
                parent_name=parent.name,
            )
        )
        return child


class InfinityRouter:
    """Routes an incoming event to the world best able to handle it,
    growing the universe (via `DomainBirthKernel`) whenever no existing
    world qualifies -- so routing capacity expands without bound as load
    grows, rather than rejecting excess load."""

    def __init__(self, kernel: DomainBirthKernel | None = None) -> None:
        """Initialize the router with a domain-birth kernel."""
        self.kernel = kernel or DomainBirthKernel()

    def route(
        self,
        event: object,
        worlds: list[World],
        capacity_fn: Callable[[World], bool],
    ) -> World:
        """Return the first world in `worlds` for which `capacity_fn(world)`
        is True; if none qualify, birth a fresh world via the kernel and
        return it (unconditionally has capacity, since it starts empty)."""
        del event
        if not callable(capacity_fn):
            raise TypeError("capacity_fn must be callable")
        for index, world in enumerate(worlds):
            if not isinstance(world, World):
                raise TypeError(f"worlds[{index}] must be a World")
        for world in worlds:
            if bool(capacity_fn(world)):
                return world
        return self.kernel.spawn()


class MetaExpansionEngine:
    """EC-Loop at the universe level: expands a `Universe`'s reachable
    space by birthing new worlds, optionally nesting a sub-universe inside
    each, once existing worlds cross a load threshold."""

    def __init__(self, kernel: DomainBirthKernel | None = None) -> None:
        """Initialize the engine with its domain-birth kernel."""
        self.kernel = kernel or DomainBirthKernel()
        self.expansions: int = 0

    def should_expand(self, worlds: list[World], load_fn: Callable[[World], float], threshold: float) -> bool:
        """Return True when every current world is at or above the load threshold."""
        if not callable(load_fn):
            raise TypeError("load_fn must be callable")
        threshold_value = _validate_threshold(threshold)
        if not worlds:
            return False
        loads = [_validate_load(load_fn(world), name="load_fn") for world in worlds]
        return all(load >= threshold_value for load in loads)

    def expand(
        self,
        universe: Universe,
        load_fn: Callable[[World], float],
        threshold: float,
        name: str | None = None,
    ) -> World | None:
        """If every existing world in `universe` is at/above `threshold`
        load, birth and attach a new world; returns the new world, or None
        if expansion was not triggered."""
        if not hasattr(universe, "worlds") or not hasattr(universe, "add_world"):
            raise TypeError("universe must provide 'worlds' and 'add_world'")
        worlds = list(universe.worlds.values())
        if not self.should_expand(worlds, load_fn, threshold):
            return None
        new_world = self.kernel.spawn(name=name)
        universe.add_world(new_world)
        self.expansions += 1
        return new_world
