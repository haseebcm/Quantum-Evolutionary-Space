"""Global Invariant Engine (Phase 1, "Harden the core").

QES's per-object `snapshot()`/`restore()` methods (on `Room`, `QESSpace`,
`World`, `Universe`) already checkpoint state faithfully. What was missing
was a single, cross-cutting engine that can answer, at any level of the
containment hierarchy `Q supseteq R_i supseteq A_j`:

    Is this state actually internally consistent?

This module runs the same battery of structural invariants over a `Room`,
a `QESSpace` (room population), a `World`, or a whole `Universe`
(recursing through nested worlds/universes), and reports every violation
found rather than raising on the first one -- so a caller gets a complete
diagnosis in one pass.

Checked invariants:

- **State validity**: every state vector is finite (no NaN/Inf).
- **Dimensional consistency**: `x`, `x_star`, `lower`, `upper`, `activation`
  all share one dimension; `couplings` is a square matrix of that dimension.
- **Bounds**: `lower <= upper` elementwise (an empty/inverted envelope is
  never admissible, regardless of what state currently occupies it).
- **Lineage integrity**: a room's `lineage` chain contains no duplicate or
  self-referential ancestor ids.
- **Resource consistency**: allocated `compute` amounts are finite and
  non-negative.
- **Probability normalization**: room population weights (`Room.weight`,
  used as `p_i` inside a `QESSpace`) are finite and non-negative.
- **Lifecycle consistency**: `Room.state` is one of the states recognized
  by `qes.orchestrator.RoomLifecycle`.

This is deliberately a read-only diagnostic layer: it never mutates the
objects it inspects, and never raises for a failed check -- callers decide
what to do with an `InvariantReport` (log it, refuse to persist, roll back
via `restore()`, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from qes.orchestrator import VALID_STATES

if TYPE_CHECKING:
    from qes.room import Room
    from qes.space import QESSpace
    from qes.universe import Universe
    from qes.world import World


@dataclass
class InvariantViolation:
    """One failed invariant check."""

    scope: str  # e.g. "room", "space", "world", "universe"
    check: str  # invariant name, e.g. "dimensional_consistency"
    message: str
    object_id: str = ""


@dataclass
class InvariantReport:
    """Aggregate result of running the invariant engine over one object
    (and, for containers, everything nested inside it)."""

    passed: bool
    violations: list[InvariantViolation] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    def merge(self, other: InvariantReport) -> InvariantReport:
        """Combine this report with another, keeping every violation."""
        violations = [*self.violations, *other.violations]
        return InvariantReport(passed=not violations, violations=violations)

    def summary(self) -> str:
        """One-line human-readable summary, e.g. for logging."""
        if self.passed:
            return "invariants OK (0 violations)"
        by_check: dict[str, int] = {}
        for v in self.violations:
            by_check[v.check] = by_check.get(v.check, 0) + 1
        detail = ", ".join(f"{name}={count}" for name, count in sorted(by_check.items()))
        return f"invariants FAILED ({len(self.violations)} violations: {detail})"


def _finite(arr: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(arr)))


class InvariantEngine:
    """Runs the global invariant battery over rooms/spaces/worlds/universes."""

    def check_room(self, room: Room) -> InvariantReport:
        """Check one room's structural invariants (does not check `weight`
        normalization across a population -- see `check_space`)."""
        violations: list[InvariantViolation] = []

        def fail(check: str, message: str) -> None:
            violations.append(
                InvariantViolation(scope="room", check=check, message=message, object_id=room.id)
            )

        vectors = {
            "x": room.x,
            "x_star": room.x_star,
            "lower": room.lower,
            "upper": room.upper,
            "activation": room.activation,
        }

        # -- State validity: every vector finite.
        for name, arr in vectors.items():
            if not _finite(arr):
                fail("state_validity", f"{name} contains non-finite values")

        # -- Dimensional consistency: all vectors share one dimension.
        dims = {name: arr.shape for name, arr in vectors.items()}
        unique_dims = set(dims.values())
        if len(unique_dims) > 1:
            fail(
                "dimensional_consistency",
                f"inconsistent shapes across state vectors: {dims}",
            )
        if room.couplings is not None:
            expected = (room.dim, room.dim)
            if room.couplings.shape != expected:
                fail(
                    "dimensional_consistency",
                    f"couplings shape {room.couplings.shape} != expected {expected}",
                )

        # -- Bounds: lower <= upper elementwise.
        if room.lower.shape == room.upper.shape and np.any(room.lower > room.upper):
            fail("bounds", "lower bound exceeds upper bound in at least one dimension")

        # -- Lineage integrity: no duplicate/self-referential ancestors.
        if len(room.lineage) != len(set(room.lineage)):
            fail("lineage_integrity", "lineage contains duplicate ancestor ids")
        if room.id in room.lineage:
            fail("lineage_integrity", "room is listed as its own ancestor")

        # -- Resource consistency: compute allocations finite and >= 0.
        for key, amount in room.compute.items():
            try:
                value = float(amount)
            except (TypeError, ValueError):
                fail("resource_consistency", f"compute[{key!r}] is not numeric: {amount!r}")
                continue
            if not np.isfinite(value):
                fail("resource_consistency", f"compute[{key!r}]={value} is not finite")
            elif value < 0:
                fail("resource_consistency", f"compute[{key!r}]={value} is negative")

        # -- Probability normalization: weight finite and >= 0.
        if not np.isfinite(room.weight):
            fail("probability_normalization", f"weight={room.weight} is not finite")
        elif room.weight < 0:
            fail("probability_normalization", f"weight={room.weight} is negative")

        # -- Lifecycle consistency: state recognized by RoomLifecycle.
        if room.state not in VALID_STATES:
            fail("lifecycle_consistency", f"unrecognized lifecycle state {room.state!r}")

        return InvariantReport(passed=not violations, violations=violations)

    def check_space(self, space: QESSpace) -> InvariantReport:
        """Check every room in a `QESSpace`'s population."""
        report = InvariantReport(passed=True)
        for room in space.rooms.values():
            report = report.merge(self.check_room(room))
        return report

    def check_world(self, world: World) -> InvariantReport:
        """Check a world's room population and, if present, its nested universe."""
        report = InvariantReport(passed=True)
        if world.space is not None:
            report = report.merge(self.check_space(world.space))
        if world.nested_universe is not None:
            report = report.merge(self.check_universe(world.nested_universe))
        return report

    def check_universe(self, universe: Universe) -> InvariantReport:
        """Recursively check every world (and any nested universes) in a `Universe`."""
        report = InvariantReport(passed=True)
        for world in universe.worlds.values():
            report = report.merge(self.check_world(world))
        return report

    def check(self, obj: Any) -> InvariantReport:
        """Dispatch to the right `check_*` method based on `obj`'s type.

        Accepts a `Room`, `QESSpace`, `World`, or `Universe`.

        Raises:
            TypeError: if `obj` is not one of the recognized QES container
                types.
        """
        from qes.room import Room
        from qes.space import QESSpace
        from qes.universe import Universe
        from qes.world import World

        if isinstance(obj, Room):
            return self.check_room(obj)
        if isinstance(obj, QESSpace):
            return self.check_space(obj)
        if isinstance(obj, World):
            return self.check_world(obj)
        if isinstance(obj, Universe):
            return self.check_universe(obj)
        raise TypeError(
            f"InvariantEngine.check() does not support {type(obj).__name__}; "
            "expected Room, QESSpace, World, or Universe"
        )


__all__ = ["InvariantEngine", "InvariantReport", "InvariantViolation"]
