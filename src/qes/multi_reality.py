"""Multi-Reality Domain and VQCE — the QES virtual-world fabric.

Source-derived components (see the PDF-extraction summary, "Multi-Reality
Domain" and "VQCE"):

    Multi-Reality Domain
        Simulation Engine
        Virtual Node Projection
        Pattern Replication Layer

    H^11 Virtual Quantum-less Compute Engine (VQCE)
        Runs *inside* COSMIC VP, explicitly without physical quantum
        gates/qubits/coherent photons. Flow:

            COSMIC VP -> allocate virtual states/storage -> H^11 metric
            stabilization -> ACROS V12-BIE computation -> dual-engine
            validation -> stabilized output state

`SimulationEngine` and `VirtualNodeProjection` build directly on the
existing room/reality-generation primitives (`Room`, `RealityGenerator`,
`QESSpace`); `PatternReplicationLayer` builds on `PatternMemory`; `VQCE`
is a new, self-contained virtual compute engine implementing the flow
above without any dependency on real quantum hardware or libraries.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qes.patterns import Pattern, PatternMemory
from qes.reality_generator import RealityGenerator
from qes.room import Room

_node_id_counter = itertools.count(1)

RoomStepFn = Callable[[Room, float, float], np.ndarray]
VQCEComputeFn = Callable[[np.ndarray], np.ndarray]
VQCEValidateFn = Callable[[np.ndarray, np.ndarray], bool]


def _next_node_id() -> str:
    return f"N-{next(_node_id_counter):05d}"


def _as_finite_vector(value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_non_negative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


class SimulationEngine:
    """Runs a single room forward under one or more candidate step
    functions, in parallel "realities" (one simulated trajectory per
    candidate), without mutating the source room."""

    def __init__(self, generator: RealityGenerator | None = None) -> None:
        self.generator = generator or RealityGenerator()

    def simulate(
        self,
        room: Room,
        step_fns: list[RoomStepFn],
        steps: int,
        dt: float = 1.0,
    ) -> list[Room]:
        """Evolve independent copies of `room`, one per step function in
        `step_fns`, for `steps` ticks each; returns the list of final
        (evolved) room copies."""
        if not isinstance(room, Room):
            raise TypeError("room must be a Room")
        _validate_non_negative_int(steps, "steps")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and > 0")

        step_functions = list(step_fns)
        for step_fn in step_functions:
            if not callable(step_fn):
                raise TypeError("step_fns must contain callables")

        results: list[Room] = []
        for step_fn in step_functions:
            trial = room.clone()
            t = 0.0
            for _ in range(steps):
                next_state = _as_finite_vector(step_fn(trial, t, dt), "step result")
                if next_state.shape != trial.x.shape:
                    raise ValueError(
                        "step result shape must match room.x shape "
                        f"{trial.x.shape}, got {next_state.shape}"
                    )
                trial.x = next_state
                t += dt
            results.append(trial)
        return results


@dataclass
class VirtualNode:
    """A projected virtual node hosting one branched reality."""

    room: Room
    id: str = field(default_factory=_next_node_id)

    def __post_init__(self) -> None:
        if not isinstance(self.room, Room):
            raise TypeError("room must be a Room")


class VirtualNodeProjection:
    """Projects a room into a population of virtual compute nodes, one per
    branched child reality -- the Multi-Reality analogue of provisioning a
    COSMIC VP VM per candidate reality."""

    def __init__(self, generator: RealityGenerator | None = None) -> None:
        self.generator = generator or RealityGenerator()

    def project(self, room: Room, count: int, scale: float = 0.05) -> list[VirtualNode]:
        """Branch `room` into `count` children and wrap each in a `VirtualNode`."""
        if not isinstance(room, Room):
            raise TypeError("room must be a Room")
        _validate_non_negative_int(count, "count")
        scale = float(scale)
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("scale must be finite and >= 0")
        if count == 0:
            return []

        children = self.generator.branch(room, count=count, scale=scale)
        return [VirtualNode(room=child) for child in children]


class PatternReplicationLayer:
    """Replicates a winning pattern across a population of virtual nodes,
    seeding each node's room memory with the pattern's payload so future
    generations at every node start from the validated configuration
    rather than searching from zero."""

    def __init__(self, memory: PatternMemory | None = None) -> None:
        self.memory = memory or PatternMemory()

    def replicate(self, pattern: Pattern, nodes: list[VirtualNode]) -> int:
        """Copy `pattern` into every node's room memory; returns the count
        of nodes updated."""
        if not isinstance(pattern, Pattern):
            raise TypeError("pattern must be a Pattern")

        node_list = list(nodes)
        for node in node_list:
            if not isinstance(node, VirtualNode):
                raise TypeError("nodes must contain VirtualNode instances")
            patterns = node.room.memory.setdefault("patterns", [])
            if not isinstance(patterns, list):
                raise TypeError("room.memory['patterns'] must be a list when present")
            if pattern.id not in patterns:
                patterns.append(pattern.id)
        self.memory.store(pattern)
        return len(node_list)

    def best_for(self, intent: str, context: dict[str, Any] | None = None) -> Pattern | None:
        """Return the best remembered pattern for `intent`, optionally scoped
        by a context filter."""
        return self.memory.generate(intent, context or {})


@dataclass
class VQCEResult:
    """Stabilized output of one VQCE compute cycle."""

    state: np.ndarray
    stabilized: bool
    validated: bool


class VQCE:
    """H^11 Virtual Quantum-less Compute Engine.

    Explicitly *not* a quantum computer: this models the source's described
    flow -- allocate a virtual state, stabilize it against a metric
    baseline `h11`, run a supplied `compute_fn` (standing in for ACROS
    V12-BIE computation), then dual-engine-validate the result (agreement
    between two independent evaluations of the same computation) before
    returning a stabilized output.
    """

    def __init__(self, h11: float = 1.0, stability_tolerance: float = 1.0) -> None:
        h11 = float(h11)
        stability_tolerance = float(stability_tolerance)
        if not np.isfinite(h11) or h11 <= 0:
            raise ValueError("h11 must be finite and positive")
        if not np.isfinite(stability_tolerance) or stability_tolerance <= 0:
            raise ValueError("stability_tolerance must be finite and positive")
        self.h11 = h11
        self.stability_tolerance = stability_tolerance
        self._states: dict[str, np.ndarray] = {}

    def allocate_state(self, dim: int, name: str | None = None) -> str:
        """Allocate a virtual compute state (analogous to COSMIC VP
        allocating virtual storage for this computation)."""
        _validate_non_negative_int(dim, "dim")
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("name must be a string")
            if not name:
                raise ValueError("name must be non-empty")
            if name in self._states:
                raise ValueError(f"state {name!r} is already allocated")

        key = name or f"S-{len(self._states) + 1:05d}"
        self._states[key] = np.zeros(dim, dtype=float)
        return key

    def _stabilize(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        """H^11 metric stabilization: scale the state so its norm stays
        within `stability_tolerance * h11`."""
        x = _as_finite_vector(x, "x")
        norm = float(np.linalg.norm(x))
        limit = self.stability_tolerance * self.h11
        if norm <= limit or norm == 0.0:
            return x, True
        return x * (limit / norm), False

    def compute(
        self,
        state_key: str,
        x0: np.ndarray,
        compute_fn: VQCEComputeFn,
        validate_fn: VQCEValidateFn | None = None,
    ) -> VQCEResult:
        """Run one VQCE cycle: stabilize -> compute -> dual-engine validate."""
        if not isinstance(state_key, str):
            raise TypeError("state_key must be a string")
        if state_key not in self._states:
            raise KeyError(f"unallocated VQCE state: {state_key!r}")
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")
        if validate_fn is not None and not callable(validate_fn):
            raise TypeError("validate_fn must be callable")

        initial = _as_finite_vector(x0, "x0")
        allocated = self._states[state_key]
        if initial.shape != allocated.shape:
            raise ValueError(
                f"x0 shape {initial.shape} does not match allocated state shape {allocated.shape}"
            )

        stabilized_x, was_stable = self._stabilize(initial)
        primary = _as_finite_vector(compute_fn(stabilized_x.copy()), "compute_fn result")
        secondary = _as_finite_vector(compute_fn(stabilized_x.copy()), "compute_fn result")
        if primary.shape != stabilized_x.shape or secondary.shape != stabilized_x.shape:
            raise ValueError("compute_fn must preserve the allocated state shape")

        validated = bool(np.array_equal(primary, secondary) or np.allclose(primary, secondary))
        if validate_fn is not None:
            validated = validated and bool(validate_fn(primary, secondary))

        output, _ = self._stabilize(primary)
        self._states[state_key] = output
        return VQCEResult(state=output, stabilized=was_stable, validated=validated)
