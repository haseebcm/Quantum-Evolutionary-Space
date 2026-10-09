"""QESSpace — the top-level engine (docs/QES-architecture.md, sections 1, 30-32).

    Q(t) = { (R_i(t), p_i(t)) },  i = 1..N(t)

QES master operator:

    Q_{t+dt} = C o S o P o D o E o G (Q_t, Y_{t+dt})

    Generate -> Execute -> Measure Divergence -> Check Permission
             -> Select Survivors -> Converge -> Regenerate

This module wires together the room population, divergence tracking (DSA),
the Genesis permission kernel, and the convergence metric into a single
step()/run() loop. It intentionally keeps the room-level dynamics/selection
pluggable: callers supply how rooms evolve, are scored, and how many children
survive, while QESSpace enforces the overall generate/execute/measure/permit/
select/converge cycle and bookkeeping (weights, lifecycle, entropy).
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from qes.convergence import convergence_coefficient, qes_entropy
from qes.divergence import DSA
from qes.domain import DomainNullification
from qes.orchestrator import RoomLifecycle
from qes.permission import GenesisPermission
from qes.room import Room

# A step function evolves a room's state one tick forward: (room, t, dt) -> new_x
StepFn = Callable[[Room, float, float], np.ndarray]


@dataclass
class SpaceTelemetry:
    """Snapshot of Q(t) used for the "living universe" view (doc section 19)."""

    time: float
    active: int
    collapsed: int
    total_generated: int
    entropy: float
    convergence: float
    dominant_room_id: str | None
    dominant_permission: float


class QESSpace:
    """Q(t): the governed possibility space of competing rooms."""

    def __init__(
        self,
        permission_gate: GenesisPermission,
        domain: DomainNullification | None = None,
        step_fn: StepFn | None = None,
        dt: float = 1.0,
        max_workers: int | None = None,
    ):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.permission_gate = permission_gate
        self.domain = domain
        self.step_fn = step_fn
        self.dt = dt
        self.max_workers = max_workers

        self.rooms: dict = {}  # id -> Room
        self._dsa: dict = {}  # id -> DSA tracker
        self._active_cache: list | None = None
        self.time: float = 0.0
        self.total_generated: int = 0
        self.total_collapsed: int = 0
        self.history: list = []

    # ------------------------------------------------------------------
    # Generation (G)
    # ------------------------------------------------------------------
    def add_room(self, room: Room) -> None:
        if room.id in self.rooms:
            raise ValueError(f"duplicate room id: {room.id}")
        room.state = "Active" if room.state == "Seed" else room.state
        self.rooms[room.id] = room
        self._dsa[room.id] = DSA()
        self.total_generated += 1
        self._active_cache = None

    def spawn(self, rooms: Sequence[Room]) -> None:
        for room in rooms:
            self.add_room(room)

    # ------------------------------------------------------------------
    # Execution (E)
    # ------------------------------------------------------------------
    def _default_step(self, room: Room, t: float, dt: float) -> np.ndarray:
        return room.x

    def execute(self) -> None:
        """Advance every active room's state one tick (E)."""
        step = self.step_fn or self._default_step
        active = self.active_rooms()
        if self.max_workers and self.max_workers > 1 and len(active) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                new_states = list(
                    pool.map(lambda room: step(room, self.time, self.dt), active)
                )
            for room, new_state in zip(active, new_states, strict=True):
                room.x = np.asarray(new_state, dtype=float)
        else:
            for room in active:
                room.x = np.asarray(step(room, self.time, self.dt), dtype=float)

    # ------------------------------------------------------------------
    # Divergence measurement (D)
    # ------------------------------------------------------------------
    def measure_divergence(self) -> dict:
        """Update DSA/DR/HSA for every active room. Returns id -> DivergenceResult."""
        results = {}
        domain = self.domain
        for room in self.active_rooms():
            dsa = self._dsa[room.id]
            w = domain.metric(room.activation) if domain is not None else None
            result = dsa.update(room.x, room.x_star, self.dt, w)
            room.memory["divergence"] = result
            results[room.id] = result
        return results

    # ------------------------------------------------------------------
    # Permission check (P)
    # ------------------------------------------------------------------
    def check_permission(self) -> dict:
        """Evaluate the Genesis permission kernel for every active room."""
        results = {}
        active = self.active_rooms()
        collapsed_any = False
        eval_fn = self.permission_gate.evaluate
        for room in active:
            cci_weights = room.gates.get("cci_weights")
            result = eval_fn(
                room.x, room.lower, room.upper, cci_weights, room.couplings
            )
            room.memory["permission"] = result
            room.weight = result.soft_permission
            results[room.id] = result
            if not result.admitted:
                RoomLifecycle.transition(room, "Collapsed")
                self.total_collapsed += 1
                collapsed_any = True
        if collapsed_any:
            self._active_cache = None
        return results

    # ------------------------------------------------------------------
    # Selection (S)
    # ------------------------------------------------------------------
    def select(self, survivors_per_kind: int | None = None,
               signature_fn: Callable[[Room], object] | None = None) -> None:
        """Prune surviving rooms; optionally keep only the top-N per signature kind."""
        if survivors_per_kind is not None and (
            not isinstance(survivors_per_kind, int) or isinstance(survivors_per_kind, bool)
            or survivors_per_kind < 0
        ):
            raise ValueError("survivors_per_kind must be a nonnegative integer")
        active = self.active_rooms()
        if not active:
            return
        if signature_fn is not None and survivors_per_kind is not None:
            groups: dict = {}
            for room in active:
                groups.setdefault(signature_fn(room), []).append(room)
            shadow_any = False
            for members in groups.values():
                ranked = sorted(members, key=lambda r: r.weight, reverse=True)
                for loser in ranked[survivors_per_kind:]:
                    RoomLifecycle.transition(loser, "Shadow")
                    shadow_any = True
            if shadow_any:
                self._active_cache = None

    # ------------------------------------------------------------------
    # Convergence (C)
    # ------------------------------------------------------------------
    def convergence(self) -> float:
        weights = [r.weight for r in self.active_rooms()]
        if not weights:
            return 0.0
        return convergence_coefficient(weights)

    def entropy(self) -> float:
        weights = [r.weight for r in self.active_rooms()]
        return qes_entropy(weights) if weights else 0.0

    # ------------------------------------------------------------------
    # Master operator: Q_{t+dt} = C o S o P o D o E o G (Q_t, Y_{t+dt})
    # ------------------------------------------------------------------
    def step(self) -> SpaceTelemetry:
        """Advance the whole space by one tick through E -> D -> P -> S -> C."""
        self.execute()
        self.measure_divergence()
        self.check_permission()
        self.select()
        self.time += self.dt
        telemetry = self.telemetry()
        self.history.append(telemetry)
        return telemetry

    def run(self, steps: int) -> list:
        return [self.step() for _ in range(steps)]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def active_rooms(self) -> list:
        if self._active_cache is None:
            self._active_cache = [r for r in self.rooms.values() if r.state == "Active"]
        return list(self._active_cache)

    def collapsed_rooms(self) -> list:
        return [r for r in self.rooms.values() if r.state == "Collapsed"]

    def dominant_room(self) -> Room | None:
        active = self.active_rooms()
        if not active:
            return None
        return max(active, key=lambda r: r.weight)

    def telemetry(self) -> SpaceTelemetry:
        active = self.active_rooms()
        n_active = len(active)
        if n_active > 0:
            dominant = max(active, key=lambda r: r.weight)
            weights = [r.weight for r in active]
            ent = qes_entropy(weights)
            conv = convergence_coefficient(weights)
            dom_id = dominant.id
            dom_perm = dominant.weight
        else:
            ent = 0.0
            conv = 0.0
            dom_id = None
            dom_perm = 0.0

        return SpaceTelemetry(
            time=self.time,
            active=n_active,
            collapsed=self.total_collapsed,
            total_generated=self.total_generated,
            entropy=ent,
            convergence=conv,
            dominant_room_id=dom_id,
            dominant_permission=dom_perm,
        )

    # ------------------------------------------------------------------
    # Reality branching at the space level: clone/snapshot/restore
    # ------------------------------------------------------------------
    def clone(self) -> QESSpace:
        """Deep-copy this whole space (rooms, DSA trackers, history) so it can
        diverge independently -- reality branching promoted to the space level."""
        # deepcopy preserves aliases within the branch (e.g. strategy RNG state)
        # while isolating it from the original. Plain functions/closures remain
        # shared; callers must supply stateless callbacks or cloneable objects.
        clone = copy.deepcopy(self)
        clone._active_cache = None
        return clone

    def snapshot(self) -> dict:
        """Capture a restorable checkpoint of this space's full mutable state."""
        return {
            "rooms": copy.deepcopy(self.rooms),
            "dsa": copy.deepcopy(self._dsa),
            "time": self.time,
            "total_generated": self.total_generated,
            "total_collapsed": self.total_collapsed,
            "history": copy.deepcopy(self.history),
            "permission_gate": copy.deepcopy(self.permission_gate),
            "domain": copy.deepcopy(self.domain),
            "step_fn": copy.deepcopy(self.step_fn),
            "dt": self.dt,
        }

    def restore(self, snapshot: dict) -> None:
        """Restore state previously captured by `snapshot()`."""
        self.rooms = copy.deepcopy(snapshot["rooms"])
        self._dsa = copy.deepcopy(snapshot["dsa"])
        self.time = snapshot["time"]
        self.total_generated = snapshot["total_generated"]
        self.total_collapsed = snapshot["total_collapsed"]
        self.history = copy.deepcopy(snapshot["history"])
        self.permission_gate = copy.deepcopy(snapshot.get("permission_gate", self.permission_gate))
        self.domain = copy.deepcopy(snapshot.get("domain", self.domain))
        self.step_fn = copy.deepcopy(snapshot.get("step_fn", self.step_fn))
        self.dt = snapshot.get("dt", self.dt)
        self._active_cache = None
