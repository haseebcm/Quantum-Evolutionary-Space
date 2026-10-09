"""High-level developer-facing SDK for common QES search workflows.

This module stays honest about what it provides: a small classical orchestration
layer over existing QES primitives. It does not introduce a new optimizer or a
new permission model. Instead it wires together:

* `RealityGenerator` for initial population branching,
* `QESSpace` for governed possibility-space execution,
* `GenesisPermission` or `AdaptivePermission` for admissibility checks, and
* `PatternMemory` for lightweight result retention across runs.

Use `QESClient` when you want the framework's existing pieces assembled with a
minimal, ergonomic API, and `quick_search()` when a one-liner is enough.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt

from qes.execution import CountedObjective, ExecutionBudget, ExecutionStopped
from qes.intelligence import AdaptiveGradientSearch, AdaptiveSearchConfig
from qes.patterns import Pattern, PatternMemory
from qes.permission import AdaptivePermission, GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace, SpaceTelemetry, StepFn

ObjectiveFn = Callable[[npt.NDArray[np.float64]], float]
RoomScoreFn = Callable[[Room], float]
Bounds = tuple[npt.ArrayLike, npt.ArrayLike]


def _as_vector(
    name: str,
    value: npt.ArrayLike,
    *,
    shape: tuple[int, ...] | None = None,
) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match expected shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _coerce_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, int) and not isinstance(rng, bool):
        return np.random.default_rng(rng)
    raise TypeError("rng must be a numpy.random.Generator, an integer seed, or None")


@dataclass(frozen=True)
class SDKPermissionConfig:
    """Configuration for the SDK's permission gate selection."""

    theta: float = 10.0
    adaptive: bool = False
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    eps_phi: float = 1e-9
    m_min: float = 0.0
    target_rate: float = 0.5
    adapt_rate: float = 0.05
    theta_min: float = 1e-6
    theta_max: float | None = None
    window: int = 20

    def __post_init__(self) -> None:
        for name in ("theta", "alpha", "beta", "gamma", "eps_phi", "m_min", "theta_min"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in {"theta", "alpha", "beta", "gamma", "eps_phi", "theta_min"} and value <= 0.0:
                raise ValueError(f"{name} must be > 0")
            object.__setattr__(self, name, value)

        object.__setattr__(self, "target_rate", float(self.target_rate))
        object.__setattr__(self, "adapt_rate", float(self.adapt_rate))
        if not 0.0 <= self.target_rate <= 1.0:
            raise ValueError("target_rate must lie in [0, 1]")
        if not np.isfinite(self.adapt_rate) or self.adapt_rate < 0.0:
            raise ValueError("adapt_rate must be finite and >= 0")
        if not isinstance(self.window, int) or self.window <= 0:
            raise ValueError("window must be an integer > 0")
        if self.theta_max is not None:
            theta_max = float(self.theta_max)
            if not np.isfinite(theta_max) or theta_max < self.theta_min:
                raise ValueError("theta_max must be finite and >= theta_min")
            object.__setattr__(self, "theta_max", theta_max)

    def build(self) -> GenesisPermission:
        """Construct the configured permission gate."""
        common = dict(
            theta=self.theta,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            eps_phi=self.eps_phi,
            m_min=self.m_min,
        )
        if self.adaptive:
            return AdaptivePermission(
                **common,
                target_rate=self.target_rate,
                adapt_rate=self.adapt_rate,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                window=self.window,
            )
        return GenesisPermission(**common)


@dataclass(frozen=True)
class SDKRunResult:
    """Summary returned by `QESClient.run()` and `quick_search()`."""

    best_room_id: str | None
    best_state: npt.NDArray[np.float64] | None
    best_score: float | None
    final_entropy: float
    final_convergence: float
    dominant_permission: float
    active_rooms: int
    collapsed_rooms: int
    total_generated: int
    remembered_patterns: int
    wall_time_seconds: float
    steps: int
    telemetry: SpaceTelemetry
    evaluations: int = 0
    stopping_reason: str = "steps_completed"

    @property
    def status(self) -> str:
        """Whether a certified admissible candidate was returned."""
        return "success" if self.best_state is not None else "no_feasible_solution"


class QESClient:
    """Convenience wrapper that assembles a governed QES search in one place.

    Provide either an `objective` (which uses the existing
    `AdaptiveGradientSearch` step operator) or a room-level `step_fn` of your
    own. The client then builds the initial population, permission gate,
    `QESSpace`, and `PatternMemory` around that existing computation.
    """

    def __init__(
        self,
        bounds: Bounds,
        *,
        step_fn: StepFn | None = None,
        objective: ObjectiveFn | None = None,
        target: npt.ArrayLike | None = None,
        seed_state: npt.ArrayLike | None = None,
        score_fn: RoomScoreFn | None = None,
        population: int = 40,
        branch_scale: float = 0.25,
        permission: SDKPermissionConfig | None = None,
        rng: np.random.Generator | int | None = None,
        search_config: AdaptiveSearchConfig | None = None,
        dt: float = 1.0,
        max_workers: int | None = None,
        intent: str = "sdk-search",
        max_evaluations: int | None = None,
        max_wall_time: float | None = None,
        survivors_per_kind: int | None = None,
        signature_fn: Callable[[Room], object] | None = None,
    ) -> None:
        if (step_fn is None) == (objective is None):
            raise ValueError("provide exactly one of step_fn or objective")
        if objective is not None and score_fn is not None:
            raise ValueError("score_fn cannot be combined with objective")
        if step_fn is not None and search_config is not None:
            raise ValueError("search_config can only be used with objective-driven searches")
        if not isinstance(population, int) or population <= 0:
            raise ValueError("population must be an integer > 0")
        if not np.isfinite(branch_scale) or branch_scale < 0.0:
            raise ValueError("branch_scale must be finite and >= 0")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0")
        if max_workers is not None and (not isinstance(max_workers, int) or max_workers <= 0):
            raise ValueError("max_workers must be None or an integer > 0")
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")

        lower = _as_vector("lower bound", bounds[0])
        upper = _as_vector("upper bound", bounds[1], shape=lower.shape)
        if np.any(lower > upper):
            raise ValueError("lower bound must be <= upper bound elementwise")

        midpoint = lower + 0.5 * (upper - lower)
        seed = _as_vector("seed_state", midpoint if seed_state is None else seed_state, shape=lower.shape)
        if np.any(seed < lower) or np.any(seed > upper):
            raise ValueError("seed_state must lie within the provided bounds")

        reference = _as_vector("target", midpoint if target is None else target, shape=lower.shape)

        self.lower = lower
        self.upper = upper
        self.seed_state = seed
        self.target = reference
        self.population = population
        self.branch_scale = float(branch_scale)
        self.intent = intent
        self.rng = _coerce_rng(rng)
        self.permission_config = permission or SDKPermissionConfig()
        self.permission_gate = self.permission_config.build()
        self.memory = PatternMemory()
        self.generator = RealityGenerator(rng=self.rng)
        self.seed_room = Room(
            x=self.seed_state.copy(),
            x_star=self.target.copy(),
            lower=self.lower.copy(),
            upper=self.upper.copy(),
            activation=np.ones_like(self.lower),
        )

        if max_evaluations is not None and objective is None:
            raise ValueError("max_evaluations requires an objective")
        self.budget = ExecutionBudget(max_evaluations)
        self.budget.start(max_wall_time)
        self.max_wall_time = max_wall_time
        if (survivors_per_kind is None) != (signature_fn is None):
            raise ValueError("provide both survivors_per_kind and signature_fn")
        if survivors_per_kind is not None and (
            not isinstance(survivors_per_kind, int) or isinstance(survivors_per_kind, bool) or survivors_per_kind < 0
        ):
            raise ValueError("survivors_per_kind must be a nonnegative integer")
        self.survivors_per_kind = survivors_per_kind
        self.signature_fn = signature_fn
        self._objective = objective
        self._score_fn = score_fn
        self._search = (
            AdaptiveGradientSearch(objective=CountedObjective(objective, self.budget), config=search_config, rng=self.rng)
            if objective is not None
            else None
        )
        self.step_fn = self._search if self._search is not None else step_fn
        assert self.step_fn is not None

        self.space = QESSpace(
            permission_gate=self.permission_gate,
            step_fn=self.step_fn,
            dt=float(dt),
            max_workers=max_workers,
        )
        children = self.generator.branch(
            self.seed_room,
            count=self.population,
            scale=self.branch_scale,
        )
        for child in children:
            child.x = np.clip(child.x, child.lower, child.upper)
        self.space.spawn(children)

    def dominant_room(self) -> Room | None:
        """Return the current dominant room in the underlying `QESSpace`."""
        return self.space.dominant_room()

    def remembered_patterns(self) -> list[Pattern]:
        """Return retained patterns for this client's intent."""
        return self.memory.all_patterns(self.intent)

    def _candidate_for_room(
        self,
        room: Room,
    ) -> tuple[npt.NDArray[np.float64], float | None]:
        if self._search is not None:
            best_x, best_value = self._search.best_known(room)
            return best_x, float(best_value)

        state = room.x.copy()
        if self._score_fn is None:
            return state, None

        score = float(self._score_fn(room))
        if not np.isfinite(score):
            raise ValueError("score_fn must return a finite scalar")
        return state, score

    def _best_snapshot(
        self,
    ) -> tuple[str | None, npt.NDArray[np.float64] | None, float | None, float]:
        rooms = self.space.active_rooms()
        if not rooms:
            return None, None, None, 0.0

        if self._search is None and self._score_fn is None:
            dominant = self.space.dominant_room()
            if dominant is None or not self.space.permission_gate.inspect(
                dominant.x, dominant.lower, dominant.upper,
                dominant.gates.get("cci_weights"), dominant.couplings,
            ).admitted:
                return None, None, None, 0.0
            return dominant.id, dominant.x.copy(), None, dominant.weight

        ranked: list[tuple[Room, npt.NDArray[np.float64], float | None]] = [
            (room, *self._candidate_for_room(room)) for room in rooms
            if self._search is None or self._search.config.memory_key in room.memory
        ]
        scored: list[tuple[Room, npt.NDArray[np.float64], float]] = [
            (room, state, score)
            for room, state, score in ranked
            if score is not None and self.space.permission_gate.inspect(
                state, room.lower, room.upper, room.gates.get("cci_weights"), room.couplings
            ).admitted
        ]
        if scored:
            room, best_state, best_score = min(scored, key=lambda item: item[2])
            return room.id, best_state.copy(), best_score, room.weight

        return None, None, None, 0.0

    def _record_pattern(self, telemetry: SpaceTelemetry) -> None:
        room_id, best_state, best_score, permission_margin = self._best_snapshot()
        if room_id is None or best_state is None:
            return

        pattern = Pattern(
            intent=self.intent,
            context={"step": len(self.space.history), "time": telemetry.time, "room_id": room_id},
            payload=best_state.tolist(),
            phi=max(0.0, 1.0 - telemetry.convergence),
            cci=0.0 if best_score is None else best_score,
            margin=permission_margin,
        )
        self.memory.store(pattern)
        self.memory.retire_dominated(self.intent)

    def cancel(self) -> None:
        """Refuse subsequent callback calls; running callbacks finish normally."""
        self.budget.cancel()

    def run(self, steps: int) -> SDKRunResult:
        """Advance the search for up to `steps` QES ticks and return a summary."""
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be an integer >= 0")

        # Restore may have replaced a cloneable strategy/budget object graph.
        if isinstance(self.space.step_fn, AdaptiveGradientSearch):
            self._search = self.space.step_fn
            if isinstance(self._search.objective, CountedObjective):
                self.budget = self._search.objective.budget
        self.budget.start(self.max_wall_time)
        started = perf_counter()
        executed_steps = 0
        stopping_reason = "steps_completed"
        for _ in range(steps):
            if not self.space.active_rooms():
                stopping_reason = "no_active_rooms"
                break
            try:
                self.budget.check()
                telemetry = self.space.step()
            except ExecutionStopped as exc:
                stopping_reason = exc.reason
                break
            if self.signature_fn is not None:
                self.space.select(self.survivors_per_kind, self.signature_fn)
                telemetry = self.space.telemetry()
            self._record_pattern(telemetry)
            executed_steps += 1
        wall_time = perf_counter() - started

        telemetry = self.space.telemetry()
        best_room_id, best_state, best_score, _ = self._best_snapshot()
        return SDKRunResult(
            best_room_id=best_room_id,
            best_state=None if best_state is None else best_state.copy(),
            best_score=best_score,
            final_entropy=telemetry.entropy,
            final_convergence=telemetry.convergence,
            dominant_permission=telemetry.dominant_permission,
            active_rooms=telemetry.active,
            collapsed_rooms=telemetry.collapsed,
            total_generated=telemetry.total_generated,
            remembered_patterns=len(self.memory.all_patterns(self.intent)),
            wall_time_seconds=wall_time,
            steps=executed_steps,
            telemetry=telemetry,
            evaluations=self.budget.evaluations,
            stopping_reason=stopping_reason,
        )


def quick_search(
    bounds: Bounds,
    *,
    step_fn: StepFn | None = None,
    objective: ObjectiveFn | None = None,
    target: npt.ArrayLike | None = None,
    seed_state: npt.ArrayLike | None = None,
    score_fn: RoomScoreFn | None = None,
    steps: int = 50,
    population: int = 40,
    branch_scale: float = 0.25,
    permission: SDKPermissionConfig | None = None,
    rng: np.random.Generator | int | None = None,
    search_config: AdaptiveSearchConfig | None = None,
    dt: float = 1.0,
    max_workers: int | None = None,
    intent: str = "quick-search",
    max_evaluations: int | None = None,
    max_wall_time: float | None = None,
) -> SDKRunResult:
    """One-shot helper for a full QES search run."""
    client = QESClient(
        bounds,
        step_fn=step_fn,
        objective=objective,
        target=target,
        seed_state=seed_state,
        score_fn=score_fn,
        population=population,
        branch_scale=branch_scale,
        permission=permission,
        rng=rng,
        search_config=search_config,
        dt=dt,
        max_workers=max_workers,
        intent=intent,
        max_evaluations=max_evaluations,
        max_wall_time=max_wall_time,
    )
    return client.run(steps)


__all__ = [
    "QESClient",
    "SDKPermissionConfig",
    "SDKRunResult",
    "quick_search",
]
