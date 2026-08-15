"""Reusable domain packs for classical QES search and allocation workflows.

These adapters package a few common, fully classical problem shapes on top of
the existing QES primitives so callers do not have to repeat the same
``Room``/``RealityGenerator``/``QESSpace`` setup boilerplate. The packs use the
framework's quantum/universe vocabulary, but every computation here is ordinary
NumPy-based simulation, local search, and resource allocation.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from qes.compute_allocator import ComputeAllocator, RoomComputeProfile
from qes.convergence import convergence_coefficient, gini_coefficient
from qes.dynamics import RoomDynamics
from qes.permission import AdaptivePermission, GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.selection import GenesisSelectionPipeline
from qes.space import QESSpace
from qes.state_space import MCCStateSpace

FloatArray = npt.NDArray[np.float64]
ConstraintFn = Callable[[FloatArray], float]
CostFn = Callable[[FloatArray], float]


def _as_1d_array(name: str, value: Sequence[float] | npt.ArrayLike, dim: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if dim is not None and array.shape[0] != dim:
        raise ValueError(f"{name} must have length {dim}, got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(float, copy=True)


def _as_scalar_or_vector(name: str, value: float | Sequence[float], dim: int) -> FloatArray:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _as_1d_array(name, value, dim=dim)
    scalar_array = np.asarray(value, dtype=float)
    if scalar_array.ndim != 0:
        raise ValueError(f"{name} must be a scalar or a length-{dim} vector")
    scalar = float(scalar_array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return np.full(dim, scalar, dtype=float)


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_non_negative_float(name: str, value: float) -> float:
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite value >= 0")
    return normalized


def _validate_positive_float(name: str, value: float) -> float:
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite value > 0")
    return normalized


def _validate_bounds(
    lower_name: str,
    lower: Sequence[float] | npt.ArrayLike,
    upper_name: str,
    upper: Sequence[float] | npt.ArrayLike,
    *,
    expected_dim: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    lower_array = _as_1d_array(lower_name, lower, dim=expected_dim)
    upper_array = _as_1d_array(upper_name, upper, dim=expected_dim)
    if lower_array.shape != upper_array.shape:
        raise ValueError(f"{lower_name} and {upper_name} must share the same shape")
    if np.any(lower_array > upper_array):
        raise ValueError(f"{lower_name} must be <= {upper_name} elementwise")
    return lower_array, upper_array


def _normalized_weights(weights: Sequence[float]) -> FloatArray:
    array = np.asarray(weights, dtype=float)
    total = float(array.sum())
    if total <= 0.0:
        return np.zeros_like(array)
    return array / total


@dataclass(frozen=True)
class DomainPackDescriptor:
    """Human-readable metadata for one available domain pack."""

    name: str
    builder: str
    summary: str
    returns: str


@dataclass(frozen=True)
class ControlPolicyPackConfig:
    """Configuration for a classical control-policy gain search.

    Attributes:
        state_dim: Number of plant state dimensions.
        target_state: Desired plant state.
        state_lower: Lower admissible plant-state bound used in simulation penalties.
        state_upper: Upper admissible plant-state bound used in simulation penalties.
        initial_state: Initial plant state for policy evaluation.
        controller_kind: ``"linear"`` for one proportional gain per state, or
            ``"pid"`` for per-state ``Kp/Ki/Kd`` gains.
        plant_decay: First-order drift coefficient ``a`` in
            ``dx/dt = -a * x + b * u``; scalar or per-state vector.
        plant_gain: Control coefficient ``b`` in ``dx/dt = -a * x + b * u``;
            scalar or per-state vector.
        gain_lower: Lower bounds for the controller gain search space.
        gain_upper: Upper bounds for the controller gain search space.
        initial_gains: Seed controller gains.
        simulation_steps: Closed-loop rollout length for scoring a candidate.
        simulation_dt: Closed-loop simulation time step.
        control_penalty: Penalty multiplier on squared control effort.
        boundary_penalty: Penalty multiplier on state-envelope violation energy.
        branch_count: Number of initial branched candidates.
        branch_scale: Gaussian branching scale for the initial population.
        mutation_scale: Gaussian local-search mutation scale in gain space.
        permission_theta: Genesis/adaptive permission threshold.
        adaptive_permission: Whether to use :class:`AdaptivePermission`.
        space_dt: QESSpace step size.
        rng_seed: Optional deterministic random seed.
        max_workers: Optional QESSpace worker count.
    """

    state_dim: int
    target_state: Sequence[float] | npt.ArrayLike
    state_lower: Sequence[float] | npt.ArrayLike
    state_upper: Sequence[float] | npt.ArrayLike
    initial_state: Sequence[float] | npt.ArrayLike | None = None
    controller_kind: Literal["linear", "pid"] = "pid"
    plant_decay: float | Sequence[float] = 0.5
    plant_gain: float | Sequence[float] = 1.0
    gain_lower: Sequence[float] | npt.ArrayLike | None = None
    gain_upper: Sequence[float] | npt.ArrayLike | None = None
    initial_gains: Sequence[float] | npt.ArrayLike | None = None
    simulation_steps: int = 30
    simulation_dt: float = 0.05
    control_penalty: float = 0.01
    boundary_penalty: float = 100.0
    branch_count: int = 40
    branch_scale: float = 0.5
    mutation_scale: float = 0.15
    permission_theta: float = 1.5
    adaptive_permission: bool = False
    space_dt: float = 1.0
    rng_seed: int | None = None
    max_workers: int | None = None


@dataclass(frozen=True)
class ControlPolicyPackResult:
    """Summary of one classical control-policy search run."""

    best_gains: FloatArray
    objective: float
    final_state: FloatArray
    final_distance_to_target: float
    admitted: bool
    iterations: int
    active_rooms: int
    collapsed_rooms: int
    convergence: float
    wall_time_seconds: float


class ControlPolicyPack:
    """Prebuilt QESSpace wrapper for classical controller-gain search."""

    def __init__(self, config: ControlPolicyPackConfig):
        self.config = config
        self._state_dim = _validate_positive_int("state_dim", config.state_dim)
        self._target_state = _as_1d_array("target_state", config.target_state, dim=self._state_dim)
        state_lower, state_upper = _validate_bounds(
            "state_lower",
            config.state_lower,
            "state_upper",
            config.state_upper,
            expected_dim=self._state_dim,
        )
        self._state_bounds = MCCStateSpace(
            lower=state_lower,
            upper=state_upper,
            reference=self._target_state,
            names=[f"x_{index}" for index in range(self._state_dim)],
        )
        if config.initial_state is None:
            self._initial_state: FloatArray = np.zeros(self._state_dim, dtype=float)
        else:
            self._initial_state = _as_1d_array("initial_state", config.initial_state, dim=self._state_dim)
        self._plant_decay = _as_scalar_or_vector("plant_decay", config.plant_decay, self._state_dim)
        self._plant_gain = _as_scalar_or_vector("plant_gain", config.plant_gain, self._state_dim)
        if np.any(self._plant_decay < 0.0):
            raise ValueError("plant_decay must be >= 0")
        if np.any(self._plant_gain <= 0.0):
            raise ValueError("plant_gain must be > 0")
        if config.controller_kind not in {"linear", "pid"}:
            raise ValueError("controller_kind must be 'linear' or 'pid'")
        self._controller_kind = config.controller_kind
        self._gain_dim = self._state_dim if self._controller_kind == "linear" else 3 * self._state_dim
        gain_lower, gain_upper = self._resolve_gain_bounds()
        self._gain_lower = gain_lower.copy()
        self._gain_upper = gain_upper.copy()
        initial_gains = self._resolve_initial_gains()
        self._gain_space = MCCStateSpace(
            lower=self._gain_lower,
            upper=self._gain_upper,
            reference=initial_gains,
            names=self._gain_names(),
        )
        self._simulation_steps = _validate_positive_int("simulation_steps", config.simulation_steps)
        self._simulation_dt = _validate_positive_float("simulation_dt", config.simulation_dt)
        self._control_penalty = _validate_non_negative_float("control_penalty", config.control_penalty)
        self._boundary_penalty = _validate_non_negative_float("boundary_penalty", config.boundary_penalty)
        self._branch_count = _validate_positive_int("branch_count", config.branch_count)
        self._branch_scale = _validate_non_negative_float("branch_scale", config.branch_scale)
        self._mutation_scale = _validate_non_negative_float("mutation_scale", config.mutation_scale)
        permission_theta = _validate_positive_float("permission_theta", config.permission_theta)
        self._space_dt = _validate_positive_float("space_dt", config.space_dt)
        if config.max_workers is not None and config.max_workers < 1:
            raise ValueError("max_workers must be >= 1 when provided")
        self._rng = np.random.default_rng(config.rng_seed)
        self._generator = RealityGenerator(rng=self._rng)
        self._noise = RoomDynamics()
        self._dynamics = RoomDynamics(
            drift=lambda x, t: -self._plant_decay * x,
            control_matrix=np.diag(self._plant_gain),
        )
        self.space = QESSpace(
            permission_gate=(
                AdaptivePermission(theta=permission_theta)
                if config.adaptive_permission
                else GenesisPermission(theta=permission_theta)
            ),
            step_fn=self._step_room,
            dt=self._space_dt,
            max_workers=config.max_workers,
        )
        self._seed_room = self._make_seed_room()
        self._spawn_initial_population()

    def _resolve_gain_bounds(self) -> tuple[FloatArray, FloatArray]:
        if self.config.gain_lower is None or self.config.gain_upper is None:
            if self._controller_kind == "linear":
                default_lower = np.zeros(self._gain_dim, dtype=float)
                default_upper = np.full(self._gain_dim, 10.0, dtype=float)
            else:
                default_lower = np.zeros(self._gain_dim, dtype=float)
                default_upper = np.concatenate(
                    [
                        np.full(self._state_dim, 10.0, dtype=float),
                        np.full(self._state_dim, 5.0, dtype=float),
                        np.full(self._state_dim, 2.0, dtype=float),
                    ]
                )
            if self.config.gain_lower is None:
                gain_lower: FloatArray = default_lower
            else:
                gain_lower = _as_1d_array("gain_lower", self.config.gain_lower, dim=self._gain_dim)
            if self.config.gain_upper is None:
                gain_upper: FloatArray = default_upper
            else:
                gain_upper = _as_1d_array("gain_upper", self.config.gain_upper, dim=self._gain_dim)
            if np.any(gain_lower > gain_upper):
                raise ValueError("gain_lower must be <= gain_upper elementwise")
            return gain_lower, gain_upper
        return _validate_bounds(
            "gain_lower",
            self.config.gain_lower,
            "gain_upper",
            self.config.gain_upper,
            expected_dim=self._gain_dim,
        )

    def _resolve_initial_gains(self) -> FloatArray:
        if self.config.initial_gains is None:
            if self._controller_kind == "linear":
                gains: FloatArray = np.ones(self._gain_dim, dtype=float)
            else:
                gains = np.concatenate(
                    [
                        np.full(self._state_dim, 1.0, dtype=float),
                        np.full(self._state_dim, 0.1, dtype=float),
                        np.full(self._state_dim, 0.01, dtype=float),
                    ]
                )
        else:
            gains = _as_1d_array("initial_gains", self.config.initial_gains, dim=self._gain_dim)
        if np.any(gains < self._gain_lower) or np.any(gains > self._gain_upper):
            raise ValueError("initial_gains must lie within [gain_lower, gain_upper]")
        return gains

    def _gain_names(self) -> list[str]:
        if self._controller_kind == "linear":
            return [f"k_{index}" for index in range(self._state_dim)]
        names: list[str] = []
        for prefix in ("kp", "ki", "kd"):
            names.extend(f"{prefix}_{index}" for index in range(self._state_dim))
        return names

    def _make_seed_room(self) -> Room:
        gains = self._gain_space.reference.copy()
        return Room(
            x=gains,
            x_star=gains.copy(),
            lower=self._gain_space.lower.copy(),
            upper=self._gain_space.upper.copy(),
            activation=np.ones(self._gain_dim, dtype=float),
        )

    def _spawn_initial_population(self) -> None:
        self.space.add_room(self._seed_room)
        children = self._generator.branch(self._seed_room, count=self._branch_count, scale=self._branch_scale)
        for child in children:
            child.x = self._gain_space.clip(child.x)
        self.space.spawn(children)

    def _control_input(
        self,
        gains: FloatArray,
        error: FloatArray,
        integral: FloatArray,
        derivative: FloatArray,
    ) -> FloatArray:
        if self._controller_kind == "linear":
            return gains * error
        kp = gains[: self._state_dim]
        ki = gains[self._state_dim : 2 * self._state_dim]
        kd = gains[2 * self._state_dim :]
        return kp * error + ki * integral + kd * derivative

    def simulate_policy(self, gains: Sequence[float] | npt.ArrayLike) -> tuple[FloatArray, float]:
        """Simulate a candidate controller and return ``(final_state, objective)``."""

        gain_array = _as_1d_array("gains", gains, dim=self._gain_dim)
        x = self._initial_state.copy()
        integral: FloatArray = np.zeros(self._state_dim, dtype=float)
        prev_error = self._target_state - x
        total_error = 0.0
        control_energy = 0.0
        boundary_violation = 0.0
        for _ in range(self._simulation_steps):
            error = self._target_state - x
            integral = integral + error * self._simulation_dt
            derivative = (error - prev_error) / self._simulation_dt
            control = self._control_input(gain_array, error, integral, derivative)
            x = self._dynamics.integrate_rk4(x, control, 0.0, self._simulation_dt)
            over = np.maximum(0.0, x - self._state_bounds.upper)
            under = np.maximum(0.0, self._state_bounds.lower - x)
            boundary_violation += float(np.sum(over**2 + under**2))
            total_error += float(np.mean(np.abs(error)) * self._simulation_dt)
            control_energy += float(np.mean(control**2) * self._simulation_dt)
            prev_error = error
        objective = (
            total_error
            + self._control_penalty * control_energy
            + self._boundary_penalty * boundary_violation
        )
        return x, float(objective)

    def evaluate_gains(self, gains: Sequence[float] | npt.ArrayLike) -> float:
        """Return the search objective for one candidate gain vector."""

        _final_state, objective = self.simulate_policy(gains)
        return objective

    def _step_room(self, room: Room, t: float, dt: float) -> FloatArray:
        current_objective = self.evaluate_gains(room.x)
        proposal = room.x + self._noise.stochastic_noise(room.dim, sigma=self._mutation_scale, rng=self._rng)
        proposal = self._gain_space.clip(proposal)
        proposal_objective = self.evaluate_gains(proposal)
        return proposal if proposal_objective < current_objective else cast(FloatArray, room.x.copy())

    def _best_room(self) -> tuple[Room, bool]:
        candidates = [room for room in self.space.rooms.values() if room.state != "Collapsed"]
        if candidates:
            pipeline = GenesisSelectionPipeline(
                gates=[lambda room: 0.0],
                signature_fn=lambda room: "controller",
                score_fn=lambda room: self.evaluate_gains(cast(Room, room).x),
            )
            result = pipeline.run(candidates)
            trace = result.kinds.get("controller")
            if trace is not None and trace.survivor is not None:
                return cast(Room, trace.survivor), True
        all_rooms = list(self.space.rooms.values())
        if not all_rooms:
            raise RuntimeError("control-policy pack has no rooms to select from")
        return min(all_rooms, key=lambda room: self.evaluate_gains(room.x)), False

    def run(self, steps: int) -> ControlPolicyPackResult:
        """Execute the configured QESSpace for ``steps`` ticks and summarize the best policy."""

        step_count = _validate_positive_int("steps", steps)
        started = time.perf_counter()
        self.space.run(step_count)
        wall_time = time.perf_counter() - started
        best_room, admitted = self._best_room()
        final_state, objective = self.simulate_policy(best_room.x)
        active = self.space.active_rooms()
        normalized_weights = _normalized_weights([room.weight for room in active])
        return ControlPolicyPackResult(
            best_gains=best_room.x.copy(),
            objective=objective,
            final_state=final_state,
            final_distance_to_target=float(np.linalg.norm(final_state - self._target_state)),
            admitted=admitted,
            iterations=step_count,
            active_rooms=len(active),
            collapsed_rooms=len(self.space.collapsed_rooms()),
            convergence=(
                convergence_coefficient(normalized_weights.tolist()) if normalized_weights.size else 0.0
            ),
            wall_time_seconds=wall_time,
        )


@dataclass(frozen=True)
class DesignSpacePackConfig:
    """Configuration for a classical bounded design-parameter search.

    Attributes:
        parameter_lower: Lower search bounds.
        parameter_upper: Upper search bounds.
        cost_fn: Objective to minimize.
        constraint_fn: Constraint function where values ``<= 0`` are feasible.
        initial_design: Optional seed design; defaults to the midpoint of the bounds.
        penalty_scale: Constraint-violation penalty multiplier.
        branch_count: Number of initial Latin-hypercube candidate designs.
        mutation_scale: Gaussian local-search step scale.
        permission_theta: Genesis/adaptive permission threshold.
        adaptive_permission: Whether to use :class:`AdaptivePermission`.
        space_dt: QESSpace step size.
        rng_seed: Optional deterministic random seed.
        max_workers: Optional QESSpace worker count.
    """

    parameter_lower: Sequence[float] | npt.ArrayLike
    parameter_upper: Sequence[float] | npt.ArrayLike
    cost_fn: CostFn
    constraint_fn: ConstraintFn
    initial_design: Sequence[float] | npt.ArrayLike | None = None
    penalty_scale: float = 1.0e6
    branch_count: int = 50
    mutation_scale: float = 0.004
    permission_theta: float = 1.5
    adaptive_permission: bool = False
    space_dt: float = 1.0
    rng_seed: int | None = None
    max_workers: int | None = None


@dataclass(frozen=True)
class DesignSpacePackResult:
    """Summary of one classical design-space search run."""

    best_design: FloatArray
    objective: float
    cost: float
    constraint_value: float
    feasible: bool
    admitted: bool
    iterations: int
    active_rooms: int
    collapsed_rooms: int
    convergence: float
    wall_time_seconds: float


class DesignSpacePack:
    """Prebuilt QESSpace wrapper for classical design-parameter search."""

    def __init__(self, config: DesignSpacePackConfig):
        self.config = config
        if not callable(config.cost_fn):
            raise ValueError("cost_fn must be callable")
        if not callable(config.constraint_fn):
            raise ValueError("constraint_fn must be callable")
        lower, upper = _validate_bounds(
            "parameter_lower",
            config.parameter_lower,
            "parameter_upper",
            config.parameter_upper,
        )
        self._space_bounds = MCCStateSpace(
            lower=lower,
            upper=upper,
            reference=(lower + upper) / 2.0,
            names=[f"p_{index}" for index in range(lower.shape[0])],
        )
        if config.initial_design is None:
            self._initial_design = self._space_bounds.reference.copy()
        else:
            self._initial_design = _as_1d_array(
                "initial_design",
                config.initial_design,
                dim=self._space_bounds.dim,
            )
        if np.any(self._initial_design < lower) or np.any(self._initial_design > upper):
            raise ValueError("initial_design must lie within [parameter_lower, parameter_upper]")
        self._penalty_scale = _validate_non_negative_float("penalty_scale", config.penalty_scale)
        self._branch_count = _validate_positive_int("branch_count", config.branch_count)
        self._mutation_scale = _validate_non_negative_float("mutation_scale", config.mutation_scale)
        permission_theta = _validate_positive_float("permission_theta", config.permission_theta)
        self._space_dt = _validate_positive_float("space_dt", config.space_dt)
        if config.max_workers is not None and config.max_workers < 1:
            raise ValueError("max_workers must be >= 1 when provided")
        self._rng = np.random.default_rng(config.rng_seed)
        self._generator = RealityGenerator(rng=self._rng)
        self._noise = RoomDynamics()
        self.space = QESSpace(
            permission_gate=(
                AdaptivePermission(theta=permission_theta)
                if config.adaptive_permission
                else GenesisPermission(theta=permission_theta)
            ),
            step_fn=self._step_room,
            dt=self._space_dt,
            max_workers=config.max_workers,
        )
        self._seed_room = self._make_seed_room()
        self._spawn_initial_population()

    def _make_seed_room(self) -> Room:
        design = self._initial_design.copy()
        return Room(
            x=design,
            x_star=design.copy(),
            lower=self._space_bounds.lower.copy(),
            upper=self._space_bounds.upper.copy(),
            activation=np.ones(self._space_bounds.dim, dtype=float),
        )

    def _spawn_initial_population(self) -> None:
        self.space.add_room(self._seed_room)
        children = self._generator.branch_latin_hypercube(self._seed_room, count=self._branch_count)
        self.space.spawn(children)

    def constraint_value(self, design: Sequence[float] | npt.ArrayLike) -> float:
        """Return the configured feasibility metric where ``<= 0`` means valid."""

        design_array = _as_1d_array("design", design, dim=self._space_bounds.dim)
        value = float(self.config.constraint_fn(design_array))
        if not np.isfinite(value):
            raise ValueError("constraint_fn must return a finite float")
        return value

    def cost(self, design: Sequence[float] | npt.ArrayLike) -> float:
        """Return the configured design cost."""

        design_array = _as_1d_array("design", design, dim=self._space_bounds.dim)
        value = float(self.config.cost_fn(design_array))
        if not np.isfinite(value):
            raise ValueError("cost_fn must return a finite float")
        return value

    def objective(self, design: Sequence[float] | npt.ArrayLike) -> float:
        """Return the penalized search objective for one design vector."""

        design_array = _as_1d_array("design", design, dim=self._space_bounds.dim)
        constraint = self.constraint_value(design_array)
        penalty = self._penalty_scale * max(0.0, constraint)
        return self.cost(design_array) + penalty

    def _step_room(self, room: Room, t: float, dt: float) -> FloatArray:
        current_objective = self.objective(room.x)
        proposal = room.x + self._noise.stochastic_noise(room.dim, sigma=self._mutation_scale, rng=self._rng)
        proposal = self._space_bounds.clip(proposal)
        proposal_objective = self.objective(proposal)
        return proposal if proposal_objective < current_objective else cast(FloatArray, room.x.copy())

    def _best_room(self) -> tuple[Room, bool]:
        rooms = list(self.space.rooms.values())
        if not rooms:
            raise RuntimeError("design-space pack has no rooms to select from")
        pipeline = GenesisSelectionPipeline(
            gates=[
                lambda room: max(0.0, self.constraint_value(cast(Room, room).x)),
                lambda room: 0.0 if cast(Room, room).state != "Collapsed" else 1.0,
            ],
            signature_fn=lambda room: "design",
            score_fn=lambda room: self.objective(cast(Room, room).x),
        )
        result = pipeline.run(rooms)
        trace = result.kinds.get("design")
        if trace is not None and trace.survivor is not None:
            return cast(Room, trace.survivor), True
        return min(rooms, key=lambda room: self.objective(room.x)), False

    def run(self, steps: int) -> DesignSpacePackResult:
        """Execute the configured QESSpace for ``steps`` ticks and summarize the best design."""

        step_count = _validate_positive_int("steps", steps)
        started = time.perf_counter()
        self.space.run(step_count)
        wall_time = time.perf_counter() - started
        best_room, admitted = self._best_room()
        constraint_value = self.constraint_value(best_room.x)
        active = self.space.active_rooms()
        normalized_weights = _normalized_weights([room.weight for room in active])
        return DesignSpacePackResult(
            best_design=best_room.x.copy(),
            objective=self.objective(best_room.x),
            cost=self.cost(best_room.x),
            constraint_value=constraint_value,
            feasible=constraint_value <= 0.0,
            admitted=admitted,
            iterations=step_count,
            active_rooms=len(active),
            collapsed_rooms=len(self.space.collapsed_rooms()),
            convergence=(
                convergence_coefficient(normalized_weights.tolist()) if normalized_weights.size else 0.0
            ),
            wall_time_seconds=wall_time,
        )


@dataclass(frozen=True)
class ResourceJob:
    """One job for classical budget allocation.

    Attributes:
        name: Stable job identifier.
        demand: Requested compute budget.
        priority: Relative value/importance weight.
        permission: Permission factor in ``[0, 1]`` for the allocator priority.
        uncertainty: Uncertainty factor in ``[0, +inf)``.
        risk: Risk factor in ``[0, +inf)``.
    """

    name: str
    demand: float
    priority: float
    permission: float = 1.0
    uncertainty: float = 1.0
    risk: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("job name must be a non-empty string")
        if not np.isfinite(self.demand) or self.demand < 0.0:
            raise ValueError("job demand must be a finite value >= 0")
        if not np.isfinite(self.priority) or self.priority < 0.0:
            raise ValueError("job priority must be a finite value >= 0")
        if not np.isfinite(self.permission) or not 0.0 <= self.permission <= 1.0:
            raise ValueError("job permission must be in [0, 1]")
        if not np.isfinite(self.uncertainty) or self.uncertainty < 0.0:
            raise ValueError("job uncertainty must be >= 0")
        if not np.isfinite(self.risk) or self.risk < 0.0:
            raise ValueError("job risk must be >= 0")


@dataclass(frozen=True)
class ResourceAllocationPackConfig:
    """Configuration for classical priority-weighted budget allocation."""

    jobs: Sequence[ResourceJob]
    total_budget: float
    min_share: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0


@dataclass(frozen=True)
class ResourceAllocationPackResult:
    """Summary of one resource-allocation run."""

    allocations: dict[str, float]
    priorities: dict[str, float]
    unmet_demand: dict[str, float]
    total_allocated: float
    budget_utilization: float
    weighted_satisfaction: float
    allocation_gini: float
    allocation_convergence: float
    wall_time_seconds: float


class ResourceAllocationPack:
    """Classical priority-weighted compute-budget allocator."""

    def __init__(self, config: ResourceAllocationPackConfig):
        self.config = config
        if not config.jobs:
            raise ValueError("jobs must be non-empty")
        job_names = [job.name for job in config.jobs]
        if len(job_names) != len(set(job_names)):
            raise ValueError("job names must be unique")
        self._jobs = list(config.jobs)
        self._total_budget = _validate_non_negative_float("total_budget", config.total_budget)
        self._min_share = float(config.min_share)
        if not 0.0 <= self._min_share <= 1.0:
            raise ValueError("min_share must be in [0, 1]")
        self._allocator = ComputeAllocator(
            alpha=_validate_positive_float("alpha", config.alpha),
            beta=_validate_positive_float("beta", config.beta),
            gamma=_validate_positive_float("gamma", config.gamma),
            delta=_validate_positive_float("delta", config.delta),
        )

    def _profiles(self) -> list[RoomComputeProfile]:
        return [
            RoomComputeProfile(
                permission=job.permission,
                uncertainty=job.uncertainty,
                risk=job.risk,
                value=job.priority,
            )
            for job in self._jobs
        ]

    def run(self) -> ResourceAllocationPackResult:
        """Compute a fair, demand-capped classical allocation for the configured jobs."""

        started = time.perf_counter()
        profiles = self._profiles()
        demands = np.asarray([job.demand for job in self._jobs], dtype=float)
        priorities = np.asarray([self._allocator.priority(profile) for profile in profiles], dtype=float)
        allocations = np.minimum(
            self._allocator.allocate_with_floor(
                profiles,
                total=self._total_budget,
                min_share=self._min_share,
            ),
            demands,
        )
        remaining_budget = self._total_budget - float(allocations.sum())
        while remaining_budget > 1e-12:
            unmet = demands - allocations
            active_indexes = [index for index, value in enumerate(unmet) if value > 1e-12]
            if not active_indexes:
                break
            extra = self._allocator.allocate(
                [profiles[index] for index in active_indexes],
                total=remaining_budget,
            )
            if extra.size == 0 or float(extra.sum()) <= 0.0:
                break
            before = allocations.copy()
            for offset, index in enumerate(active_indexes):
                allocations[index] = min(demands[index], allocations[index] + float(extra[offset]))
            if np.allclose(before, allocations):
                break
            remaining_budget = self._total_budget - float(allocations.sum())
        wall_time = time.perf_counter() - started
        normalized_allocations = _normalized_weights(allocations.tolist())
        demand_total = float(demands.sum())
        if demand_total > 0.0:
            weighted_satisfaction = float(
                np.sum(priorities * np.minimum(1.0, allocations / np.where(demands > 0.0, demands, 1.0)))
                / np.sum(priorities)
            )
        else:
            weighted_satisfaction = 1.0
        return ResourceAllocationPackResult(
            allocations={job.name: float(allocations[index]) for index, job in enumerate(self._jobs)},
            priorities={job.name: float(priorities[index]) for index, job in enumerate(self._jobs)},
            unmet_demand={
                job.name: float(demands[index] - allocations[index])
                for index, job in enumerate(self._jobs)
            },
            total_allocated=float(allocations.sum()),
            budget_utilization=(
                float(allocations.sum()) / self._total_budget if self._total_budget > 0.0 else 0.0
            ),
            weighted_satisfaction=weighted_satisfaction,
            allocation_gini=gini_coefficient(normalized_allocations.tolist()),
            allocation_convergence=(
                convergence_coefficient(normalized_allocations.tolist())
                if normalized_allocations.size
                else 0.0
            ),
            wall_time_seconds=wall_time,
        )


class DomainPackRegistry:
    """Discoverability helper for the built-in classical domain packs."""

    _DESCRIPTORS = (
        DomainPackDescriptor(
            name="control_policy",
            builder="make_control_policy_space",
            summary="Classical controller-gain search over a bounded first-order plant model.",
            returns="ControlPolicyPack",
        ),
        DomainPackDescriptor(
            name="design_space",
            builder="make_design_space",
            summary="Classical bounded design-parameter search with a user-supplied cost and constraint.",
            returns="DesignSpacePack",
        ),
        DomainPackDescriptor(
            name="resource_allocation",
            builder="make_resource_allocation_pack",
            summary="Classical fair, priority-weighted compute-budget allocation across jobs.",
            returns="ResourceAllocationPack",
        ),
    )

    def list(self) -> tuple[DomainPackDescriptor, ...]:
        """Return all available domain-pack descriptors."""

        return self._DESCRIPTORS

    def names(self) -> tuple[str, ...]:
        """Return the registered pack names."""

        return tuple(descriptor.name for descriptor in self._DESCRIPTORS)

    def get(self, name: str) -> DomainPackDescriptor:
        """Return metadata for one named pack."""

        for descriptor in self._DESCRIPTORS:
            if descriptor.name == name:
                return descriptor
        raise KeyError(f"unknown domain pack: {name!r}")


def make_control_policy_space(config: ControlPolicyPackConfig) -> ControlPolicyPack:
    """Build a reusable classical control-policy search pack."""

    return ControlPolicyPack(config)


def make_design_space(config: DesignSpacePackConfig) -> DesignSpacePack:
    """Build a reusable classical design-space search pack."""

    return DesignSpacePack(config)


def make_resource_allocation_pack(config: ResourceAllocationPackConfig) -> ResourceAllocationPack:
    """Build a reusable classical resource-allocation pack."""

    return ResourceAllocationPack(config)


def list_domain_packs() -> tuple[DomainPackDescriptor, ...]:
    """Enumerate the available classical domain packs."""

    return DomainPackRegistry().list()


__all__ = [
    "ControlPolicyPack",
    "ControlPolicyPackConfig",
    "ControlPolicyPackResult",
    "DesignSpacePack",
    "DesignSpacePackConfig",
    "DesignSpacePackResult",
    "DomainPackDescriptor",
    "DomainPackRegistry",
    "ResourceAllocationPack",
    "ResourceAllocationPackConfig",
    "ResourceAllocationPackResult",
    "ResourceJob",
    "list_domain_packs",
    "make_control_policy_space",
    "make_design_space",
    "make_resource_allocation_pack",
]
