"""Adaptive, gradient-informed search intelligence for QES rooms.

The core `QESSpace` engine is deliberately agnostic about *how* a room's
state evolves tick to tick -- callers supply a `step_fn`. Earlier examples
used a naive isotropic random walk as that `step_fn`, which is a valid but
weak search strategy: on smooth objectives it converges far slower than a
purpose-built optimizer (see `examples/scipy_baseline_benchmark.py`).

This module upgrades the framework's default search intelligence with
`AdaptiveGradientSearch`, a governed local/global strategy that combines:

1. **Gradient-informed descent** -- a central-difference gradient estimate
   feeding an Adam-style per-dimension adaptive step (first/second moment
   estimates with bias correction), applied unconditionally each gradient
   tick like standard gradient descent. Adam's per-dimension scaling is what
   lets QES make real progress on ill-conditioned, curved objectives like
   the Rosenbrock function, where a naive momentum or greedy-accept step
   oscillates or stalls.
2. **Self-adaptive stochastic search** -- Rechenberg's "1/5 success rule"
   (a classical evolution-strategies result: a step size producing successful
   mutations ~1/5 of the time is close to optimal for many objectives)
   auto-tunes each room's perturbation scale, so the search neither stalls
   (step too small) nor thrashes (step too large) without any hand-tuning.
3. **Stagnation-triggered restart** -- after a configurable number of ticks
   without improvement, a room's step size is boosted and its momentum
   reset, letting it escape local structure instead of freezing there.

Per-room adaptive state (step size, momentum, best-known point, stagnation
counter) is persisted in `Room.memory`, since `QESSpace` mutates the same
`Room` object in place across ticks rather than re-instantiating state --
this is the natural "room memory" extension point the framework already
provides (see `Room.memory`, `Room.tag()`).

`optimize()` wires `AdaptiveGradientSearch` together with `RealityGenerator`
population branching and `QESSpace`'s governed generate/execute/measure/
permit/select/converge loop into a single reusable entry point, so any
domain-agnostic objective (control policies, design parameters, scientific
model fits, resource allocations, or an arbitrary black-box function) can be
optimized without hand-assembling the pipeline each time.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace

ObjectiveFn = Callable[[np.ndarray], float]


@dataclass
class AdaptiveSearchConfig:
    """Tunables for `AdaptiveGradientSearch`; defaults suit most bounded,
    continuous, low-to-moderate-dimensional objectives without adjustment."""

    step_size: float = 0.5
    min_step: float = 1e-6
    max_step: float = 5.0
    success_target: float = 0.2  # Rechenberg's 1/5 rule
    step_growth: float = 1.22
    step_shrink: float = 0.82
    gradient_prob: float = 0.85
    gradient_eps: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    learning_rate: float = 0.03
    stagnation_patience: int = 25
    success_window: int = 20
    memory_key: str = "adaptive_search"

    def __post_init__(self) -> None:
        """Validate optimizer hyperparameters."""
        for name in (
            "step_size",
            "min_step",
            "max_step",
            "step_growth",
            "step_shrink",
            "gradient_eps",
            "adam_eps",
            "learning_rate",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            setattr(self, name, value)
        for name in ("success_target", "gradient_prob", "adam_beta1", "adam_beta2"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
            setattr(self, name, value)
        if self.adam_beta1 == 1.0 or self.adam_beta2 == 1.0:
            raise ValueError("Adam beta values must be less than 1")
        if self.max_step < self.min_step:
            raise ValueError("max_step must be >= min_step")
        if not self.min_step <= self.step_size <= self.max_step:
            raise ValueError("step_size must lie within [min_step, max_step]")
        if not isinstance(self.stagnation_patience, int) or self.stagnation_patience < 0:
            raise ValueError("stagnation_patience must be an integer >= 0")
        if not isinstance(self.success_window, int) or self.success_window <= 0:
            raise ValueError("success_window must be an integer > 0")
        if not isinstance(self.memory_key, str) or not self.memory_key.strip():
            raise ValueError("memory_key must be a non-empty string")


@dataclass
class _RoomSearchState:
    step_size: float
    adam_m: np.ndarray
    adam_v: np.ndarray
    adam_t: int
    best_x: np.ndarray
    best_value: float
    stagnation: int = 0
    recent_successes: deque[bool] = field(default_factory=deque)


class AdaptiveGradientSearch:
    """Governed, self-adapting local/global search usable as a `QESSpace` `step_fn`.

    Construct with the objective to minimize and call the instance directly
    wherever a `step_fn(room, t, dt) -> new_x` is expected::

        search = AdaptiveGradientSearch(objective=my_loss)
        space = QESSpace(permission_gate=..., step_fn=search, dt=1.0)
    """

    def __init__(
        self,
        objective: ObjectiveFn,
        config: AdaptiveSearchConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize the adaptive search strategy."""
        if not callable(objective):
            raise TypeError("objective must be callable")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator when provided")
        self.objective = objective
        self.config = config or AdaptiveSearchConfig()
        self.rng = rng or np.random.default_rng()

    def _state_for(self, room: Room) -> _RoomSearchState:
        """Return per-room optimizer state, creating it on first use."""
        cfg = self.config
        stored = room.memory.get(cfg.memory_key)
        if stored is None:
            objective_value = self._evaluate_objective(room.x)
            stored = _RoomSearchState(
                step_size=cfg.step_size,
                adam_m=np.zeros(room.dim),
                adam_v=np.zeros(room.dim),
                adam_t=0,
                best_x=room.x.copy(),
                best_value=objective_value,
                recent_successes=deque(maxlen=cfg.success_window),
            )
            room.memory[cfg.memory_key] = stored
        elif not isinstance(stored, _RoomSearchState):
            raise TypeError(
                f"room.memory[{cfg.memory_key!r}] must contain a _RoomSearchState instance"
            )
        return stored

    def _validate_room(self, room: Room) -> None:
        """Validate the room state expected by the search operator."""
        if not isinstance(room, Room):
            raise TypeError("room must be a Room")
        arrays = {
            "room.x": room.x,
            "room.lower": room.lower,
            "room.upper": room.upper,
        }
        for name, value in arrays.items():
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if value.shape != room.x.shape:
                raise ValueError(f"{name} shape {value.shape} does not match room.x shape {room.x.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
        if np.any(room.lower > room.upper):
            raise ValueError("room.lower must be <= room.upper elementwise")
        if room.activation.ndim != 1 or room.activation.shape != room.x.shape:
            raise ValueError("room.activation must be one-dimensional and match room.x")
        if not np.all(np.isfinite(room.activation)):
            raise ValueError("room.activation must contain only finite values")

    def _evaluate_objective(self, x: np.ndarray) -> float:
        """Evaluate the objective and ensure it returns a finite scalar."""
        value = float(self.objective(x))
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    def _estimate_gradient(
        self, x: np.ndarray, eps: float, lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> np.ndarray:
        """Central-difference gradient estimate: O(eps^2) accurate, 2*dim evals."""
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and > 0")
        if x.ndim != 1:
            raise ValueError("x must be one-dimensional")
        if x.size == 0:
            return np.zeros_like(x)
        offsets = np.eye(x.shape[0], dtype=float) * eps
        grad = np.empty_like(x)
        for index, offset in enumerate(offsets):
            plus = x + offset
            minus = x - offset
            if lower is not None and upper is not None:
                plus = np.clip(plus, lower, upper)
                minus = np.clip(minus, lower, upper)
            distance = plus[index] - minus[index]
            if distance == 0.0:
                grad[index] = 0.0
                continue
            forward = self._evaluate_objective(plus)
            backward = self._evaluate_objective(minus)
            grad[index] = (forward - backward) / distance
        return grad

    def _adam_step(self, state: _RoomSearchState, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Per-dimension adaptive step (Adam). Handles ill-conditioned curvature far
        better than plain momentum: each coordinate gets its own effective learning
        rate scaled by the running RMS of its own gradient history."""
        if grad.shape != x.shape:
            raise ValueError(f"grad shape {grad.shape} does not match x shape {x.shape}")
        if not np.all(np.isfinite(grad)):
            raise ValueError("grad must contain only finite values")
        cfg = self.config
        state.adam_t += 1
        state.adam_m = cfg.adam_beta1 * state.adam_m + (1 - cfg.adam_beta1) * grad
        state.adam_v = cfg.adam_beta2 * state.adam_v + (1 - cfg.adam_beta2) * (grad**2)
        m_hat = state.adam_m / (1 - cfg.adam_beta1**state.adam_t)
        v_hat = state.adam_v / (1 - cfg.adam_beta2**state.adam_t)
        return x - cfg.learning_rate * m_hat / (np.sqrt(v_hat) + cfg.adam_eps)

    def __call__(self, room: Room, t: float, dt: float) -> np.ndarray:
        """Advance one room by one search tick and return the candidate state."""
        del t
        if not np.isfinite(dt):
            raise ValueError("dt must be finite")
        self._validate_room(room)
        cfg = self.config
        state = self._state_for(room)
        current_value = self._evaluate_objective(room.x)

        if current_value < state.best_value:
            state.best_value = current_value
            state.best_x = room.x.copy()
            state.stagnation = 0
        else:
            state.stagnation += 1

        if state.stagnation > cfg.stagnation_patience:
            state.step_size = min(state.step_size * 2.0, cfg.max_step)
            state.adam_m = np.zeros(room.dim)
            state.adam_v = np.zeros(room.dim)
            state.adam_t = 0
            state.stagnation = 0
            room.x = state.best_x.copy()
            current_value = state.best_value

        use_gradient = self.rng.uniform() < cfg.gradient_prob
        if use_gradient:
            grad = self._estimate_gradient(room.x, cfg.gradient_eps, room.lower, room.upper)
            candidate = self._adam_step(state, room.x, grad)
        else:
            candidate = room.x + self.rng.normal(0.0, state.step_size, size=room.dim)

        candidate = np.clip(candidate, room.lower, room.upper)
        candidate_value = self._evaluate_objective(candidate)
        success = candidate_value < current_value

        if candidate_value < state.best_value:
            state.best_value = candidate_value
            state.best_x = candidate.copy()

        # Only the stochastic branch needs greedy accept/reject: gradient steps
        # (Adam) are taken unconditionally, matching standard gradient-descent
        # semantics where a single step may transiently increase loss on curved
        # terrain (e.g. Rosenbrock's valley) yet still be part of a converging
        # trajectory. Step-size adaptation (Rechenberg 1/5-rule) still tracks
        # the stochastic branch's success rate.
        if not use_gradient:
            state.recent_successes.append(success)
            if len(state.recent_successes) >= 5:
                rate = sum(state.recent_successes) / len(state.recent_successes)
                if rate > cfg.success_target:
                    state.step_size = min(state.step_size * cfg.step_growth, cfg.max_step)
                elif rate < cfg.success_target:
                    state.step_size = max(state.step_size * cfg.step_shrink, cfg.min_step)
            return candidate if success else room.x

        return candidate

    def best_known(self, room: Room) -> tuple[np.ndarray, float]:
        """Return the best `(x, value)` this strategy has observed for `room`."""
        self._validate_room(room)
        state = room.memory.get(self.config.memory_key)
        if state is None:
            x = room.x.copy()
            return x, self._evaluate_objective(x)
        if not isinstance(state, _RoomSearchState):
            raise TypeError(
                f"room.memory[{self.config.memory_key!r}] must contain a _RoomSearchState instance"
            )
        return state.best_x.copy(), state.best_value


class NoFeasibleSolutionError(ValueError):
    """The search did not produce any candidate admitted by its gate."""


@dataclass
class OptimizationResult:
    """Outcome of `optimize()`: the best point found and how the search behaved."""

    best_x: np.ndarray
    best_value: float
    iterations: int
    evaluations: int
    survivors: int


def optimize(
    objective: ObjectiveFn,
    seed: Room,
    *,
    iterations: int = 150,
    population: int = 40,
    branch_scale: float = 0.5,
    permission_theta: float = 10.0,
    config: AdaptiveSearchConfig | None = None,
    rng: np.random.Generator | None = None,
) -> OptimizationResult:
    """Governed, adaptive optimization of `objective` starting from `seed`.

    Wires `RealityGenerator` (population diversification), a permission-gated
    `QESSpace` (governed selection), and `AdaptiveGradientSearch` (per-room
    gradient-informed, self-adapting local search) into a single call so any
    bounded, continuous objective can be optimized without hand-assembling
    the pipeline. This is the framework's default "intelligent search" entry
    point; for finer control, construct the pieces directly.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    if not isinstance(seed, Room):
        raise TypeError("seed must be a Room")
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be an integer >= 0")
    if not isinstance(population, int) or population < 0:
        raise ValueError("population must be an integer >= 0")
    if not np.isfinite(branch_scale) or branch_scale < 0.0:
        raise ValueError("branch_scale must be finite and >= 0")
    if not np.isfinite(permission_theta):
        raise ValueError("permission_theta must be finite")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator when provided")
    rng = rng or np.random.default_rng()
    evaluations = 0

    def counted_objective(x: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        value = float(objective(x))
        if not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return value

    search = AdaptiveGradientSearch(objective=counted_objective, config=config, rng=rng)
    generator = RealityGenerator(rng=rng)
    children = generator.branch(seed, count=population, scale=branch_scale)

    space = QESSpace(
        permission_gate=GenesisPermission(theta=permission_theta),
        step_fn=search,
        dt=1.0,
    )
    for child in children:
        child.x = np.clip(child.x, child.lower, child.upper)
    space.spawn(children)

    iterations_run = 0
    for _ in range(iterations):
        space.step()
        iterations_run += 1
        if not space.active_rooms():
            break

    candidates = []
    for room in space.active_rooms():
        candidate, value = search.best_known(room)
        if space.permission_gate.inspect(
            candidate, room.lower, room.upper, room.gates.get("cci_weights"), room.couplings,
        ).admitted:
            candidates.append((candidate, value))
    if not children and space.permission_gate.inspect(
        seed.x, seed.lower, seed.upper, seed.gates.get("cci_weights"), seed.couplings,
    ).admitted:
        candidates.append((seed.x.copy(), counted_objective(seed.x)))
    if not candidates:
        raise NoFeasibleSolutionError("no admissible solution found")
    best_x, best_value = min(candidates, key=lambda item: item[1])

    return OptimizationResult(
        best_x=best_x,
        best_value=best_value,
        iterations=iterations_run,
        evaluations=evaluations,
        survivors=len(space.active_rooms()),
    )


__all__ = [
    "AdaptiveSearchConfig",
    "AdaptiveGradientSearch",
    "OptimizationResult",
    "NoFeasibleSolutionError",
    "optimize",
]
