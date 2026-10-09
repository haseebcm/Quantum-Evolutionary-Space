"""Tests for `qes.intelligence`: the adaptive gradient search upgrade."""
from __future__ import annotations

import re

import numpy as np
import pytest

from qes.intelligence import (
    AdaptiveGradientSearch,
    AdaptiveSearchConfig,
    OptimizationResult,
    optimize,
)
from qes.room import Room


def make_room(x: np.ndarray, dim: int) -> Room:
    return Room(
        x=x,
        x_star=np.zeros(dim),
        lower=-np.ones(dim) * 10,
        upper=np.ones(dim) * 10,
        activation=np.ones(dim),
    )


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def test_step_fn_reduces_objective_over_many_ticks():
    room = make_room(np.array([5.0, -3.0, 4.0]), dim=3)
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(1))
    start_value = sphere(room.x)
    for t in range(120):
        room.x = search(room, t, 1.0)
    assert sphere(room.x) < start_value
    assert sphere(room.x) < 0.1


def test_state_persists_in_room_memory_across_calls():
    room = make_room(np.array([2.0, 2.0]), dim=2)
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(2))
    assert "adaptive_search" not in room.memory
    room.x = search(room, 0, 1.0)
    assert "adaptive_search" in room.memory
    state = room.memory["adaptive_search"]
    # Repeated calls reuse (not recreate) the same state object.
    room.x = search(room, 1, 1.0)
    assert room.memory["adaptive_search"] is state


def test_best_known_tracks_true_best_even_if_last_move_rejected():
    room = make_room(np.array([1.0, 1.0]), dim=2)
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(3))
    for t in range(30):
        room.x = search(room, t, 1.0)
    best_x, best_value = search.best_known(room)
    assert best_value <= sphere(room.x) + 1e-9
    assert best_value >= 0.0


def test_stagnation_triggers_restart_without_error():
    def flat(_x: np.ndarray) -> float:
        return 1.0  # constant objective: every step "stagnates"

    room = make_room(np.array([0.5, 0.5]), dim=2)
    config = AdaptiveSearchConfig(stagnation_patience=3)
    search = AdaptiveGradientSearch(objective=flat, config=config, rng=np.random.default_rng(4))
    for t in range(20):
        room.x = search(room, t, 1.0)
    state = room.memory["adaptive_search"]
    assert state.step_size <= config.max_step


def test_optimize_returns_result_close_to_seed_minimum():
    seed = make_room(np.array([6.0, -6.0, 6.0]), dim=3)
    result = optimize(sphere, seed, iterations=60, population=16, rng=np.random.default_rng(5))
    assert isinstance(result, OptimizationResult)
    assert result.best_value < sphere(seed.x)
    assert result.best_value < 20.0  # well below the seed's value of 108
    assert result.iterations == 60
    assert result.evaluations > 0


def test_optimize_handles_single_dimension():
    seed = make_room(np.array([8.0]), dim=1)
    result = optimize(sphere, seed, iterations=40, population=8, rng=np.random.default_rng(6))
    assert result.best_value < sphere(seed.x)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"step_size": 0.0}, "step_size"),
        ({"success_target": 1.1}, "success_target"),
        ({"min_step": 1.0, "max_step": 0.5}, "max_step"),
        ({"min_step": 0.1, "max_step": 1.0, "step_size": 2.0}, "step_size"),
        ({"stagnation_patience": -1}, "stagnation_patience"),
        ({"success_window": 0}, "success_window"),
        ({"memory_key": "   "}, "memory_key"),
    ],
)
def test_adaptive_search_config_rejects_invalid_values(kwargs: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        AdaptiveSearchConfig(**kwargs)


def test_search_constructor_validates_objective_and_rng():
    with pytest.raises(TypeError):
        AdaptiveGradientSearch(objective=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AdaptiveGradientSearch(objective=sphere, rng=object())  # type: ignore[arg-type]


def test_best_known_without_state_returns_current_point_and_value():
    room = make_room(np.array([2.0, -1.0]), dim=2)
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(7))

    best_x, best_value = search.best_known(room)

    assert np.allclose(best_x, room.x)
    assert best_value == pytest.approx(sphere(room.x))


def test_search_rejects_invalid_room_memory_state():
    room = make_room(np.array([1.0, 1.0]), dim=2)
    room.memory["adaptive_search"] = "bad-state"
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(8))

    with pytest.raises(TypeError, match="_RoomSearchState"):
        search(room, 0, 1.0)
    with pytest.raises(TypeError, match="_RoomSearchState"):
        search.best_known(room)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda room: setattr(room, "lower", np.zeros((1, 2))), "room.lower must be one-dimensional"),
        (
            lambda room: setattr(room, "upper", np.zeros(3)),
            "room.upper shape (3,) does not match room.x shape (2,)",
        ),
        (
            lambda room: setattr(room, "x", np.array([np.nan, 0.0])),
            "room.x must contain only finite values",
        ),
        (
            lambda room: setattr(room, "lower", np.array([11.0, 11.0])),
            "room.lower must be <= room.upper elementwise",
        ),
        (
            lambda room: setattr(room, "activation", np.zeros((1, 2))),
            "room.activation must be one-dimensional and match room.x",
        ),
        (
            lambda room: setattr(room, "activation", np.array([1.0, np.nan])),
            "room.activation must contain only finite values",
        ),
    ],
)
def test_validate_room_rejects_inconsistent_shapes_and_values(mutator, message: str):
    room = make_room(np.array([1.0, 1.0]), dim=2)
    mutator(room)
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(9))

    with pytest.raises((TypeError, ValueError), match=re.escape(message)):
        search._validate_room(room)


def test_validate_room_requires_room_instance():
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(10))

    with pytest.raises(TypeError, match="room must be a Room"):
        search._validate_room(object())  # type: ignore[arg-type]


def test_objective_gradient_and_adam_helpers_validate_inputs():
    room = make_room(np.array([1.0, -1.0]), dim=2)
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(11))
    state = search._state_for(room)

    nonfinite_search = AdaptiveGradientSearch(
        objective=lambda _x: float("inf"),
        rng=np.random.default_rng(12),
    )
    with pytest.raises(ValueError, match="objective must return a finite scalar"):
        nonfinite_search._evaluate_objective(room.x)
    with pytest.raises(ValueError, match="eps must be finite and > 0"):
        search._estimate_gradient(room.x, 0.0)
    with pytest.raises(ValueError, match="x must be one-dimensional"):
        search._estimate_gradient(np.array([[1.0]]), 1e-3)
    assert np.array_equal(search._estimate_gradient(np.array([]), 1e-3), np.array([]))
    with pytest.raises(ValueError, match="grad shape"):
        search._adam_step(state, room.x, np.array([1.0]))
    with pytest.raises(ValueError, match="grad must contain only finite values"):
        search._adam_step(state, room.x, np.array([np.nan, 0.0]))


def test_call_validates_dt_and_updates_best_state_when_room_improves():
    room = make_room(np.array([3.0, 0.0]), dim=2)
    search = AdaptiveGradientSearch(objective=sphere, rng=np.random.default_rng(13))

    with pytest.raises(ValueError, match="dt must be finite"):
        search(room, 0, float("nan"))

    room.x = search(room, 0, 1.0)
    state = room.memory["adaptive_search"]
    room.x = np.array([0.0, 0.0])

    search(room, 1, 1.0)

    assert state.best_value == pytest.approx(0.0)
    assert np.allclose(state.best_x, [0.0, 0.0])
    assert state.stagnation == 0


def test_stochastic_branch_accepts_successes_and_shrinks_after_failures():
    class FakeRng:
        def __init__(self, normal_value: float) -> None:
            self.normal_value = normal_value

        def uniform(self) -> float:
            return 1.0

        def normal(self, mean: float, sigma: float, size: int) -> np.ndarray:
            del mean, sigma
            return np.full(size, self.normal_value)

    success_room = make_room(np.array([5.0]), dim=1)
    success_config = AdaptiveSearchConfig(
        gradient_prob=0.0,
        success_window=5,
        success_target=0.2,
        step_size=0.5,
    )
    success_search = AdaptiveGradientSearch(
        objective=sphere,
        config=success_config,
        rng=np.random.default_rng(14),
    )
    success_search.rng = FakeRng(-0.5)  # type: ignore[assignment]
    for t in range(5):
        success_room.x = success_search(success_room, t, 1.0)
    success_state = success_room.memory["adaptive_search"]
    assert success_state.step_size > success_config.step_size

    failure_room = make_room(np.array([0.0]), dim=1)
    failure_config = AdaptiveSearchConfig(
        gradient_prob=0.0,
        success_window=5,
        success_target=0.2,
        step_size=0.5,
    )
    failure_search = AdaptiveGradientSearch(
        objective=sphere,
        config=failure_config,
        rng=np.random.default_rng(15),
    )
    failure_search.rng = FakeRng(1.0)  # type: ignore[assignment]
    for t in range(5):
        failure_room.x = failure_search(failure_room, t, 1.0)
    failure_state = failure_room.memory["adaptive_search"]
    assert np.allclose(failure_room.x, [0.0])
    assert failure_state.step_size < failure_config.step_size


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"objective": 1, "seed": make_room(np.array([0.0]), dim=1)}, TypeError, "objective"),
        ({"objective": sphere, "seed": object()}, TypeError, "seed"),
        (
            {"objective": sphere, "seed": make_room(np.array([0.0]), dim=1), "iterations": -1},
            ValueError,
            "iterations",
        ),
        (
            {"objective": sphere, "seed": make_room(np.array([0.0]), dim=1), "population": -1},
            ValueError,
            "population",
        ),
        (
            {"objective": sphere, "seed": make_room(np.array([0.0]), dim=1), "branch_scale": -0.1},
            ValueError,
            "branch_scale",
        ),
        (
            {
                "objective": sphere,
                "seed": make_room(np.array([0.0]), dim=1),
                "permission_theta": float("nan"),
            },
            ValueError,
            "permission_theta",
        ),
        (
            {"objective": sphere, "seed": make_room(np.array([0.0]), dim=1), "rng": object()},
            TypeError,
            "rng",
        ),
    ],
)
def test_optimize_rejects_invalid_inputs(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
):
    with pytest.raises(error_type, match=message):
        optimize(**kwargs)  # type: ignore[arg-type]


def test_optimize_breaks_when_all_rooms_collapse_and_falls_back_when_population_is_empty():
    seed = make_room(np.array([5.0]), dim=1)

    with pytest.raises(ValueError, match="no admissible solution"):
        optimize(
            sphere, seed, iterations=5, population=2, permission_theta=0.0,
            rng=np.random.default_rng(16),
        )

    empty = optimize(sphere, seed, iterations=0, population=0, rng=np.random.default_rng(17))
    assert np.allclose(empty.best_x, seed.x)
    assert empty.best_value == pytest.approx(sphere(seed.x))
    assert empty.survivors == 0


def test_optimize_rejects_nonfinite_objective_result():
    seed = make_room(np.array([1.0]), dim=1)

    with pytest.raises(ValueError, match="objective must return a finite scalar"):
        optimize(lambda _x: float("inf"), seed, iterations=0, population=0, rng=np.random.default_rng(18))
