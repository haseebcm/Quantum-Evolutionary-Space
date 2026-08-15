from __future__ import annotations

import numpy as np
import pytest

import qes.sdk as sdk_module
from qes.permission import AdaptivePermission, GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.sdk import QESClient, SDKPermissionConfig, SDKRunResult, quick_search
from qes.space import QESSpace


def make_bounds(dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    return -np.ones(dim), np.ones(dim)


def make_room(bounds: tuple[np.ndarray, np.ndarray], *, target: np.ndarray, seed: np.ndarray) -> Room:
    lower, upper = bounds
    return Room(
        x=seed.copy(),
        x_star=target.copy(),
        lower=lower.copy(),
        upper=upper.copy(),
        activation=np.ones_like(lower),
    )


def make_step(rng: np.random.Generator):
    def step(room: Room, t: float, dt: float) -> np.ndarray:
        del t
        pull = -0.2 * (room.x - room.x_star)
        noise = rng.normal(0.0, 0.01, size=room.dim)
        return room.x + dt * pull + noise

    return step


def test_client_constructs_default_qes_stack() -> None:
    client = QESClient(
        make_bounds(),
        step_fn=lambda room, t, dt: room.x,
        population=5,
        branch_scale=0.0,
        rng=7,
    )

    assert isinstance(client.permission_gate, GenesisPermission)
    assert len(client.space.active_rooms()) == 5
    assert client.memory.all_patterns("sdk-search") == []


def test_client_supports_objective_mode_and_adaptive_permission() -> None:
    def sphere(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    client = QESClient(
        make_bounds(),
        objective=sphere,
        population=8,
        permission=SDKPermissionConfig(theta=2.0, adaptive=True, window=4),
        rng=11,
        intent="objective-demo",
    )

    result = client.run(steps=6)

    assert isinstance(client.permission_gate, AdaptivePermission)
    assert isinstance(result, SDKRunResult)
    assert result.best_state is not None
    assert result.best_score is not None
    assert result.steps == 6
    assert result.remembered_patterns >= 1
    assert result.final_entropy >= 0.0
    assert result.wall_time_seconds >= 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"step_fn": lambda room, t, dt: room.x, "objective": lambda x: 0.0}, "exactly one"),
        ({"step_fn": lambda room, t, dt: room.x, "population": 0}, "population"),
        ({"step_fn": lambda room, t, dt: room.x, "branch_scale": -0.1}, "branch_scale"),
        ({"step_fn": lambda room, t, dt: room.x, "dt": 0.0}, "dt"),
        ({"step_fn": lambda room, t, dt: room.x, "intent": ""}, "intent"),
        (
            {"step_fn": lambda room, t, dt: room.x, "search_config": object()},
            "search_config",
        ),
    ],
)
def test_client_rejects_invalid_configuration(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        QESClient(make_bounds(), **kwargs)


def test_client_rejects_invalid_bounds_and_seed_state() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        QESClient((np.zeros((2, 2)), np.ones((2, 2))), step_fn=lambda room, t, dt: room.x)

    with pytest.raises(ValueError, match="lower bound must be <= upper bound"):
        QESClient((np.ones(2), -np.ones(2)), step_fn=lambda room, t, dt: room.x)

    with pytest.raises(ValueError, match="seed_state must lie within"):
        QESClient(make_bounds(), step_fn=lambda room, t, dt: room.x, seed_state=np.array([5.0, 0.0]))


def test_run_rejects_negative_steps() -> None:
    client = QESClient(make_bounds(), step_fn=lambda room, t, dt: room.x, rng=0)

    with pytest.raises(ValueError, match="steps"):
        client.run(-1)


def test_run_raises_when_score_function_returns_non_finite_value() -> None:
    client = QESClient(
        make_bounds(),
        step_fn=lambda room, t, dt: room.x,
        score_fn=lambda room: float("nan"),
        rng=0,
    )

    with pytest.raises(ValueError, match="score_fn"):
        client.run(1)


def test_sdk_matches_manual_qes_space_for_same_step_configuration() -> None:
    bounds = make_bounds()
    lower, upper = bounds
    target = np.array([0.25, -0.25])
    seed = np.array([0.0, 0.0])
    population = 6
    branch_scale = 0.15
    steps = 5

    generator_rng_seed = 123
    step_rng_seed = 456

    manual_seed = make_room(bounds, target=target, seed=seed)
    manual_generator = RealityGenerator(rng=np.random.default_rng(generator_rng_seed))
    manual_space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=make_step(np.random.default_rng(step_rng_seed)),
        dt=1.0,
    )
    manual_space.spawn(manual_generator.branch(manual_seed, count=population, scale=branch_scale))
    for _ in range(steps):
        manual_space.step()

    manual_dominant = manual_space.dominant_room()
    assert manual_dominant is not None

    client = QESClient(
        bounds,
        step_fn=make_step(np.random.default_rng(step_rng_seed)),
        target=target,
        seed_state=seed,
        population=population,
        branch_scale=branch_scale,
        permission=SDKPermissionConfig(theta=2.0),
        rng=np.random.default_rng(generator_rng_seed),
        intent="consistency-check",
    )
    result = client.run(steps)

    assert result.best_state is not None
    assert result.best_score is None
    assert result.active_rooms == manual_space.telemetry().active
    assert result.collapsed_rooms == manual_space.telemetry().collapsed
    assert result.total_generated == manual_space.telemetry().total_generated
    assert result.final_entropy == pytest.approx(manual_space.telemetry().entropy)
    assert result.final_convergence == pytest.approx(manual_space.telemetry().convergence)
    assert result.dominant_permission == pytest.approx(manual_space.telemetry().dominant_permission)
    np.testing.assert_allclose(result.best_state, manual_dominant.x)
    assert result.remembered_patterns >= 1


def test_quick_search_runs_objective_search_in_one_call() -> None:
    def sphere(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    result = quick_search(
        make_bounds(3),
        objective=sphere,
        steps=4,
        population=5,
        branch_scale=0.1,
        rng=5,
        intent="quick-search-test",
    )

    assert result.steps == 4
    assert result.best_state is not None
    assert result.best_state.shape == (3,)
    assert result.best_score is not None
    assert result.remembered_patterns >= 1


def test_sdk_helpers_validate_vectors_rng_and_permission_config() -> None:
    with pytest.raises(ValueError, match="expected shape"):
        sdk_module._as_vector("seed", np.zeros(2), shape=(3,))
    with pytest.raises(ValueError, match="finite"):
        sdk_module._as_vector("seed", [1.0, float("inf")])

    assert isinstance(sdk_module._coerce_rng(None), np.random.Generator)
    assert isinstance(sdk_module._coerce_rng(3), np.random.Generator)
    with pytest.raises(TypeError, match="rng"):
        sdk_module._coerce_rng(True)  # type: ignore[arg-type]

    config = SDKPermissionConfig(theta=2.0)
    assert isinstance(config.build(), GenesisPermission)
    adaptive = SDKPermissionConfig(theta=2.0, adaptive=True, theta_max=3.0)
    assert isinstance(adaptive.build(), AdaptivePermission)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"theta": float("inf")}, "theta"),
        ({"theta": 0.0}, "theta"),
        ({"target_rate": 2.0}, "target_rate"),
        ({"adapt_rate": -0.1}, "adapt_rate"),
        ({"window": 0}, "window"),
        ({"theta_max": 0.0}, "theta_max"),
    ],
)
def test_permission_config_rejects_invalid_values(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SDKPermissionConfig(**kwargs)


def test_client_rejects_additional_invalid_configuration_paths() -> None:
    def sphere(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    with pytest.raises(ValueError, match="score_fn"):
        QESClient(make_bounds(), objective=sphere, score_fn=lambda room: 0.0)
    with pytest.raises(ValueError, match="search_config"):
        QESClient(
            make_bounds(),
            step_fn=lambda room, t, dt: room.x,
            search_config=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="max_workers"):
        QESClient(make_bounds(), step_fn=lambda room, t, dt: room.x, max_workers=0)


def test_client_helpers_cover_best_snapshot_and_pattern_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QESClient(make_bounds(), step_fn=lambda room, t, dt: room.x, rng=0)
    assert client.dominant_room() is not None
    assert client.remembered_patterns() == []

    monkeypatch.setattr(client.space, "active_rooms", lambda: [])
    result = client.run(steps=3)
    assert result.steps == 0
    assert result.best_state is None

    fresh_client = QESClient(make_bounds(), step_fn=lambda room, t, dt: room.x, rng=0)
    room_id, best_state, best_score, margin = fresh_client._best_snapshot()
    assert room_id is not None
    assert best_state is not None
    assert best_score is None
    assert margin >= 0.0

    client_with_score = QESClient(
        make_bounds(),
        step_fn=lambda room, t, dt: room.x,
        score_fn=lambda room: 1.25,
        rng=0,
    )
    scored_state, scored_value = client_with_score._candidate_for_room(client_with_score.seed_room)
    assert scored_state.shape == (2,)
    assert scored_value == pytest.approx(1.25)

    client_no_rooms = QESClient(make_bounds(), step_fn=lambda room, t, dt: room.x, rng=0)
    client_no_rooms.space.rooms.clear()
    assert client_no_rooms._best_snapshot() == (None, None, None, 0.0)

    client_fallback = QESClient(make_bounds(), step_fn=lambda room, t, dt: room.x, score_fn=lambda room: 0.0, rng=0)
    monkeypatch.setattr(
        client_fallback,
        "_candidate_for_room",
        lambda room: (room.x.copy(), None),
    )
    monkeypatch.setattr(client_fallback.space, "dominant_room", lambda: None)
    assert client_fallback._best_snapshot() == (None, None, None, 0.0)

    client_record = QESClient(make_bounds(), step_fn=lambda room, t, dt: room.x, rng=0)
    monkeypatch.setattr(client_record, "_best_snapshot", lambda: (None, None, None, 0.0))
    client_record._record_pattern(client_record.space.telemetry())
    assert client_record.remembered_patterns() == []


def test_client_rejects_non_finite_score_function_and_quick_search_step_mode() -> None:
    client = QESClient(
        make_bounds(),
        step_fn=lambda room, t, dt: room.x,
        score_fn=lambda room: float("inf"),
        rng=0,
    )
    with pytest.raises(ValueError, match="score_fn"):
        client._candidate_for_room(client.seed_room)

    result = quick_search(
        make_bounds(),
        step_fn=lambda room, t, dt: room.x,
        steps=1,
        population=2,
        branch_scale=0.0,
        rng=1,
    )
    assert result.steps == 1
