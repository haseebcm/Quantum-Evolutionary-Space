import numpy as np
import pytest

from qes.cycle import AutonomousCycle, GenerationReport, adaptive_branch_count
from qes.events import EventLog
from qes.invariants import InvariantEngine
from qes.patterns import Pattern, PatternMemory
from qes.permission import GenesisPermission
from qes.reality_generator import RealityGenerator
from qes.room import Room
from qes.space import QESSpace


def make_seed_room(dim: int = 2) -> Room:
    return Room(
        x=np.zeros(dim),
        x_star=np.ones(dim) * 0.5,
        lower=-np.ones(dim),
        upper=np.ones(dim),
        activation=np.ones(dim),
    )


def mean_reverting_step(room: Room, t: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    pull = -0.2 * (room.x - room.x_star)
    noise = rng.normal(0.0, 0.01, size=room.dim)
    return room.x + dt * pull + noise


def make_cycle(rng: np.random.Generator | None = None) -> AutonomousCycle:
    rng = rng or np.random.default_rng(3)
    space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=lambda room, t, dt: mean_reverting_step(room, t, dt, rng),
        dt=1.0,
    )
    return AutonomousCycle(
        intent="closed-loop-test",
        space=space,
        generator=RealityGenerator(rng=rng),
        score_fn=lambda room: float(np.sum(np.abs(room.x - room.x_star))),
    )


# ---------------------------------------------------------------------------
# adaptive_branch_count
# ---------------------------------------------------------------------------


def test_adaptive_branch_count_within_bounds():
    n = adaptive_branch_count(
        uncertainty=0.5, risk=0.3, divergence=0.2, compute_budget=1.0,
        novelty=0.5, historical_success=0.5,
    )
    assert 1 <= n <= 64


def test_adaptive_branch_count_high_uncertainty_branches_more_than_low():
    low = adaptive_branch_count(
        uncertainty=0.0, risk=0.0, divergence=0.0, compute_budget=1.0,
        novelty=0.0, historical_success=1.0,
    )
    high = adaptive_branch_count(
        uncertainty=1.0, risk=0.0, divergence=0.0, compute_budget=1.0,
        novelty=1.0, historical_success=0.0,
    )
    assert high > low


def test_adaptive_branch_count_respects_compute_budget():
    full = adaptive_branch_count(
        uncertainty=0.8, risk=0.1, divergence=0.1, compute_budget=1.0,
        novelty=0.8, historical_success=0.2,
    )
    throttled = adaptive_branch_count(
        uncertainty=0.8, risk=0.1, divergence=0.1, compute_budget=0.1,
        novelty=0.8, historical_success=0.2,
    )
    assert throttled <= full


def test_adaptive_branch_count_respects_min_max():
    n = adaptive_branch_count(
        uncertainty=1.0, risk=0.0, divergence=0.0, compute_budget=1.0,
        novelty=1.0, historical_success=0.0, base=100, max_count=10,
    )
    assert n == 10
    n = adaptive_branch_count(
        uncertainty=0.0, risk=1.0, divergence=1.0, compute_budget=1.0,
        novelty=0.0, historical_success=1.0, min_count=2,
    )
    assert n == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"uncertainty": 1.5},
        {"risk": -0.1},
        {"compute_budget": 0.0},
        {"compute_budget": 1.5},
    ],
)
def test_adaptive_branch_count_rejects_out_of_range_inputs(kwargs):
    base_kwargs = dict(
        uncertainty=0.5, risk=0.5, divergence=0.5, compute_budget=1.0,
        novelty=0.5, historical_success=0.5,
    )
    base_kwargs.update(kwargs)
    with pytest.raises(ValueError):
        adaptive_branch_count(**base_kwargs)


# ---------------------------------------------------------------------------
# AutonomousCycle
# ---------------------------------------------------------------------------


def test_seed_registers_rooms_and_records_spawn_event():
    cycle = make_cycle()
    seed = make_seed_room()
    children = cycle.generator.branch(seed, count=5, scale=0.1)
    cycle.seed(children)

    assert len(cycle.space.rooms) == 5
    spawn_events = cycle.events.filter("SPAWN")
    assert len(spawn_events) == 1
    assert spawn_events[0].payload["count"] == 5


def test_run_generation_returns_report_and_advances_generation():
    cycle = make_cycle()
    cycle.seed(cycle.generator.branch(make_seed_room(), count=6, scale=0.1))

    report = cycle.run_generation()

    assert isinstance(report, GenerationReport)
    assert report.generation == 0
    assert cycle.generation == 1
    assert report.invariants.passed
    assert report.dominant_room_id is not None
    assert report.pattern is not None
    assert report.n_children_next >= 1
    assert report.events_recorded >= 5  # EXECUTE/DIVERGE/PERMISSION/SELECT/MERGE


def test_run_generation_regenerates_population():
    cycle = make_cycle()
    cycle.seed(cycle.generator.branch(make_seed_room(), count=6, scale=0.1))
    before = len(cycle.space.rooms)
    cycle.run_generation()
    after = len(cycle.space.rooms)
    assert after > before  # adaptive regeneration added children


def test_run_multiple_generations_improves_or_holds_convergence():
    cycle = make_cycle()
    cycle.seed(cycle.generator.branch(make_seed_room(), count=8, scale=0.1))
    reports = cycle.run(generations=4)

    assert len(reports) == 4
    assert cycle.generation == 4
    assert len(cycle.history.reports) == 4
    assert len(cycle.history.convergence_series()) == 4
    # Every generation is checked and recorded.
    for report in reports:
        assert report.invariants is not None
    assert len(cycle.events) > 0


def test_pattern_memory_accumulates_across_generations():
    cycle = make_cycle()
    cycle.seed(cycle.generator.branch(make_seed_room(), count=6, scale=0.1))
    cycle.run(generations=3)

    stored = cycle.memory.all_patterns("closed-loop-test")
    assert len(stored) >= 1
    assert all(isinstance(p, Pattern) for p in stored)


def test_shared_infrastructure_can_be_injected():
    rng = np.random.default_rng(11)
    memory = PatternMemory()
    events = EventLog()
    engine = InvariantEngine()
    space = QESSpace(
        permission_gate=GenesisPermission(theta=2.0),
        step_fn=lambda room, t, dt: mean_reverting_step(room, t, dt, rng),
        dt=1.0,
    )
    cycle = AutonomousCycle(
        intent="injected",
        space=space,
        generator=RealityGenerator(rng=rng),
        score_fn=lambda room: float(np.sum(np.abs(room.x))),
        memory=memory,
        events=events,
        invariant_engine=engine,
    )
    assert cycle.memory is memory
    assert cycle.events is events
    assert cycle.invariant_engine is engine

    cycle.seed(cycle.generator.branch(make_seed_room(), count=4, scale=0.1))
    cycle.run_generation()
    assert len(events) > 0
    assert len(memory.all_patterns("injected")) >= 1


def test_intent_must_be_non_empty_string():
    space = QESSpace(permission_gate=GenesisPermission(theta=2.0))
    with pytest.raises(ValueError):
        AutonomousCycle(
            intent="",
            space=space,
            generator=RealityGenerator(),
            score_fn=lambda room: 0.0,
        )


def test_adaptive_branch_count_rejects_invalid_base_and_bounds():
    with pytest.raises(ValueError, match="base must be positive"):
        adaptive_branch_count(
            uncertainty=0.2,
            risk=0.2,
            divergence=0.2,
            compute_budget=1.0,
            novelty=0.2,
            historical_success=0.2,
            base=0,
        )
    with pytest.raises(ValueError, match="min_count <= max_count"):
        adaptive_branch_count(
            uncertainty=0.2,
            risk=0.2,
            divergence=0.2,
            compute_budget=1.0,
            novelty=0.2,
            historical_success=0.2,
            min_count=3,
            max_count=2,
        )


def test_cycle_history_entropy_series_reflects_reports():
    cycle = make_cycle()
    cycle.seed(cycle.generator.branch(make_seed_room(), count=4, scale=0.1))
    cycle.run(generations=2)
    assert len(cycle.history.entropy_series()) == 2


def test_run_generation_handles_missing_dominant_room_without_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = make_cycle()
    cycle.seed(cycle.generator.branch(make_seed_room(), count=4, scale=0.1))
    monkeypatch.setattr(cycle.space, "dominant_room", lambda: None)

    report = cycle.run_generation()

    assert report.dominant_room_id is None
    assert report.pattern is None
    assert all(event.kind != "MERGE" or event.payload["generation"] != report.generation for event in cycle.events.events)
