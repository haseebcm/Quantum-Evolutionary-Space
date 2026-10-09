import numpy as np
import pytest

from qes.agent import Agent
from qes.events import EventLog
from qes.intelligence import NoFeasibleSolutionError, optimize
from qes.permission import AdaptivePermission, GenesisPermission
from qes.room import Room
from qes.runtime import PostgresRuntimeStore, RuntimeScheduler
from qes.sdk import QESClient, quick_search
from qes.space import QESSpace


def room(value=0.5):
    return Room(np.array([value]), np.zeros(1), -np.ones(1), np.ones(1), np.ones(1))


@pytest.mark.parametrize("limit", [0, 1, 2, 5, 11])
def test_sdk_hard_evaluation_budget_includes_every_callback(limit):
    calls = []
    client = QESClient(([-1], [1]), objective=lambda x: calls.append(x.copy()) or float(x @ x),
                       population=3, rng=1, max_evaluations=limit)
    result = client.run(10)
    assert len(calls) == result.evaluations == limit
    assert result.stopping_reason == "max_evaluations"
    # Ranking and a subsequent run cannot spend beyond the lifetime cap.
    client.run(10)
    assert len(calls) == limit
    if limit == 0:
        assert result.best_state is None
    else:
        assert result.best_state is not None


def test_cancel_and_zero_deadline_refuse_callbacks():
    effects = []
    client = QESClient(([-1], [1]), step_fn=lambda r, t, dt: effects.append(1) or r.x)
    client.cancel()
    assert client.run(10).stopping_reason == "cancelled"
    result = quick_search(([-1], [1]), objective=lambda x: effects.append(1) or 0,
                          max_wall_time=0)
    assert result.stopping_reason == "max_wall_time"
    assert effects == []


def test_optimizer_budget_returns_only_already_evaluated_results():
    calls = []
    result = optimize(lambda x: calls.append(x.copy()) or float(x @ x), room(),
                      population=3, max_evaluations=1)
    assert result.evaluations == len(calls) == 1
    assert result.stopping_reason == "max_evaluations"
    with pytest.raises(NoFeasibleSolutionError):
        optimize(lambda x: 0.0, room(), max_evaluations=0)


def test_opt_in_selection_prunes_without_collapsing():
    client = QESClient(([-1], [1]), objective=lambda x: float(x @ x), population=4,
                       survivors_per_kind=1, signature_fn=lambda r: "all", rng=2)
    result = client.run(1)
    assert result.active_rooms == 1
    assert result.collapsed_rooms == 0
    assert sum(r.state == "Shadow" for r in client.space.rooms.values()) == 3


def test_callback_failure_does_not_commit_staged_vector_outputs():
    first, second = room(0.2), room(0.3)

    def step(candidate, t, dt):
        if candidate.id == second.id:
            raise RuntimeError("failed callback")
        return np.array([0.8])

    space = QESSpace(GenesisPermission(1), step_fn=step)
    space.spawn([first, second])
    with pytest.raises(RuntimeError):
        space.step()
    np.testing.assert_array_equal(first.x, [0.2])
    assert space.time == 0
    assert space.history == []


@pytest.mark.parametrize("invalid", [np.array([np.nan]), np.array([[0.0]]), np.array([0.0, 0.0])])
def test_step_output_contract(invalid):
    space = QESSpace(GenesisPermission(1), step_fn=lambda r, t, dt: invalid)
    space.spawn([room()])
    with pytest.raises(ValueError):
        space.step()


def test_adaptive_population_admission_is_order_independent():
    samples = [(np.array([x]), -np.ones(1), np.ones(1), None, None) for x in [0, 2, 0.5, -2]]
    a, b = AdaptivePermission(1), AdaptivePermission(1)
    forward = a.evaluate_batch(samples)
    backward = b.evaluate_batch(list(reversed(samples)))
    assert [r.admitted for r in forward] == list(reversed([r.admitted for r in backward]))
    assert a.theta == b.theta
    assert a.admission_rate == b.admission_rate == 0.5


def test_configured_retention_and_backpressure():
    space = QESSpace(GenesisPermission(1), max_rooms=1, max_history=2)
    space.spawn([room()])
    with pytest.raises(OverflowError):
        space.add_room(room())
    space.run(5)
    assert len(space.history) == 2
    sender, receiver = Agent(), Agent(inbox_capacity=1)
    sender.send(receiver, "one")
    with pytest.raises(OverflowError):
        sender.send(receiver, "two")
    assert len(receiver.receive()) == 1
    events = EventLog(max_events=1)
    events.record("SPAWN")
    with pytest.raises(OverflowError):
        events.record("SPAWN")
    scheduler = RuntimeScheduler(max_queue=1)
    scheduler.submit("one", lambda: 1)
    with pytest.raises(OverflowError):
        scheduler.submit("two", lambda: 2)


def test_postgres_rejects_sql_identifier_injection_before_connection():
    with pytest.raises(ValueError, match="SQL identifier"):
        PostgresRuntimeStore(table="jobs; DROP TABLE jobs", connection=object())


def test_batched_permission_matches_scalar_for_mixed_feasibility_and_couplings():
    rng = np.random.default_rng(81)
    gate = GenesisPermission(theta=2, gamma=0.7, m_min=-0.2)
    candidates = []
    for index in range(80):
        state = rng.uniform(-2, 2, size=3)
        lower = -np.ones(3)
        upper = np.ones(3)
        if index % 7 == 0:
            lower[1] = upper[1] = 0
        weights = None if index % 2 else rng.uniform(0, 2, size=3)
        coupling = None if index % 3 else rng.normal(size=(3, 3))
        candidates.append((state, lower, upper) if weights is None and coupling is None
                          else (state, lower, upper, weights, coupling))
    expected = [gate.evaluate(*candidate) for candidate in candidates]
    actual = gate.evaluate_batch(candidates)
    for scalar, batched in zip(expected, actual, strict=True):
        assert batched.admitted == scalar.admitted
        assert batched.hard_permission == scalar.hard_permission
        assert batched.phi == pytest.approx(scalar.phi)
        assert batched.cci == pytest.approx(scalar.cci)
        assert batched.margin == pytest.approx(scalar.margin)
        assert batched.soft_permission == pytest.approx(scalar.soft_permission)


@pytest.mark.parametrize("field,bad", [(0, [np.nan]), (1, [2.0]), (3, [-1.0]),
                                       (4, [[np.inf]])])
def test_batched_permission_rejects_invalid_inputs(field, bad):
    sample = (np.zeros(1), -np.ones(1), np.ones(1), None, None)
    candidates = [sample] * 8
    changed = list(sample)
    changed[field] = np.array(bad)
    candidates[-1] = tuple(changed)
    with pytest.raises(ValueError):
        GenesisPermission(1).evaluate_batch(candidates)
