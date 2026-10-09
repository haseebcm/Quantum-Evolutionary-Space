import numpy as np
import pytest

from qes.convergence import convergence_coefficient, qes_entropy
from qes.intelligence import AdaptiveGradientSearch, AdaptiveSearchConfig, optimize
from qes.permission import AdaptivePermission, GenesisPermission
from qes.room import Room
from qes.runtime import RuntimeScheduler, RuntimeStore
from qes.sdk import QESClient, SDKPermissionConfig
from qes.space import QESSpace


def room(x=0.0):
    return Room(np.array([x]), np.zeros(1), -np.ones(1), np.ones(1), np.ones(1))


def test_uniform_viable_rooms_are_not_concentrated():
    space = QESSpace(GenesisPermission(10))
    space.spawn([room(-0.8), room(0.8)])
    space.step()
    assert space.entropy() == pytest.approx(np.log(2))
    assert space.convergence() == pytest.approx(0)
    assert qes_entropy([10, 10]) == pytest.approx(space.entropy())
    assert qes_entropy([1e308, 1e308]) == pytest.approx(space.entropy())
    assert convergence_coefficient([]) == 0
    assert convergence_coefficient([0, 0]) == 0


@pytest.mark.parametrize("weights", [[-1], [np.nan], [np.inf], [[1]]])
def test_entropy_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        qes_entropy(weights)


def test_sdk_never_returns_or_remembers_collapsed_winner():
    client = QESClient(
        ([-1], [1]), step_fn=lambda r, t, dt: np.array([2.0]),
        score_fn=lambda r: -float(r.x[0]), population=2, rng=1,
    )
    result = client.run(1)
    assert result.active_rooms == 0
    assert result.collapsed_rooms == 2
    assert result.status == "no_feasible_solution"
    assert result.best_state is None
    assert result.best_score is None
    assert client.remembered_patterns() == []


def test_result_inspection_does_not_adapt_gate():
    client = QESClient(
        ([-1], [1]), objective=lambda x: float(x @ x), rng=3,
        permission=SDKPermissionConfig(adaptive=True), population=3,
    )
    client.run(1)
    before = (client.permission_gate.theta, list(client.permission_gate._history))
    client.run(0)
    assert (client.permission_gate.theta, client.permission_gate._history) == before


def test_result_revalidates_historical_state_under_current_margin():
    client = QESClient(([-1], [1]), objective=lambda x: float(x[0]), population=1, rng=4)
    candidate = client.space.active_rooms()[0]
    client._search.best_known(candidate)
    client._search._state_for(candidate).best_x = np.array([-1.0])
    candidate.x = np.array([0.0])
    client.space.permission_gate.m_min = 0.1
    assert client.run(0).status == "no_feasible_solution"


def test_restore_updates_cached_canonical_objects_and_gate():
    space = QESSpace(AdaptivePermission(10))
    space.spawn([room()])
    cached = space.active_rooms()[0]
    snapshot = space.snapshot()
    space.step()
    space.restore(snapshot)
    restored = space.active_rooms()[0]
    assert restored is space.rooms[restored.id]
    assert restored is not cached
    assert space.permission_gate.theta == 10
    space.step()
    assert "permission" in restored.memory


def test_clone_isolates_nested_metadata_and_adaptive_strategy():
    parent = room()
    parent.memory["nested"] = {"values": [1]}
    parent.gates["nested"] = {"values": [2]}
    child = parent.clone()
    child.memory["nested"]["values"].append(3)
    child.gates["nested"]["values"].append(4)
    assert parent.memory["nested"] == {"values": [1]}
    assert parent.gates["nested"] == {"values": [2]}
    search = AdaptiveGradientSearch(lambda x: float(x @ x), rng=np.random.default_rng(5))
    space = QESSpace(AdaptivePermission(10), step_fn=search)
    space.spawn([parent])
    clone = space.clone()
    assert clone.step_fn is not search
    clone.step()
    assert space.permission_gate.theta == 10
    assert parent.memory.get("adaptive_search") is None


def test_checkpoint_replays_stateful_search_rng():
    search = AdaptiveGradientSearch(lambda x: float(x @ x), rng=np.random.default_rng(6))
    space = QESSpace(GenesisPermission(10), step_fn=search)
    space.spawn([room(0.7)])
    snapshot = space.snapshot()
    space.run(5)
    expected = space.active_rooms()[0].x.copy()
    space.restore(snapshot)
    space.run(5)
    np.testing.assert_array_equal(space.active_rooms()[0].x, expected)


def test_active_list_and_duplicate_ids_do_not_corrupt_membership():
    space = QESSpace(GenesisPermission(10))
    candidate = room()
    space.add_room(candidate)
    space.active_rooms().clear()
    assert len(space.active_rooms()) == 1
    with pytest.raises(ValueError, match="duplicate"):
        space.add_room(candidate)
    assert space.total_generated == 1


@pytest.mark.parametrize("beta", ["adam_beta1", "adam_beta2"])
def test_adam_rejects_unit_decay(beta):
    with pytest.raises(ValueError, match="less than 1"):
        AdaptiveSearchConfig(**{beta: 1.0})


def test_bounded_objective_never_receives_out_of_domain_probes():
    observed = []

    def objective(x):
        assert np.all(x >= -1) and np.all(x <= 1)
        observed.append(x.copy())
        return float(x @ x)

    result = optimize(objective, room(1), iterations=4, population=3, branch_scale=2,
                      config=AdaptiveSearchConfig(gradient_prob=1), rng=np.random.default_rng(7))
    assert result.best_value <= 1
    assert observed


class FailingStore(RuntimeStore):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def put(self, key, value):
        if self.failures:
            self.failures -= 1
            raise OSError("store unavailable")
        return super().put(key, value)


def test_storage_retry_does_not_repeat_successful_handler():
    effects = []
    store = FailingStore(1)
    scheduler = RuntimeScheduler(store=store)
    job = scheduler.submit("effect", lambda: effects.append(1) or {"ok": True}, retries=1)
    scheduler.run_all()
    assert effects == [1]
    assert job.attempts == 1
    assert store.get(job.job_id)["status"] == "completed"


def test_permanent_storage_failure_retains_result_and_surfaces_error():
    effects = []
    scheduler = RuntimeScheduler(store=FailingStore(3))
    job = scheduler.submit("effect", lambda: effects.append(1) or {"ok": True}, retries=1)
    with pytest.raises(OSError, match="store unavailable"):
        scheduler.run_all()
    assert effects == [1]
    assert job.status == "persist_failed"
    assert job.result == {"ok": True}


def test_scheduler_reports_running_work_inside_handler():
    scheduler = RuntimeScheduler()
    observed = []
    job = scheduler.submit("telemetry", lambda: observed.append(scheduler.telemetry().running))
    scheduler.run_all()
    assert observed == [1]
    assert scheduler.telemetry().running == 0
    assert job.status == "completed"
