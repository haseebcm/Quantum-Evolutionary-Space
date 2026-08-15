"""Tests for `qes.distributed` (Phase 11 -- Distributed QES)."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np
import pytest

import qes.distributed as distributed_module
from qes.distributed import (
    Controller,
    Worker,
    WorkItem,
    autoscaling_policy,
    distributed_checkpoint,
    restore_from_checkpoint,
)
from qes.room import Room


def make_room(seed: float = 0.0, *, label: str = "room") -> Room:
    state = np.asarray([seed, seed + 0.25, seed + 0.5], dtype=float)
    room = Room(
        x=state.copy(),
        x_star=np.ones_like(state) * 0.5,
        lower=-np.ones_like(state),
        upper=np.ones_like(state),
        activation=np.ones_like(state),
        compute={"compute": 1.0},
    )
    room.tag("label", label)
    return room


def wait_until(predicate, *, timeout: float = 1.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class ClusterHarness:
    def __init__(self, **controller_kwargs: Any) -> None:
        self.controller = Controller(**controller_kwargs)
        self.controller.start()
        self.workers: list[Worker] = []

    def add_worker(
        self,
        worker_id: str,
        process_fn,
        *,
        capacity: int = 1,
        has_gpu: bool = False,
        heartbeat_interval: float = 0.02,
        poll_interval: float = 0.005,
    ) -> Worker:
        worker = Worker(
            worker_id,
            self.controller,
            process_fn,
            capacity=capacity,
            has_gpu=has_gpu,
            heartbeat_interval=heartbeat_interval,
            poll_interval=poll_interval,
        )
        worker.start()
        self.workers.append(worker)
        return worker

    def close(self) -> None:
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.controller.stop()
        self.controller.join(timeout=1.0)


@pytest.fixture
def cluster_factory():
    harnesses: list[ClusterHarness] = []

    def factory(**controller_kwargs: Any) -> ClusterHarness:
        harness = ClusterHarness(**controller_kwargs)
        harnesses.append(harness)
        return harness

    yield factory

    for harness in reversed(harnesses):
        harness.close()


class TestWorkItem:
    def test_roundtrip_with_room_payload(self) -> None:
        item = WorkItem(item_id="WORK-1", room=make_room(0.1), requires_gpu=True)
        restored = WorkItem.from_dict(item.to_dict())
        assert restored.item_id == item.item_id
        assert restored.requires_gpu is True
        assert isinstance(restored.room, Room)
        np.testing.assert_allclose(restored.room.x, item.room.x)

    def test_roundtrip_with_generic_payload(self) -> None:
        item = WorkItem(item_id="WORK-2", room={"task": ["alpha", 3]})
        restored = WorkItem.from_dict(item.to_dict())
        assert restored.room == {"task": ["alpha", 3]}

    def test_rejects_unknown_status(self) -> None:
        with pytest.raises(ValueError, match="status must be one of"):
            WorkItem(item_id="WORK-3", room=make_room(), status="mystery")


class TestValidation:
    def test_controller_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError, match="max_in_flight"):
            Controller(max_in_flight=0)
        with pytest.raises(ValueError, match="lease_duration"):
            Controller(lease_duration=0.0)

    def test_worker_rejects_invalid_capacity(self) -> None:
        controller = Controller()
        with pytest.raises(ValueError, match="capacity"):
            Worker("bad", controller, lambda room: room, capacity=0)

    def test_low_level_validation_and_checkpoint_helpers_cover_edge_cases(self) -> None:
        with pytest.raises(TypeError, match="worker_id must be a string"):
            Worker(123, Controller(), lambda room: room)
        with pytest.raises(ValueError, match="worker_id must be non-empty"):
            Worker("", Controller(), lambda room: room)
        with pytest.raises(TypeError, match="process_fn must be callable"):
            Worker("alpha", Controller(), None)
        with pytest.raises(TypeError, match="has_gpu must be a bool"):
            Worker("alpha", Controller(), lambda room: room, has_gpu=1)
        with pytest.raises(TypeError, match="attempts must be an integer"):
            WorkItem(item_id="WORK-X", room={}, attempts=1.5)
        with pytest.raises(ValueError, match="last_error must be non-empty"):
            WorkItem(item_id="WORK-Y", room={}, last_error="")
        with pytest.raises(ValueError, match="submitted_at must be finite"):
            WorkItem(item_id="WORK-Z", room={}, submitted_at=float("nan"))
        with pytest.raises(TypeError, match="event_log must be an EventLog or None"):
            Controller(event_log="bad")
        with pytest.raises(TypeError, match="controller must be a Controller"):
            distributed_checkpoint("bad")
        with pytest.raises(TypeError, match="data must be a dict"):
            restore_from_checkpoint("bad")
        with pytest.raises(TypeError, match="controller must be a Controller"):
            autoscaling_policy("bad")
        with pytest.raises(ValueError, match="target_utilization must be in"):
            autoscaling_policy(Controller(), target_utilization=0.0)

    def test_jsonable_helpers_handle_numpy_and_reject_unsupported_values(self) -> None:
        room = make_room(0.25)
        restored_room = distributed_module._from_jsonable(distributed_module._to_jsonable(room))
        assert isinstance(restored_room, Room)
        np.testing.assert_allclose(restored_room.x, room.x)

        restored_array = distributed_module._from_jsonable(
            distributed_module._to_jsonable(np.array([1.0, 2.0], dtype=float))
        )
        np.testing.assert_allclose(restored_array, [1.0, 2.0])
        assert distributed_module._to_jsonable(np.int64(2)) == 2

        with pytest.raises(ValueError, match="must be finite"):
            distributed_module._to_jsonable(float("inf"))
        with pytest.raises(TypeError, match="unsupported checkpoint value type"):
            distributed_module._to_jsonable({1, 2, 3})


class TestDiscoveryAndHeartbeats:
    def test_register_worker_populates_discovery_snapshot(self, cluster_factory) -> None:
        cluster = cluster_factory()
        cluster.add_worker("alpha", lambda room: {"score": float(np.sum(room.x))})

        assert wait_until(lambda: "alpha" in cluster.controller.worker_snapshot())
        snapshot = cluster.controller.worker_snapshot()["alpha"]
        assert snapshot["capacity"] == 1
        assert snapshot["has_gpu"] is False
        assert snapshot["alive"] is True

    def test_duplicate_worker_id_rejected(self, cluster_factory) -> None:
        cluster = cluster_factory()
        cluster.add_worker("alpha", lambda room: {"score": float(np.sum(room.x))})
        with pytest.raises(ValueError, match="already registered"):
            cluster.add_worker("alpha", lambda room: {"score": float(np.sum(room.x))})

    def test_worker_heartbeats_refresh_controller_state(self, cluster_factory) -> None:
        cluster = cluster_factory(heartbeat_timeout=0.12, dispatch_interval=0.005)
        cluster.add_worker("alpha", lambda room: {"score": float(np.sum(room.x))})

        assert wait_until(lambda: cluster.controller.worker_snapshot()["alpha"]["last_heartbeat"] > 0.0)
        initial = cluster.controller.worker_snapshot()["alpha"]["last_heartbeat"]
        assert wait_until(
            lambda: cluster.controller.worker_snapshot()["alpha"]["last_heartbeat"] > initial,
            timeout=0.2,
        )

    def test_register_and_start_are_idempotent(self, cluster_factory) -> None:
        cluster = cluster_factory()
        worker = Worker("alpha", cluster.controller, lambda room: room)
        worker.register()
        worker.register()
        worker.start()
        worker.start()
        assert wait_until(lambda: "alpha" in cluster.controller.worker_snapshot())
        worker.stop()
        worker.join(timeout=1.0)


class TestSchedulingAndCompletion:
    def test_submit_and_complete_work_across_multiple_workers(self, cluster_factory) -> None:
        cluster = cluster_factory(max_in_flight=3, lease_duration=0.1, heartbeat_timeout=0.1)

        def process(room: Room) -> dict[str, Any]:
            return {"room_id": room.id, "score": round(float(np.sum(room.x)), 3)}

        cluster.add_worker("alpha", process)
        cluster.add_worker("beta", process)
        cluster.add_worker("gamma", process)

        item_ids = [cluster.controller.submit(make_room(float(i), label=f"room-{i}")) for i in range(6)]
        assert cluster.controller.wait_for_all(timeout=1.0)
        results = cluster.controller.results()
        assert set(results) == set(item_ids)
        assert cluster.controller.status_counts()["done"] == 6

    def test_results_method_returns_completed_payloads(self, cluster_factory) -> None:
        cluster = cluster_factory()
        cluster.add_worker("alpha", lambda room: {"worker": "alpha", "label": room.get_tag("label")})

        item_id = cluster.controller.submit(make_room(0.2, label="result-check"))
        assert cluster.controller.wait_for_all(timeout=1.0)
        assert cluster.controller.results()[item_id] == {"worker": "alpha", "label": "result-check"}

    def test_wait_for_all_times_out_when_work_is_blocked(self, cluster_factory) -> None:
        gate = threading.Event()
        cluster = cluster_factory(lease_duration=0.3, heartbeat_timeout=0.3)

        def process(room: Room) -> dict[str, Any]:
            gate.wait(0.3)
            return {"room_id": room.id}

        cluster.add_worker("alpha", process)
        cluster.controller.submit(make_room(0.1))
        try:
            assert cluster.controller.wait_for_all(timeout=0.03) is False
        finally:
            gate.set()
        assert cluster.controller.wait_for_all(timeout=1.0) is True

    def test_wait_for_all_without_timeout_blocks_until_completion(self, cluster_factory) -> None:
        cluster = cluster_factory()
        cluster.add_worker("alpha", lambda room: {"room": room.id})
        cluster.controller.submit(make_room(0.1))
        assert cluster.controller.wait_for_all() is True

    def test_gpu_required_work_prefers_gpu_worker(self, cluster_factory) -> None:
        cluster = cluster_factory()
        cluster.add_worker("cpu-worker", lambda room: {"worker": "cpu", "room": room.id}, has_gpu=False)
        cluster.add_worker("gpu-worker", lambda room: {"worker": "gpu", "room": room.id}, has_gpu=True)

        gpu_item_id = cluster.controller.submit(make_room(0.4), requires_gpu=True)
        cpu_item_id = cluster.controller.submit(make_room(0.5), requires_gpu=False)

        assert cluster.controller.wait_for_all(timeout=1.0)
        assert cluster.controller.results()[gpu_item_id]["worker"] == "gpu"
        assert cluster.controller.results()[cpu_item_id]["worker"] in {"cpu", "gpu"}

    def test_gpu_required_work_falls_back_to_cpu_without_gpu_workers(self, cluster_factory) -> None:
        cluster = cluster_factory()
        cluster.add_worker("cpu-worker", lambda room: {"worker": "cpu", "room": room.id})

        item_id = cluster.controller.submit(make_room(0.4), requires_gpu=True)
        assert cluster.controller.wait_for_all(timeout=1.0)
        assert cluster.controller.results()[item_id]["worker"] == "cpu"


class TestBackpressure:
    def test_backpressure_caps_leased_work(self, cluster_factory) -> None:
        gate = threading.Event()
        cluster = cluster_factory(max_in_flight=1, lease_duration=0.2, heartbeat_timeout=0.2)

        def process(room: Room) -> dict[str, Any]:
            gate.wait(0.3)
            return {"room": room.id}

        cluster.add_worker("alpha", process)
        cluster.add_worker("beta", process)
        for i in range(3):
            cluster.controller.submit(make_room(float(i)))

        assert wait_until(lambda: cluster.controller.status_counts()["leased"] == 1)
        counts = cluster.controller.status_counts()
        assert counts["leased"] == 1
        assert counts["pending"] == 2
        gate.set()
        assert cluster.controller.wait_for_all(timeout=1.0)

    def test_queued_work_is_not_dropped_under_backpressure(self, cluster_factory) -> None:
        cluster = cluster_factory(max_in_flight=1, lease_duration=0.1, heartbeat_timeout=0.1)
        cluster.add_worker("alpha", lambda room: {"worker": "alpha", "room": room.id})
        cluster.add_worker("beta", lambda room: {"worker": "beta", "room": room.id})

        item_ids = [cluster.controller.submit(make_room(float(i))) for i in range(5)]
        assert cluster.controller.wait_for_all(timeout=1.0)
        assert set(cluster.controller.results()) == set(item_ids)
        assert cluster.controller.status_counts() == {"pending": 0, "leased": 0, "done": 5, "failed": 0}


class TestFaultTolerance:
    def test_crashed_worker_is_detected_and_work_reclaimed(self, cluster_factory) -> None:
        started = threading.Event()
        gate = threading.Event()
        cluster = cluster_factory(lease_duration=0.06, heartbeat_timeout=0.06, dispatch_interval=0.005)

        def fragile_process(room: Room) -> dict[str, Any]:
            started.set()
            gate.wait(0.2)
            return {"worker": "fragile", "room": room.id}

        fragile = cluster.add_worker("fragile", fragile_process, heartbeat_interval=0.02)
        item_id = cluster.controller.submit(make_room(0.8))
        assert started.wait(0.5)
        fragile.simulate_crash()
        cluster.add_worker(
            "stable",
            lambda room: {"worker": "stable", "room": room.id},
            heartbeat_interval=0.02,
        )
        gate.set()

        assert cluster.controller.wait_for_all(timeout=1.0)
        assert cluster.controller.results()[item_id]["worker"] == "stable"
        assert wait_until(lambda: cluster.controller.worker_snapshot()["fragile"]["alive"] is False)

    def test_expired_lease_is_reassigned_and_stale_completion_ignored(self, cluster_factory) -> None:
        hold = threading.Event()
        started = threading.Event()
        cluster = cluster_factory(lease_duration=0.05, heartbeat_timeout=0.3, dispatch_interval=0.005)

        def slow_process(room: Room) -> dict[str, Any]:
            started.set()
            hold.wait(0.15)
            return {"worker": "slow", "room": room.id}

        cluster.add_worker("slow", slow_process, heartbeat_interval=0.02)
        item_id = cluster.controller.submit(make_room(0.9))
        assert started.wait(0.5)

        cluster.add_worker("fast", lambda room: {"worker": "fast", "room": room.id}, heartbeat_interval=0.02)
        assert cluster.controller.wait_for_all(timeout=1.0)
        hold.set()

        assert cluster.controller.results()[item_id]["worker"] == "fast"
        assert cluster.controller.status_counts()["done"] == 1

    def test_processing_exception_marks_work_failed(self, cluster_factory) -> None:
        cluster = cluster_factory()

        def process(room: Room) -> dict[str, Any]:
            raise RuntimeError("boom")

        cluster.add_worker("alpha", process)
        item_id = cluster.controller.submit(make_room(1.0))
        assert cluster.controller.wait_for_all(timeout=1.0)
        state = cluster.controller.to_dict()["work_items"][item_id]
        assert state["status"] == "failed"
        assert state["last_error"] == "boom"

    def test_crashed_worker_exception_does_not_emit_failure_message(self, cluster_factory) -> None:
        started = threading.Event()
        cluster = cluster_factory(lease_duration=0.3, heartbeat_timeout=0.3, dispatch_interval=0.005)
        worker_holder: dict[str, Worker] = {}

        def process(room: Room) -> dict[str, Any]:
            started.set()
            worker_holder["worker"]._crash_event.set()
            raise RuntimeError("boom-after-crash")

        worker_holder["worker"] = cluster.add_worker("fragile", process)
        item_id = cluster.controller.submit(make_room(0.6))
        assert started.wait(0.5)
        assert wait_until(lambda: cluster.controller.status_counts()["pending"] == 1)
        state = cluster.controller.to_dict()["work_items"][item_id]
        assert state["status"] == "pending"
        assert state["last_error"] is None

    def test_reclaim_expired_work_method_requeues_leased_item(self, cluster_factory) -> None:
        started = threading.Event()
        gate = threading.Event()
        cluster = cluster_factory(lease_duration=0.3, heartbeat_timeout=0.3, dispatch_interval=0.005)

        def process(room: Room) -> dict[str, Any]:
            started.set()
            gate.wait(0.2)
            return {"room": room.id}

        cluster.add_worker("alpha", process)
        item_id = cluster.controller.submit(make_room(0.3))
        assert started.wait(0.5)

        snapshot = cluster.controller.to_dict()
        snapshot["work_items"][item_id]["leased_until"] = time.monotonic() - 0.01
        restored = restore_from_checkpoint(snapshot)
        reclaimed = restored.reclaim_expired_work()
        gate.set()

        assert reclaimed == [item_id]
        assert restored.status_counts()["pending"] == 1


class TestCheckpointing:
    def test_distributed_checkpoint_roundtrip_preserves_completed_results(self, cluster_factory) -> None:
        cluster = cluster_factory()
        cluster.add_worker("alpha", lambda room: {"score": float(np.sum(room.x))})
        item_id = cluster.controller.submit(make_room(0.4))
        assert cluster.controller.wait_for_all(timeout=1.0)

        checkpoint = distributed_checkpoint(cluster.controller)
        restored = restore_from_checkpoint(checkpoint)
        assert restored.results()[item_id] == cluster.controller.results()[item_id]
        assert restored.status_counts()["done"] == 1

    def test_restore_from_checkpoint_preserves_pending_and_worker_registry(self) -> None:
        controller = Controller(max_in_flight=2, lease_duration=0.2, heartbeat_timeout=0.2)
        worker = Worker("alpha", controller, lambda room: {"room": room.id}, has_gpu=True)
        worker.register()
        controller.submit(make_room(0.1))
        controller.submit({"task": "generic"}, requires_gpu=False)

        restored = restore_from_checkpoint(distributed_checkpoint(controller))
        assert restored.status_counts()["pending"] == 2
        snapshot = restored.worker_snapshot()["alpha"]
        assert snapshot["has_gpu"] is True
        assert snapshot["capacity"] == 1

    def test_controller_from_dict_preserves_generic_payload(self) -> None:
        controller = Controller()
        item_id = controller.submit({"task": ["alpha", 1]}, item_id="WORK-CUSTOM")

        restored = Controller.from_dict(controller.to_dict())
        assert restored.to_dict()["work_items"][item_id]["room"] == {"task": ["alpha", 1]}

    def test_submit_duplicate_id_and_event_logging_roundtrip(self) -> None:
        event_log = distributed_module.EventLog()
        controller = Controller(event_log=event_log)
        controller.submit(make_room(0.1), item_id="WORK-CUSTOM")

        with pytest.raises(ValueError, match="duplicate work item id"):
            controller.submit(make_room(0.2), item_id="WORK-CUSTOM")

        assert event_log.filter("SPAWN")


class TestAutoscaling:
    def test_autoscaling_policy_recommends_scale_out_under_load(self) -> None:
        controller = Controller()
        Worker("alpha", controller, lambda room: room, capacity=1).register()
        for i in range(6):
            controller.submit(make_room(float(i)))

        assert autoscaling_policy(controller, target_utilization=0.75) > 0

    def test_autoscaling_policy_recommends_scale_in_when_idle(self) -> None:
        controller = Controller()
        Worker("alpha", controller, lambda room: room).register()
        Worker("beta", controller, lambda room: room).register()
        Worker("gamma", controller, lambda room: room).register()

        assert autoscaling_policy(controller, target_utilization=0.75) < 0

    def test_autoscaling_policy_returns_zero_when_balanced(self, cluster_factory) -> None:
        hold = threading.Event()
        cluster = cluster_factory(max_in_flight=2, lease_duration=0.3, heartbeat_timeout=0.3)

        def process(room: Room) -> dict[str, Any]:
            hold.wait(0.2)
            return {"room": room.id}

        cluster.add_worker("alpha", process)
        cluster.add_worker("beta", process)
        cluster.controller.submit(make_room(0.1))
        cluster.controller.submit(make_room(0.2))
        assert wait_until(lambda: cluster.controller.status_counts()["leased"] == 2)

        try:
            assert autoscaling_policy(cluster.controller, target_utilization=1.0) == 0
        finally:
            hold.set()
        assert cluster.controller.wait_for_all(timeout=1.0)

    def test_autoscaling_policy_handles_no_workers_and_idle_scale_in(self) -> None:
        empty = Controller()
        assert autoscaling_policy(empty) == 0
        empty.submit(make_room(0.1))
        assert autoscaling_policy(empty) == 1

        controller = Controller()
        Worker("alpha", controller, lambda room: room).register()
        Worker("beta", controller, lambda room: room).register()
        done_item = WorkItem(item_id="DONE", room={}, status="done")
        controller._work_items[done_item.item_id] = done_item
        assert autoscaling_policy(controller, target_utilization=0.75) == -1


class TestShutdown:
    def test_controller_stop_and_worker_join_cleanly(self, cluster_factory) -> None:
        cluster = cluster_factory()
        worker = cluster.add_worker("alpha", lambda room: {"room": room.id})
        cluster.controller.submit(make_room(0.1))
        assert cluster.controller.wait_for_all(timeout=1.0)

        worker.stop()
        worker.join(timeout=1.0)
        cluster.controller.stop()
        cluster.controller.join(timeout=1.0)

        assert worker.is_alive() is False

    def test_direct_controller_internal_branches_are_safe(self) -> None:
        controller = Controller(dispatch_interval=0.001)
        worker = Worker("alpha", controller, lambda room: room)

        with pytest.raises(TypeError, match="worker must be a Worker"):
            controller.register("alpha")

        worker.register()
        controller.start()
        controller.start()

        with controller._condition:
            item_id = controller.submit(make_room(0.1))
            controller._worker_handles.pop("alpha")
            controller._dispatch_locked()
            assert controller._pending_order == [item_id]

            state = controller._workers["alpha"]
            state.stopped = True
            state.alive = False

        controller._handle_message({"type": "heartbeat", "worker_id": "missing", "timestamp": time.monotonic()})
        controller._handle_message({"type": "heartbeat", "worker_id": "alpha", "timestamp": time.monotonic()})
        assert controller.worker_snapshot()["alpha"]["alive"] is False

        controller._handle_message({"type": "completed", "worker_id": "alpha", "item_id": "nope", "lease_id": "L"})
        controller._handle_message(
            {"type": "failed", "worker_id": "alpha", "item_id": "nope", "lease_id": "L", "error": "boom"}
        )
        controller._handle_message({"type": "stopped", "worker_id": "missing", "crashed": True})

        with pytest.raises(ValueError, match="unknown controller message type"):
            controller._handle_message({"type": "mystery"})

        with controller._condition:
            assert controller.reclaim_expired_work() == []
            assert controller._requeue_worker_items_locked("missing", "none") == []
            assert controller._requeue_item_locked(controller._work_items[item_id], "noop") == []

            timeout_item = controller._work_items[item_id]
            timeout_item.status = "leased"
            timeout_item.lease_id = "LEASE-1"
            timeout_item.assigned_worker_id = "alpha"
            timeout_item.leased_until = time.monotonic() + 1.0
            controller._pending_order.append(item_id)
            state = controller._workers["alpha"]
            state.stopped = False
            state.alive = True
            state.last_heartbeat = time.monotonic() - controller.heartbeat_timeout - 1.0

        assert controller.reclaim_expired_work() == [item_id]
        assert controller.status_counts()["pending"] == 1

        with controller._condition:
            timeout_item = controller._work_items[item_id]
            timeout_item.status = "leased"
            timeout_item.lease_id = "LEASE-2"
            timeout_item.assigned_worker_id = "missing-worker"
            timeout_item.leased_until = time.monotonic() + 1.0
            controller._handle_completion(
                {
                    "type": "completed",
                    "worker_id": "missing-worker",
                    "item_id": item_id,
                    "lease_id": "LEASE-2",
                    "result": {"ok": True},
                }
            )
            assert controller.results()[item_id] == {"ok": True}

            timeout_item.status = "leased"
            timeout_item.lease_id = "LEASE-3"
            timeout_item.assigned_worker_id = "missing-worker"
            timeout_item.result = None
            controller._handle_failure(
                {
                    "type": "failed",
                    "worker_id": "missing-worker",
                    "item_id": item_id,
                    "lease_id": "LEASE-3",
                    "error": "bad",
                }
            )
            assert controller.to_dict()["work_items"][item_id]["status"] == "failed"

            controller._workers.pop("alpha")
            timeout_item.status = "leased"
            timeout_item.lease_id = "LEASE-4"
            timeout_item.assigned_worker_id = "alpha"
            timeout_item.result = None
            assert controller._requeue_item_locked(timeout_item, "already-queued") == [item_id]
            assert controller._pending_order.count(item_id) >= 1

        controller.stop()
        controller.join(timeout=1.0)

    def test_stop_and_simulate_crash_swallow_queue_full(self, cluster_factory, monkeypatch) -> None:
        cluster = cluster_factory()
        worker = Worker("alpha", cluster.controller, lambda room: room)
        worker.register()
        worker.start()

        def raise_full(*args, **kwargs):
            raise queue.Full

        monkeypatch.setattr(worker._inbox, "put_nowait", raise_full)
        worker.stop()
        worker.simulate_crash()
        worker.join(timeout=1.0)
