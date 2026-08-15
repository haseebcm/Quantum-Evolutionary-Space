"""Phase 11 -- Distributed QES.

This module implements an honest in-process simulation of the roadmap's
distributed-control layer: workers are simulated as threads within a single
process; the coordination logic (heartbeats, work stealing, fault tolerance,
autoscaling policy) is real and would generalize to real distributed workers
behind a real transport, but no real network/multi-machine execution occurs
here.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qes.events import EventLog
from qes.room import Room

WORK_STATUSES = {"pending", "leased", "done", "failed"}


def _require_nonempty_string(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _require_int(name: str, value: int, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        comparator = ">=" if minimum == 0 else ">"
        threshold = minimum if minimum == 0 else minimum - 1
        raise ValueError(f"{name} must be {comparator} {threshold}")
    return value


def _require_float(name: str, value: float, *, minimum: float | None = None) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number <= minimum:
        raise ValueError(f"{name} must be > {minimum}")
    return number


def _optional_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _require_float(name, value)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("checkpoint values must be finite")
        return value
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return {"__qes_type__": "ndarray", "values": value.tolist()}
    if isinstance(value, Room):
        return {
            "__qes_type__": "Room",
            "x": value.x.tolist(),
            "x_star": value.x_star.tolist(),
            "lower": value.lower.tolist(),
            "upper": value.upper.tolist(),
            "activation": value.activation.tolist(),
            "equations": _to_jsonable(list(value.equations)),
            "theta": _to_jsonable(dict(value.theta)),
            "gates": _to_jsonable(dict(value.gates)),
            "couplings": None if value.couplings is None else value.couplings.tolist(),
            "compute": _to_jsonable(dict(value.compute)),
            "memory": _to_jsonable(dict(value.memory)),
            "lineage": _to_jsonable(list(value.lineage)),
            "id": value.id,
            "state": value.state,
            "weight": float(value.weight),
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    raise TypeError(f"unsupported checkpoint value type: {type(value)!r}")


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__qes_type__")
    if marker == "ndarray":
        return np.asarray(value["values"], dtype=float)
    if marker == "Room":
        couplings = value.get("couplings")
        return Room(
            x=np.asarray(value["x"], dtype=float),
            x_star=np.asarray(value["x_star"], dtype=float),
            lower=np.asarray(value["lower"], dtype=float),
            upper=np.asarray(value["upper"], dtype=float),
            activation=np.asarray(value["activation"], dtype=float),
            equations=_from_jsonable(value.get("equations", [])),
            theta=_from_jsonable(value.get("theta", {})),
            gates=_from_jsonable(value.get("gates", {})),
            couplings=None if couplings is None else np.asarray(couplings, dtype=float),
            compute=_from_jsonable(value.get("compute", {})),
            memory=_from_jsonable(value.get("memory", {})),
            lineage=_from_jsonable(value.get("lineage", [])),
            id=value["id"],
            state=value.get("state", "Seed"),
            weight=float(value.get("weight", 1.0)),
        )
    return {key: _from_jsonable(item) for key, item in value.items()}


@dataclass
class WorkItem:
    """One unit of simulated distributed work.

    The payload lives in the `room` field because the QES roadmap centers on
    distributing `Room` objects, but any checkpoint-serializable payload may be
    stored there.
    """

    item_id: str
    room: Any
    requires_gpu: bool = False
    lease_id: str | None = None
    leased_until: float | None = None
    status: str = "pending"
    assigned_worker_id: str | None = None
    attempts: int = 0
    result: Any = None
    last_error: str | None = None
    submitted_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def __post_init__(self) -> None:
        self.item_id = _require_nonempty_string("item_id", self.item_id)
        self.requires_gpu = _require_bool("requires_gpu", self.requires_gpu)
        if self.lease_id is not None:
            self.lease_id = _require_nonempty_string("lease_id", self.lease_id)
        self.leased_until = _optional_float("leased_until", self.leased_until)
        if self.status not in WORK_STATUSES:
            raise ValueError(f"status must be one of {sorted(WORK_STATUSES)}, got {self.status!r}")
        if self.assigned_worker_id is not None:
            self.assigned_worker_id = _require_nonempty_string(
                "assigned_worker_id", self.assigned_worker_id
            )
        self.attempts = _require_int("attempts", self.attempts, minimum=0)
        if self.last_error is not None:
            self.last_error = _require_nonempty_string("last_error", self.last_error)
        self.submitted_at = _require_float("submitted_at", self.submitted_at)
        self.completed_at = _optional_float("completed_at", self.completed_at)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) representation of this work item."""
        return {
            "item_id": self.item_id,
            "room": _to_jsonable(self.room),
            "requires_gpu": self.requires_gpu,
            "lease_id": self.lease_id,
            "leased_until": self.leased_until,
            "status": self.status,
            "assigned_worker_id": self.assigned_worker_id,
            "attempts": self.attempts,
            "result": _to_jsonable(self.result),
            "last_error": self.last_error,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkItem:
        """Reconstruct a `WorkItem` previously serialized via `to_dict()`."""
        return cls(
            item_id=data["item_id"],
            room=_from_jsonable(data.get("room")),
            requires_gpu=bool(data.get("requires_gpu", False)),
            lease_id=data.get("lease_id"),
            leased_until=data.get("leased_until"),
            status=data.get("status", "pending"),
            assigned_worker_id=data.get("assigned_worker_id"),
            attempts=int(data.get("attempts", 0)),
            result=_from_jsonable(data.get("result")),
            last_error=data.get("last_error"),
            submitted_at=float(data.get("submitted_at", time.time())),
            completed_at=data.get("completed_at"),
        )


@dataclass
class _WorkerState:
    worker_id: str
    capacity: int
    has_gpu: bool
    last_heartbeat: float
    discovered_at: float
    alive: bool = True
    assigned_count: int = 0
    completed_count: int = 0
    crashed: bool = False
    stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "capacity": self.capacity,
            "has_gpu": self.has_gpu,
            "last_heartbeat": self.last_heartbeat,
            "discovered_at": self.discovered_at,
            "alive": self.alive,
            "assigned_count": self.assigned_count,
            "completed_count": self.completed_count,
            "crashed": self.crashed,
            "stopped": self.stopped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _WorkerState:
        return cls(
            worker_id=data["worker_id"],
            capacity=int(data["capacity"]),
            has_gpu=bool(data.get("has_gpu", False)),
            last_heartbeat=float(data["last_heartbeat"]),
            discovered_at=float(data["discovered_at"]),
            alive=bool(data.get("alive", True)),
            assigned_count=int(data.get("assigned_count", 0)),
            completed_count=int(data.get("completed_count", 0)),
            crashed=bool(data.get("crashed", False)),
            stopped=bool(data.get("stopped", False)),
        )


class Worker:
    """Simulated distributed worker backed by a Python thread and `queue.Queue`."""

    def __init__(
        self,
        worker_id: str,
        controller: Controller,
        process_fn: Callable[[Any], Any],
        *,
        capacity: int = 1,
        has_gpu: bool = False,
        heartbeat_interval: float = 0.05,
        poll_interval: float = 0.01,
    ) -> None:
        self.worker_id = _require_nonempty_string("worker_id", worker_id)
        if not callable(process_fn):
            raise TypeError("process_fn must be callable")
        self.process_fn = process_fn
        self.capacity = _require_int("capacity", capacity, minimum=1)
        self.has_gpu = _require_bool("has_gpu", has_gpu)
        self.heartbeat_interval = _require_float("heartbeat_interval", heartbeat_interval, minimum=0.0)
        self.poll_interval = _require_float("poll_interval", poll_interval, minimum=0.0)
        self.controller = controller
        self._inbox: queue.Queue[tuple[str, str, Any]] = queue.Queue(maxsize=self.capacity)
        self._stop_event = threading.Event()
        self._crash_event = threading.Event()
        self._registered = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"qes-worker-{self.worker_id}",
            daemon=True,
        )

    def register(self) -> None:
        """Register this worker with the controller for discovery/scheduling."""
        if not self._registered:
            self.controller.register(self)
            self._registered = True

    def start(self) -> None:
        """Register this worker if needed, then start its processing thread."""
        self.register()
        if self._thread.is_alive():
            return
        self._thread.start()

    def stop(self) -> None:
        """Request a graceful stop after any current task completes."""
        self._stop_event.set()
        try:
            self._inbox.put_nowait(("", "", None))
        except queue.Full:
            pass

    def join(self, timeout: float | None = None) -> None:
        """Join the worker thread."""
        self._thread.join(timeout=timeout)

    def simulate_crash(self) -> None:
        """Stop heartbeats and drop any in-flight completion, simulating a crash."""
        self._crash_event.set()
        try:
            self._inbox.put_nowait(("", "", None))
        except queue.Full:
            pass

    def enqueue_lease(self, item_id: str, lease_id: str, room: Any) -> None:
        """Enqueue a leased work item for local processing."""
        self._inbox.put_nowait((item_id, lease_id, room))

    def is_alive(self) -> bool:
        """Return whether the worker thread is still alive."""
        return self._thread.is_alive()

    def _run(self) -> None:
        last_heartbeat = 0.0
        while not self._stop_event.is_set() and not self._crash_event.is_set():
            now = time.monotonic()
            if now - last_heartbeat >= self.heartbeat_interval:
                self.controller.enqueue_message(
                    {"type": "heartbeat", "worker_id": self.worker_id, "timestamp": now}
                )
                last_heartbeat = now
            try:
                item_id, lease_id, room = self._inbox.get(timeout=self.poll_interval)
            except queue.Empty:
                continue

            if not item_id and not lease_id:
                self._inbox.task_done()
                continue

            try:
                result = self.process_fn(room)
            except Exception as exc:
                if not self._crash_event.is_set():
                    self.controller.enqueue_message(
                        {
                            "type": "failed",
                            "worker_id": self.worker_id,
                            "item_id": item_id,
                            "lease_id": lease_id,
                            "error": str(exc),
                        }
                    )
            else:
                if not self._crash_event.is_set():
                    self.controller.enqueue_message(
                        {
                            "type": "completed",
                            "worker_id": self.worker_id,
                            "item_id": item_id,
                            "lease_id": lease_id,
                            "result": result,
                        }
                    )
            finally:
                self._inbox.task_done()

        self.controller.enqueue_message(
            {
                "type": "stopped",
                "worker_id": self.worker_id,
                "timestamp": time.monotonic(),
                "crashed": self._crash_event.is_set(),
            }
        )


class Controller:
    """Central scheduler/aggregator for the Phase 11 thread-based simulation."""

    def __init__(
        self,
        *,
        max_in_flight: int = 4,
        lease_duration: float = 0.2,
        heartbeat_timeout: float = 0.2,
        dispatch_interval: float = 0.01,
        event_log: EventLog | None = None,
    ) -> None:
        self.max_in_flight = _require_int("max_in_flight", max_in_flight, minimum=1)
        self.lease_duration = _require_float("lease_duration", lease_duration, minimum=0.0)
        self.heartbeat_timeout = _require_float("heartbeat_timeout", heartbeat_timeout, minimum=0.0)
        self.dispatch_interval = _require_float("dispatch_interval", dispatch_interval, minimum=0.0)
        if event_log is not None and not isinstance(event_log, EventLog):
            raise TypeError("event_log must be an EventLog or None")
        self.event_log = event_log

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._dispatcher_thread = threading.Thread(
            target=self._run_dispatcher,
            name="qes-distributed-controller",
            daemon=True,
        )
        self._work_items: dict[str, WorkItem] = {}
        self._pending_order: list[str] = []
        self._workers: dict[str, _WorkerState] = {}
        self._worker_handles: dict[str, Worker] = {}
        self._next_item_index = 1
        self._next_lease_index = 1

    def start(self) -> None:
        """Start the controller's dispatch/heartbeat-monitor thread."""
        if self._dispatcher_thread.is_alive():
            return
        self._dispatcher_thread.start()

    def stop(self) -> None:
        """Request a clean stop of the controller dispatcher."""
        self._stop_event.set()
        self.enqueue_message({"type": "controller-stop"})

    def join(self, timeout: float | None = None) -> None:
        """Join the controller dispatcher thread."""
        self._dispatcher_thread.join(timeout=timeout)

    def register(self, worker: Worker) -> None:
        """Register one worker for discovery and future work assignment."""
        if not isinstance(worker, Worker):
            raise TypeError("worker must be a Worker")
        now = time.monotonic()
        with self._condition:
            if worker.worker_id in self._workers:
                raise ValueError(f"worker {worker.worker_id!r} is already registered")
            self._workers[worker.worker_id] = _WorkerState(
                worker_id=worker.worker_id,
                capacity=worker.capacity,
                has_gpu=worker.has_gpu,
                last_heartbeat=now,
                discovered_at=now,
            )
            self._worker_handles[worker.worker_id] = worker
            self._record_event("SPAWN", {"worker_id": worker.worker_id, "has_gpu": worker.has_gpu})
            self._condition.notify_all()

    def enqueue_message(self, message: dict[str, Any]) -> None:
        """Queue an internal worker/controller message for asynchronous handling."""
        self._message_queue.put(message)

    def submit(self, room: Any, *, requires_gpu: bool = False, item_id: str | None = None) -> str:
        """Submit one work item to the distributed population."""
        created_item_id = item_id or f"WORK-{self._next_item_index:06d}"
        item = WorkItem(item_id=created_item_id, room=room, requires_gpu=requires_gpu)
        with self._condition:
            if item.item_id in self._work_items:
                raise ValueError(f"duplicate work item id: {item.item_id!r}")
            self._work_items[item.item_id] = item
            self._pending_order.append(item.item_id)
            if item_id is None:
                self._next_item_index += 1
            self._record_event(
                "SPAWN",
                {"item_id": item.item_id, "requires_gpu": item.requires_gpu},
            )
            self._condition.notify_all()
        return item.item_id

    def wait_for_all(self, timeout: float | None = None) -> bool:
        """Wait until every submitted work item reaches a terminal state."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while not self._all_terminal_locked():
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._condition.wait(timeout=min(remaining, self.dispatch_interval))
                else:
                    self._condition.wait(timeout=self.dispatch_interval)
            return True

    def reclaim_expired_work(self) -> list[str]:
        """Immediately reclaim dead-worker or expired-lease work back to pending."""
        with self._condition:
            reclaimed = self._reclaim_expired_locked(time.monotonic())
            if reclaimed:
                self._condition.notify_all()
            return reclaimed

    def status_counts(self) -> dict[str, int]:
        """Return counts for pending, leased, done, and failed work items."""
        with self._lock:
            counts = {status: 0 for status in WORK_STATUSES}
            for item in self._work_items.values():
                counts[item.status] += 1
            return counts

    def results(self) -> dict[str, Any]:
        """Return aggregated results for completed work items."""
        with self._lock:
            return {
                item_id: item.result
                for item_id, item in self._work_items.items()
                if item.status == "done"
            }

    def worker_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a plain-dict snapshot of the registered worker registry."""
        with self._lock:
            return {worker_id: state.to_dict() for worker_id, state in self._workers.items()}

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) representation of this controller."""
        with self._lock:
            return {
                "config": {
                    "max_in_flight": self.max_in_flight,
                    "lease_duration": self.lease_duration,
                    "heartbeat_timeout": self.heartbeat_timeout,
                    "dispatch_interval": self.dispatch_interval,
                },
                "pending_order": list(self._pending_order),
                "work_items": {item_id: item.to_dict() for item_id, item in self._work_items.items()},
                "workers": {worker_id: state.to_dict() for worker_id, state in self._workers.items()},
                "next_item_index": self._next_item_index,
                "next_lease_index": self._next_lease_index,
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Controller:
        """Reconstruct a `Controller` previously serialized via `to_dict()`."""
        config = data.get("config", {})
        controller = cls(
            max_in_flight=int(config.get("max_in_flight", 4)),
            lease_duration=float(config.get("lease_duration", 0.2)),
            heartbeat_timeout=float(config.get("heartbeat_timeout", 0.2)),
            dispatch_interval=float(config.get("dispatch_interval", 0.01)),
        )
        controller._pending_order = list(data.get("pending_order", []))
        controller._work_items = {
            item_id: WorkItem.from_dict(item_data)
            for item_id, item_data in dict(data.get("work_items", {})).items()
        }
        controller._workers = {
            worker_id: _WorkerState.from_dict(worker_data)
            for worker_id, worker_data in dict(data.get("workers", {})).items()
        }
        controller._next_item_index = int(data.get("next_item_index", len(controller._work_items) + 1))
        controller._next_lease_index = int(data.get("next_lease_index", 1))
        return controller

    def _all_terminal_locked(self) -> bool:
        return all(item.status in {"done", "failed"} for item in self._work_items.values())

    def _record_event(self, kind: str, payload: dict[str, Any]) -> None:
        if self.event_log is not None:
            self.event_log.record(kind, payload=payload, timestamp=time.time())

    def _run_dispatcher(self) -> None:
        while not self._stop_event.is_set():
            self._drain_messages()
            with self._condition:
                self._reclaim_expired_locked(time.monotonic())
                self._dispatch_locked()
                self._condition.notify_all()
            self._stop_event.wait(self.dispatch_interval)
        self._drain_messages()
        with self._condition:
            self._reclaim_expired_locked(time.monotonic())
            self._condition.notify_all()

    def _drain_messages(self) -> None:
        while True:
            try:
                message = self._message_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_message(message)
            finally:
                self._message_queue.task_done()

    def _handle_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "controller-stop":
            return
        if kind == "heartbeat":
            self._handle_heartbeat(message)
            return
        if kind == "completed":
            self._handle_completion(message)
            return
        if kind == "failed":
            self._handle_failure(message)
            return
        if kind == "stopped":
            self._handle_stopped(message)
            return
        raise ValueError(f"unknown controller message type: {kind!r}")

    def _handle_heartbeat(self, message: dict[str, Any]) -> None:
        worker_id = _require_nonempty_string("worker_id", str(message["worker_id"]))
        timestamp = _require_float("timestamp", float(message["timestamp"]))
        with self._condition:
            state = self._workers.get(worker_id)
            if state is None:
                return
            state.last_heartbeat = timestamp
            if not state.stopped:
                state.alive = True
            self._condition.notify_all()

    def _handle_completion(self, message: dict[str, Any]) -> None:
        worker_id = _require_nonempty_string("worker_id", str(message["worker_id"]))
        item_id = _require_nonempty_string("item_id", str(message["item_id"]))
        lease_id = _require_nonempty_string("lease_id", str(message["lease_id"]))
        with self._condition:
            item = self._work_items.get(item_id)
            if item is None or item.status != "leased" or item.lease_id != lease_id:
                return
            item.status = "done"
            item.result = message.get("result")
            item.completed_at = time.time()
            worker = self._workers.get(worker_id)
            if worker is not None:
                worker.assigned_count = max(worker.assigned_count - 1, 0)
                worker.completed_count += 1
            self._record_event("MERGE", {"item_id": item_id, "worker_id": worker_id})
            self._condition.notify_all()

    def _handle_failure(self, message: dict[str, Any]) -> None:
        worker_id = _require_nonempty_string("worker_id", str(message["worker_id"]))
        item_id = _require_nonempty_string("item_id", str(message["item_id"]))
        lease_id = _require_nonempty_string("lease_id", str(message["lease_id"]))
        with self._condition:
            item = self._work_items.get(item_id)
            if item is None or item.status != "leased" or item.lease_id != lease_id:
                return
            item.status = "failed"
            item.last_error = _require_nonempty_string("error", str(message["error"]))
            item.completed_at = time.time()
            worker = self._workers.get(worker_id)
            if worker is not None:
                worker.assigned_count = max(worker.assigned_count - 1, 0)
            self._record_event("COLLAPSE", {"item_id": item_id, "worker_id": worker_id})
            self._condition.notify_all()

    def _handle_stopped(self, message: dict[str, Any]) -> None:
        worker_id = _require_nonempty_string("worker_id", str(message["worker_id"]))
        crashed = bool(message.get("crashed", False))
        with self._condition:
            state = self._workers.get(worker_id)
            if state is None:
                return
            state.alive = False
            state.stopped = True
            state.crashed = crashed
            reclaimed = self._requeue_worker_items_locked(worker_id, "worker-stopped")
            if reclaimed:
                self._condition.notify_all()

    def _dispatch_locked(self) -> None:
        while self._in_flight_count_locked() < self.max_in_flight:
            progressed = False
            for worker_id, state in list(self._workers.items()):
                if not state.alive or state.stopped or state.assigned_count >= state.capacity:
                    continue
                handle = self._worker_handles.get(worker_id)
                if handle is None:
                    continue
                item_id = self._select_pending_for_worker_locked(state)
                if item_id is None:
                    continue
                item = self._work_items[item_id]
                lease_id = f"{worker_id}-LEASE-{self._next_lease_index:06d}"
                self._next_lease_index += 1
                item.status = "leased"
                item.lease_id = lease_id
                item.leased_until = time.monotonic() + self.lease_duration
                item.assigned_worker_id = worker_id
                item.attempts += 1
                state.assigned_count += 1
                try:
                    handle.enqueue_lease(item.item_id, lease_id, item.room)
                except queue.Full:
                    state.assigned_count = max(state.assigned_count - 1, 0)
                    item.status = "pending"
                    item.lease_id = None
                    item.leased_until = None
                    item.assigned_worker_id = None
                    self._pending_order.insert(0, item.item_id)
                    continue
                self._record_event("EXECUTE", {"item_id": item.item_id, "worker_id": worker_id})
                progressed = True
                if self._in_flight_count_locked() >= self.max_in_flight:
                    break
            if not progressed:
                break

    def _select_pending_for_worker_locked(self, worker: _WorkerState) -> str | None:
        gpu_workers_available = any(
            state.alive and not state.stopped and state.has_gpu for state in self._workers.values()
        )

        def choose(predicate: Callable[[WorkItem], bool]) -> str | None:
            for index, item_id in enumerate(self._pending_order):
                item = self._work_items[item_id]
                if predicate(item):
                    del self._pending_order[index]
                    return item_id
            return None

        if worker.has_gpu:
            chosen = choose(lambda item: item.requires_gpu)
            if chosen is not None:
                return chosen
            return choose(lambda item: True)

        chosen = choose(lambda item: not item.requires_gpu)
        if chosen is not None:
            return chosen
        if not gpu_workers_available:
            return choose(lambda item: item.requires_gpu)
        return None

    def _reclaim_expired_locked(self, now: float) -> list[str]:
        reclaimed: list[str] = []
        for worker_id, state in self._workers.items():
            if state.stopped:
                continue
            if now - state.last_heartbeat > self.heartbeat_timeout:
                state.alive = False
                reclaimed.extend(self._requeue_worker_items_locked(worker_id, "heartbeat-timeout"))
        for item in self._work_items.values():
            if item.status != "leased" or item.leased_until is None:
                continue
            if item.leased_until <= now:
                reclaimed.extend(self._requeue_item_locked(item, "lease-expired"))
        return reclaimed

    def _requeue_worker_items_locked(self, worker_id: str, reason: str) -> list[str]:
        reclaimed: list[str] = []
        for item in self._work_items.values():
            if item.status == "leased" and item.assigned_worker_id == worker_id:
                reclaimed.extend(self._requeue_item_locked(item, reason))
        state = self._workers.get(worker_id)
        if state is not None:
            state.assigned_count = 0
        return reclaimed

    def _requeue_item_locked(self, item: WorkItem, reason: str) -> list[str]:
        if item.status != "leased":
            return []
        worker_id = item.assigned_worker_id
        state = self._workers.get(worker_id or "")
        if state is not None:
            state.assigned_count = max(state.assigned_count - 1, 0)
        item.status = "pending"
        item.lease_id = None
        item.leased_until = None
        item.assigned_worker_id = None
        if item.item_id not in self._pending_order:
            self._pending_order.append(item.item_id)
        self._record_event("COLLAPSE", {"item_id": item.item_id, "reason": reason, "worker_id": worker_id})
        return [item.item_id]

    def _in_flight_count_locked(self) -> int:
        return sum(1 for item in self._work_items.values() if item.status == "leased")


def distributed_checkpoint(controller: Controller) -> dict[str, Any]:
    """Serialize a controller into a plain JSON-serializable checkpoint dict."""
    if not isinstance(controller, Controller):
        raise TypeError("controller must be a Controller")
    return controller.to_dict()


def restore_from_checkpoint(data: dict[str, Any]) -> Controller:
    """Restore a controller from a `distributed_checkpoint()` payload."""
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    return Controller.from_dict(data)


def autoscaling_policy(controller: Controller, target_utilization: float = 0.75) -> int:
    """Recommend how many workers to add (>0) or remove (<0).

    The heuristic is a small proportional controller over outstanding work:
    it estimates the worker count needed to keep `in_flight / capacity` near
    `target_utilization`, then returns the integer delta from the currently
    alive worker count.
    """
    if not isinstance(controller, Controller):
        raise TypeError("controller must be a Controller")
    target = _require_float("target_utilization", target_utilization)
    if not 0.0 < target <= 1.0:
        raise ValueError("target_utilization must be in (0, 1]")

    with controller._lock:
        workers = [state for state in controller._workers.values() if not state.stopped]
        alive_workers = [state for state in workers if state.alive]
        current_workers = len(alive_workers)
        if current_workers == 0:
            return 1 if controller._work_items else 0

        total_capacity = sum(state.capacity for state in alive_workers)
        pending = sum(1 for item in controller._work_items.values() if item.status == "pending")
        in_flight = sum(1 for item in controller._work_items.values() if item.status == "leased")
        done = sum(1 for item in controller._work_items.values() if item.status == "done")
        outstanding = pending + in_flight
        utilization = 0.0 if total_capacity == 0 else in_flight / total_capacity
        avg_capacity = total_capacity / current_workers

    if outstanding == 0 and done > 0 and current_workers > 1:
        return -1

    effective_capacity_per_worker = max(avg_capacity * target, 1e-9)
    desired_workers = max(1, math.ceil(outstanding / effective_capacity_per_worker)) if outstanding else 1
    delta = desired_workers - current_workers

    if delta == 0 and outstanding == 0 and utilization < target * 0.5 and current_workers > 1:  # pragma: no cover - defensive no-op autoscale branch
        return -1
    return delta
