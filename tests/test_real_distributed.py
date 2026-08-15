"""Tests for `qes.real_distributed` using real OS processes and real TCP sockets."""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import pytest

from qes.real_distributed import Coordinator


def sum_numbers(payload: Any) -> dict[str, Any]:
    """Return the sum of a list of finite numbers."""
    numbers = list(payload)
    return {"sum": sum(float(value) for value in numbers), "count": len(numbers)}


def slow_square(payload: Any) -> dict[str, Any]:
    """Sleep briefly, then square the provided value."""
    request = dict(payload)
    time.sleep(float(request.get("sleep_seconds", 0.0)))
    value = float(request["value"])
    return {"value": value, "square": value * value}


def wait_until(predicate: Any, *, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll until a predicate becomes true or a timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if bool(predicate()):
            return True
        time.sleep(interval)
    return bool(predicate())


@pytest.fixture
def coordinator_factory() -> Iterator[Any]:
    """Create coordinators with guaranteed cleanup."""
    created: list[Coordinator] = []

    def factory(worker_count: int = 3) -> Coordinator:
        coordinator = Coordinator(
            {"sum_numbers": sum_numbers, "slow_square": slow_square},
            worker_count=worker_count,
            heartbeat_interval=0.2,
            heartbeat_timeout=0.8,
            startup_timeout=20.0,
            task_timeout=4.0,
        )
        coordinator.start()
        created.append(coordinator)
        return coordinator

    yield factory

    for coordinator in reversed(created):
        coordinator.shutdown()


def test_submit_returns_result_from_real_worker_process(coordinator_factory: Any) -> None:
    """A task should execute in a separate OS process and return the expected result."""
    coordinator = coordinator_factory(worker_count=2)

    result = coordinator.submit("sum_numbers", [1, 2, 3, 4], timeout=4.0)

    assert result.result == {"sum": 10.0, "count": 4}
    assert result.worker_pid != os.getpid()
    assert result.duration_seconds >= 0.0
    assert result.task_id in coordinator.results()


def test_map_distributes_multiple_real_tasks_and_preserves_results(coordinator_factory: Any) -> None:
    """Multiple real tasks should complete correctly across the worker pool."""
    coordinator = coordinator_factory(worker_count=3)

    payloads = ([1, 2, 3], [4, 5], [10], [7, 8, 9])
    results = coordinator.map("sum_numbers", payloads, timeout=4.0)

    assert [item.result["sum"] for item in results] == [6.0, 9.0, 10.0, 24.0]
    assert all(item.worker_pid != os.getpid() for item in results)
    assert len(coordinator.results()) == 4


def test_worker_crash_is_detected_and_respawned(coordinator_factory: Any) -> None:
    """When a worker process dies, the coordinator should detect and respawn it."""
    coordinator = coordinator_factory(worker_count=2)
    before = coordinator.worker_snapshot()
    original_pid = int(before["worker-1"]["pid"])

    worker_process = coordinator._records["worker-1"].controller.process
    assert worker_process is not None
    worker_process.kill()
    worker_process.join(timeout=2.0)

    assert wait_until(
        lambda: coordinator.worker_snapshot()["worker-1"]["ready"]
        and int(coordinator.worker_snapshot()["worker-1"]["restart_count"]) >= 1
        and int(coordinator.worker_snapshot()["worker-1"]["pid"]) != original_pid,
        timeout=8.0,
    )

    result = coordinator.submit("sum_numbers", [5, 5], timeout=4.0)
    assert result.result["sum"] == 10.0


def test_shutdown_stops_all_worker_processes(coordinator_factory: Any) -> None:
    """Shutdown should stop every child worker process without leaving them alive."""
    coordinator = coordinator_factory(worker_count=2)
    controllers = [record.controller for record in coordinator._records.values()]

    coordinator.shutdown()

    assert wait_until(lambda: all(not controller.is_alive() for controller in controllers), timeout=4.0)

