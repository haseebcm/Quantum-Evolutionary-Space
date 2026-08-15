from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

import qes.worker as worker
from qes.runtime import SQLiteRuntimeStore


def test_state_dir_uses_environment_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "runtime-state"
    monkeypatch.setenv("QES_RUNTIME_STATE_DIR", str(target))
    assert worker._state_dir() == target
    assert target.is_dir()


def test_state_dir_defaults_to_local_state_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QES_RUNTIME_STATE_DIR", raising=False)
    assert worker._state_dir().name == "state"


def test_demo_jobs_returns_expected_placeholder_workload() -> None:
    assert worker._demo_jobs() == [
        ("heartbeat-control-policy", 5),
        ("heartbeat-scientific-fit", 4),
        ("heartbeat-resource-allocation", 6),
    ]


def test_run_processes_demo_jobs_with_sqlite_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QES_RUNTIME_STATE_DIR", str(tmp_path))
    worker.run(worker_count=2, cycles=1, interval_seconds=0.0)

    store = SQLiteRuntimeStore(tmp_path / "qes_runtime.sqlite")
    snapshot = store.snapshot()
    store.close()

    assert len(snapshot) == len(worker._demo_jobs())
    assert {record["name"] for record in snapshot.values()} == {
        "heartbeat-control-policy",
        "heartbeat-scientific-fit",
        "heartbeat-resource-allocation",
    }
    assert all(record["status"] == "completed" for record in snapshot.values())
    assert all(record["result"]["status"] == "ok" for record in snapshot.values())


def test_run_multiple_cycles_submits_multiple_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QES_RUNTIME_STATE_DIR", str(tmp_path))
    worker.run(worker_count=1, cycles=2, interval_seconds=0.0)

    store = SQLiteRuntimeStore(tmp_path / "qes_runtime.sqlite")
    snapshot = store.snapshot()
    store.close()

    assert len(snapshot) == len(worker._demo_jobs()) * 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"worker_count": 0, "cycles": 1, "interval_seconds": 0.0}, "worker_count must be > 0"),
        (
            {"worker_count": 1, "cycles": 1, "interval_seconds": -1.0},
            "interval_seconds must be >= 0",
        ),
    ],
)
def test_run_rejects_invalid_arguments(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        worker.run(**kwargs)


def test_run_handles_keyboard_interrupt_and_closes_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QES_RUNTIME_STATE_DIR", str(tmp_path))
    sleep_calls: list[float] = []

    def interrupting_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(worker.time, "sleep", interrupting_sleep)
    worker.run(worker_count=1, cycles=0, interval_seconds=0.0)

    store = SQLiteRuntimeStore(tmp_path / "qes_runtime.sqlite")
    assert len(store.snapshot()) == len(worker._demo_jobs())
    store.close()
    assert sleep_calls == [0.0]


def test_main_parses_arguments_and_invokes_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, float]] = []

    def fake_run(worker_count: int, cycles: int, interval_seconds: float) -> None:
        calls.append((worker_count, cycles, interval_seconds))

    monkeypatch.setattr(worker, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["qes.worker", "--worker-count", "7", "--cycles", "3", "--interval", "1.5"],
    )
    worker.main()
    assert calls == [(7, 3, 1.5)]


def test_module_entrypoint_executes_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QES_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["qes.worker", "--worker-count", "1", "--cycles", "1", "--interval", "0"],
    )
    runpy.run_module("qes.worker", run_name="__main__")
    assert (tmp_path / "qes_runtime.sqlite").exists()
