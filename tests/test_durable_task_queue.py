import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from qes.task_queue import RegisteredTaskWorker, SQLiteTaskQueue, StaleLeaseError
from qes.worker import run_registered


def test_durable_submission_survives_reopen_and_deduplicates(tmp_path):
    path = tmp_path / "queue.sqlite"
    queue = SQLiteTaskQueue(path)
    task_id = queue.submit("echo", {"x": 1}, idempotency_key="request")
    assert queue.submit("echo", {"x": 1}, idempotency_key="request") == task_id
    with pytest.raises(ValueError, match="different task"):
        queue.submit("echo", {"x": 2}, idempotency_key="request")
    queue.close()
    queue = SQLiteTaskQueue(path)
    lease = queue.claim("worker")
    assert lease.task_id == task_id
    queue.complete(lease, {"ok": True})
    assert queue.get(task_id)["result"] == {"ok": True}
    assert queue.get(task_id, tenant_id="different") is None
    queue.close()


def test_atomic_claims_across_independent_connections(tmp_path):
    path = tmp_path / "queue.sqlite"
    a, b = SQLiteTaskQueue(path), SQLiteTaskQueue(path)
    expected = {a.submit("echo", i) for i in range(20)}

    def drain(queue, owner):
        claimed = []
        while (lease := queue.claim(owner)) is not None:
            claimed.append(lease.task_id)
            queue.complete(lease, lease.payload)
        return claimed

    with ThreadPoolExecutor(2) as pool:
        one = pool.submit(drain, a, "a")
        two = pool.submit(drain, b, "b")
        ids = one.result() + two.result()
    assert set(ids) == expected
    assert len(ids) == len(expected)
    a.close()
    b.close()


def test_expired_worker_recovery_fences_stale_writes(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("qes.task_queue.time.time", lambda: clock[0])
    queue = SQLiteTaskQueue(tmp_path / "queue.sqlite")
    task_id = queue.submit("echo", 1)
    old = queue.claim("dead", lease_seconds=10)
    clock[0] += 11
    fresh = queue.claim("replacement", lease_seconds=10)
    assert fresh.task_id == old.task_id == task_id
    assert fresh.attempt == 2
    with pytest.raises(StaleLeaseError):
        queue.complete(old, "obsolete")
    with pytest.raises(StaleLeaseError):
        queue.renew(old)
    queue.complete(fresh, "fresh")
    assert queue.get(task_id)["result"] == "fresh"
    queue.close()


def test_attempt_limits_capacity_and_backup_restore(tmp_path):
    queue = SQLiteTaskQueue(tmp_path / "queue.sqlite", max_records=2, max_payload_bytes=50)
    task_id = queue.submit("fail", {}, max_attempts=1)
    lease = queue.claim("worker")
    queue.fail(lease, "test failure")
    assert queue.get(task_id)["status"] == "failed"
    queue.submit("echo", {})
    with pytest.raises(OverflowError):
        queue.submit("echo", {})
    with pytest.raises(ValueError, match="size limit"):
        queue.submit("echo", "x" * 100)
    backup = tmp_path / "backup.sqlite"
    queue.backup(backup)
    restored = SQLiteTaskQueue(backup)
    assert restored.get(task_id)["status"] == "failed"
    assert restored.claim("restore-worker").task_name == "echo"
    restored.close()
    assert queue.purge_terminal(before=time.time() + 1) == 1
    queue.close()


def test_registered_dispatch_receives_stable_task_id_and_rejects_unknown(tmp_path):
    queue = SQLiteTaskQueue(tmp_path / "queue.sqlite")
    task_id = queue.submit("echo", {"x": 1})
    received = []
    worker = RegisteredTaskWorker(queue, {"echo": lambda p, tid: received.append(tid) or p}, owner="worker")
    assert worker.run_one()
    assert received == [task_id]
    unknown = queue.submit("arbitrary.module.function", {})
    assert worker.run_one()
    assert queue.get(unknown)["status"] == "failed"
    assert not worker.run_one()
    queue.close()


def test_registered_queue_worker_runs_real_bounded_qes_search(tmp_path):
    path = tmp_path / "queue.sqlite"
    queue = SQLiteTaskQueue(path)
    task_id = queue.submit("qes.search", {"lower": [-1], "upper": [1], "max_evaluations": 20})
    queue.close()
    run_registered(str(path), cycles=1, interval_seconds=0)
    queue = SQLiteTaskQueue(path)
    record = queue.get(task_id)
    assert record["status"] == "completed"
    assert record["result"]["evaluations"] <= 20
    assert record["result"]["task_id"] == task_id
    queue.close()


def test_invalid_schema_is_rejected(tmp_path):
    import sqlite3

    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="schema version"):
        SQLiteTaskQueue(path)


def test_worker_processes_claim_without_duplicate_completion(tmp_path):
    path = tmp_path / "processes.sqlite"
    queue = SQLiteTaskQueue(path)
    ids = [queue.submit("qes.search", {"lower": [-1], "upper": [1],
                        "max_evaluations": 10}) for _ in range(2)]
    command = [sys.executable, "-m", "qes.worker", "--task-queue", str(path),
               "--cycles", "1", "--interval", "0"]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for _ in range(2)]
    try:
        for process in processes:
            _, error = process.communicate(timeout=20)
            assert process.returncode == 0, error.decode()
        records = [queue.get(task_id) for task_id in ids]
        assert all(record["status"] == "completed" for record in records)
        assert all(record["attempts"] == 1 for record in records)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        queue.close()


def test_sigterm_stops_idle_worker_cleanly(tmp_path):
    path = tmp_path / "shutdown.sqlite"
    queue = SQLiteTaskQueue(path)
    task_id = queue.submit("qes.search", {"max_evaluations": 2})
    process = subprocess.Popen([sys.executable, "-m", "qes.worker", "--task-queue", str(path),
                                "--interval", "30"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and queue.get(task_id)["status"] != "completed":
            time.sleep(0.01)
        assert queue.get(task_id)["status"] == "completed"
        process.terminate()
        _, error = process.communicate(timeout=5)
        assert process.returncode == 0, error.decode()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        queue.close()
