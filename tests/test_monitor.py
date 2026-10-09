import json
import sqlite3

import pytest

from qes.monitor import health_report, main, prometheus_text
from qes.task_queue import SQLiteTaskQueue


def test_alerts_cover_capacity_backlog_failure_expiry_and_liveness(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("qes.task_queue.time.time", lambda: clock[0])
    queue = SQLiteTaskQueue(tmp_path / "tasks.sqlite", max_records=3)
    queue.submit("work", {})
    failed = queue.claim("worker")
    queue.fail(failed, "failed", retry=False)
    queue.submit("work", {})
    queue.submit("work", {})
    queue.claim("dead", lease_seconds=1)
    clock[0] += 100
    report = health_report(queue, min_free_disk_bytes=0)
    assert set(report["alerts"]) == {"failed_tasks", "expired_leases", "queue_age",
                                      "queue_capacity", "no_live_workers"}
    assert report["metrics"]["counts"] == {"failed": 1, "running": 1, "queued": 1, "completed": 0}
    assert 'qes_alert{reason="expired_leases"} 1' in prometheus_text(report)
    queue.close()


def test_live_worker_then_clean_shutdown_and_stale_heartbeat(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("qes.task_queue.time.time", lambda: clock[0])
    queue = SQLiteTaskQueue(tmp_path / "tasks.sqlite")
    queue.heartbeat("worker")
    assert health_report(queue, require_worker=True, min_free_disk_bytes=0)["healthy"]
    clock[0] += 61
    assert "no_live_workers" in health_report(queue, require_worker=True)["alerts"]
    queue.heartbeat("worker", status="stopped")
    assert queue.operational_snapshot()["live_workers"] == 0
    queue.close()


def test_existing_v1_queue_migrates_without_losing_tasks(tmp_path):
    path = tmp_path / "old.sqlite"
    queue = SQLiteTaskQueue(path)
    task_id = queue.submit("work", {})
    queue.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE workers")
        conn.execute("PRAGMA user_version=1")
    queue = SQLiteTaskQueue(path)
    assert queue.get(task_id)["status"] == "queued"
    queue.heartbeat("new-worker")
    assert queue.operational_snapshot()["live_workers"] == 1
    queue.close()


def test_monitor_cli_health_and_alert_exit_codes(tmp_path, monkeypatch, capsys):
    path = tmp_path / "tasks.sqlite"
    queue = SQLiteTaskQueue(path)
    monkeypatch.setattr("sys.argv", ["qes-status", "--task-queue", str(path), "--min-free-disk-bytes", "0"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["healthy"]
    queue.submit("work", {})
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    assert "no_live_workers" in json.loads(capsys.readouterr().out)["alerts"]
    queue.close()
