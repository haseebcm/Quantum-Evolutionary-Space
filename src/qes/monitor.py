"""Local queue health, actionable alerts and Prometheus text export.

No network listener or payload exposure. Supervisors can alert on exit status;
operators can send stdout to a metrics textfile collector or their log pipeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from qes.task_queue import SQLiteTaskQueue


def health_report(queue: SQLiteTaskQueue, *, max_queued_age: float = 60,
                  worker_timeout: float = 60, capacity_warning: float = 0.9,
                  min_free_disk_bytes: int = 104857600, require_worker: bool = False) -> dict[str, Any]:
    if not np.isfinite(max_queued_age) or max_queued_age < 0:
        raise ValueError("max_queued_age must be finite and nonnegative")
    if not np.isfinite(capacity_warning) or not 0 < capacity_warning <= 1:
        raise ValueError("capacity_warning must lie in (0,1]")
    if not isinstance(min_free_disk_bytes, int) or min_free_disk_bytes < 0:
        raise ValueError("min_free_disk_bytes must be nonnegative")
    snapshot = queue.operational_snapshot(worker_timeout=worker_timeout)
    snapshot["disk_free_bytes"] = shutil.disk_usage(queue.path.parent).free
    snapshot["database_bytes"] = sum(path.stat().st_size for path in (
        queue.path, Path(str(queue.path)+"-wal"), Path(str(queue.path)+"-shm")) if path.exists())
    alerts = []
    if snapshot["retained_records"] >= snapshot["record_capacity"] * capacity_warning:
        alerts.append("queue_capacity")
    if snapshot["counts"]["failed"]:
        alerts.append("failed_tasks")
    if snapshot["expired_leases"]:
        alerts.append("expired_leases")
    if snapshot["oldest_queued_seconds"] > max_queued_age:
        alerts.append("queue_age")
    pending = snapshot["counts"]["queued"] + snapshot["counts"]["running"]
    if (require_worker or pending) and snapshot["live_workers"] == 0:
        alerts.append("no_live_workers")
    if snapshot["disk_free_bytes"] < min_free_disk_bytes:
        alerts.append("disk_space")
    return {"healthy": not alerts, "alerts": alerts, "metrics": snapshot}


def prometheus_text(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = ["# TYPE qes_healthy gauge", f"qes_healthy {int(report['healthy'])}"]
    lines.append("# TYPE qes_tasks gauge")
    for status, count in metrics["counts"].items():
        lines.append(f'qes_tasks{{status="{status}"}} {count}')
    for key, value in metrics.items():
        if key != "counts":
            lines.extend([f"# TYPE qes_{key} gauge", f"qes_{key} {value}"])
    for alert in report["alerts"]:
        lines.append(f'qes_alert{{reason="{alert}"}} 1')
    return "\n".join(lines)+"\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect QES local queue health; exit 1 on alerts, 2 on errors")
    parser.add_argument("--task-queue", required=True)
    parser.add_argument("--format", choices=("json", "prometheus"), default="json")
    parser.add_argument("--max-queued-age", type=float, default=60)
    parser.add_argument("--worker-timeout", type=float, default=60)
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--min-free-disk-bytes", type=int, default=104857600)
    parser.add_argument("--require-worker", action="store_true")
    args = parser.parse_args()
    if not Path(args.task_queue).is_file():
        parser.error("queue file does not exist")
    try:
        queue = SQLiteTaskQueue(args.task_queue, max_records=args.max_records)
        try:
            report = health_report(queue, max_queued_age=args.max_queued_age,
                                   worker_timeout=args.worker_timeout,
                                   min_free_disk_bytes=args.min_free_disk_bytes,
                                   require_worker=args.require_worker)
        finally:
            queue.close()
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True) if args.format == "json" else prometheus_text(report), end="\n")
    raise SystemExit(0 if report["healthy"] else 1)


if __name__ == "__main__":
    main()
