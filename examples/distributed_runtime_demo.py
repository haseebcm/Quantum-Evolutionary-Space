"""Example: deployment-oriented distributed runtime using a durable SQLite store.

This demonstrates a worker-style runtime abstraction that is suitable for a
production deployment pattern: multiple workers pull queued tasks, persistent
state is retained locally, and each completed task is recorded with status and
telemetry.
"""
from __future__ import annotations

from pathlib import Path

from qes.runtime import DistributedRuntime, SQLiteRuntimeStore


def main() -> None:
    db_path = Path("artifacts") / "qes_runtime.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    runtime = DistributedRuntime(
        worker_count=4,
        store=SQLiteRuntimeStore(db_path),
    )

    runtime.submit(
        "explore-control-policy",
        lambda: {"status": "ok", "score": 0.91},
        priority=8,
        metadata={"domain": "control"},
    )
    runtime.submit(
        "fit-scientific-model",
        lambda: {"status": "ok", "score": 0.86},
        priority=5,
        metadata={"domain": "science"},
    )
    runtime.submit(
        "allocate-resources",
        lambda: {"status": "ok", "score": 0.94},
        priority=9,
        metadata={"domain": "scheduling"},
    )

    jobs = runtime.run_all()
    print("Distributed QES runtime")
    print("=" * 28)
    for job in jobs:
        item = runtime.snapshot().get(job.job_id, {})
        print(f"{job.name:24s} | status={job.status} | result={item.get('result')}")
    print(runtime.status())


if __name__ == "__main__":
    main()
