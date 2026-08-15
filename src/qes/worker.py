"""Container/service entrypoint for running a QES production runtime worker pool.

This module is what ``python -m qes.worker`` executes inside the Docker image
(see ``Dockerfile`` / ``docker-compose.yml``). It wires a ``DistributedRuntime``
to a durable SQLite-backed store so job state survives container restarts,
then continuously drains a small demo workload and reports telemetry. Real
deployments should replace ``_demo_jobs`` with job submission from an external
queue, HTTP API, or message broker.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from qes.runtime import DistributedRuntime, SQLiteRuntimeStore

logging.basicConfig(
    level=os.environ.get("QES_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s qes.worker: %(message)s",
)
logger = logging.getLogger("qes.worker")


def _state_dir() -> Path:
    path = Path(os.environ.get("QES_RUNTIME_STATE_DIR", "./state"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _demo_jobs() -> list[tuple[str, int]]:
    """Placeholder workload; replace with real job intake in production."""
    return [
        ("heartbeat-control-policy", 5),
        ("heartbeat-scientific-fit", 4),
        ("heartbeat-resource-allocation", 6),
    ]


def run(worker_count: int, cycles: int, interval_seconds: float) -> None:
    if worker_count <= 0:
        raise ValueError(f"worker_count must be > 0, got {worker_count}")
    if interval_seconds < 0:
        raise ValueError(f"interval_seconds must be >= 0, got {interval_seconds}")

    store = SQLiteRuntimeStore(_state_dir() / "qes_runtime.sqlite")
    runtime = DistributedRuntime(worker_count=worker_count, store=store)
    logger.info("started worker pool with %d workers", worker_count)

    try:
        cycle = 0
        while cycles <= 0 or cycle < cycles:
            for name, priority in _demo_jobs():

                def handler(n: str = name) -> dict[str, str]:
                    return {"status": "ok", "job": n}

                runtime.submit(name, handler, priority=priority)
            jobs = runtime.run_all()
            status = runtime.status()
            logger.info(
                "cycle=%d ran=%d completed=%d failed=%d success_rate=%.2f",
                cycle,
                len(jobs),
                status["completed"],
                status["failed"],
                status["success_rate"],
            )
            cycle += 1
            if cycles <= 0 or cycle < cycles:
                time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("shutdown requested; draining worker pool")
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="QES production runtime worker service")
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument(
        "--cycles",
        type=int,
        default=int(os.environ.get("QES_WORKER_CYCLES", "0")),
        help="Number of scheduling cycles to run; 0 runs forever (default, for services).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("QES_WORKER_INTERVAL_SECONDS", "10")),
        help="Seconds to sleep between scheduling cycles.",
    )
    args = parser.parse_args()
    run(worker_count=args.worker_count, cycles=args.cycles, interval_seconds=args.interval)


if __name__ == "__main__":
    main()
