"""Trusted single-host worker with durable registered-task intake.

The container defaults to ``--task-queue`` and drains submitted data-only tasks.
Without that option the entrypoint retains its legacy demonstration workload.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from qes.runtime import DistributedRuntime, SQLiteRuntimeStore
from qes.sdk import QESClient
from qes.task_queue import RegisteredTaskWorker, SQLiteTaskQueue

logging.basicConfig(
    level=os.environ.get("QES_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s qes.worker: %(message)s",
)
logger = logging.getLogger("qes.worker")


def _peak_rss_kib() -> int:
    if sys.platform == "win32":
        return 0  # Unknown on platforms without resource.getrusage.
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak / 1024 if sys.platform == "darwin" else peak)


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


def search_task(payload: object, task_id: str) -> dict:
    """A bounded registered search task; never deserialize executable callbacks."""
    if not isinstance(payload, dict):
        raise ValueError("search payload must be a mapping")
    allowed = {"lower", "upper", "objective", "population", "steps", "max_evaluations", "seed"}
    if set(payload) - allowed:
        raise ValueError("unknown search configuration fields")
    lower = np.asarray(payload.get("lower", [-5.0, -5.0]), dtype=float)
    upper = np.asarray(payload.get("upper", [5.0, 5.0]), dtype=float)
    if lower.ndim != 1 or not 1 <= lower.size <= 128:
        raise ValueError("search dimension must be in [1,128]")
    population = payload.get("population", 20)
    steps = payload.get("steps", 50)
    evaluations = payload.get("max_evaluations", 1000)
    for name, value, limit in (("population", population, 256), ("steps", steps, 1000),
                               ("max_evaluations", evaluations, 10000)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= limit:
            raise ValueError(f"{name} must lie in [1,{limit}]")
    kind = payload.get("objective", "sphere")
    if kind not in {"sphere", "rastrigin"}:
        raise ValueError("objective must be a registered sphere or rastrigin task")

    def objective(x: np.ndarray) -> float:
        if kind == "rastrigin":
            return float(10 * x.size + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))
        return float(x @ x)

    client = QESClient((lower, upper), objective=objective, population=population,
                       rng=payload.get("seed", 0), max_evaluations=evaluations, max_wall_time=30)
    result = client.run(steps)
    return {"task_id": task_id, "status": result.status, "best_state": None if result.best_state is None
            else result.best_state.tolist(), "best_score": result.best_score,
            "evaluations": result.evaluations, "stopping_reason": result.stopping_reason}


def run_registered(path: str, *, cycles: int, interval_seconds: float, lease_seconds: float = 300) -> None:
    """Drain a durable local queue with registered trusted handlers and SIGTERM support."""
    if cycles < 0 or not np.isfinite(interval_seconds) or interval_seconds < 0:
        raise ValueError("invalid queue polling configuration")
    queue = SQLiteTaskQueue(path)
    worker = RegisteredTaskWorker(queue, {"qes.search": search_task}, owner=f"worker-{os.getpid()}")
    stop = threading.Event()
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: stop.set())
    try:
        cycle = 0
        logger.info(json.dumps({"event": "worker_started", "owner": worker.owner}))
        while not stop.is_set() and (cycles == 0 or cycle < cycles):
            queue.heartbeat(worker.owner, status="running", peak_rss_kib=_peak_rss_kib())
            started = time.monotonic()
            worked = worker.run_one(lease_seconds=lease_seconds)
            queue.heartbeat(worker.owner, peak_rss_kib=_peak_rss_kib())
            if worked:
                logger.info(json.dumps({"event": "task_processed", "owner": worker.owner,
                                        "duration_seconds": time.monotonic()-started}))
            cycle += 1
            if not worked and not stop.is_set() and (cycles == 0 or cycle < cycles):
                # Keep liveness current even when operators choose long idle polls.
                remaining = interval_seconds
                while remaining > 0 and not stop.is_set():
                    wait = min(remaining, 10.0)
                    stop.wait(wait)
                    remaining -= wait
                    queue.heartbeat(worker.owner, peak_rss_kib=_peak_rss_kib())
    finally:
        try:
            queue.heartbeat(worker.owner, status="stopped", peak_rss_kib=_peak_rss_kib())
            logger.info(json.dumps({"event": "worker_stopped", "owner": worker.owner}))
        finally:
            queue.close()
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)


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
    parser.add_argument("--task-queue", default=os.environ.get("QES_TASK_QUEUE"),
                        help="Durable local SQLite task queue; otherwise runs legacy demo jobs")
    parser.add_argument("--lease-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.task_queue:
        run_registered(args.task_queue, cycles=args.cycles, interval_seconds=args.interval,
                       lease_seconds=args.lease_seconds)
    else:
        run(worker_count=args.worker_count, cycles=args.cycles, interval_seconds=args.interval)


if __name__ == "__main__":
    main()
