"""Honest demo of QES real_distributed on one localhost machine.

This runs multiple real OS processes communicating over real TCP sockets,
unlike `distributed.py` which simulates workers with threads. It has only been
tested on a single machine (localhost) — it has NOT been validated across
multiple physical/virtual machines, a real cluster, or at scale.
"""
from __future__ import annotations

import time
from typing import Any

from qes.real_distributed import Coordinator


def summarize_numbers(payload: Any) -> dict[str, Any]:
    """Return a small summary for one numeric payload."""
    request = dict(payload)
    numbers = [float(value) for value in request["numbers"]]
    return {
        "label": str(request["label"]),
        "count": len(numbers),
        "sum": sum(numbers),
        "maximum": max(numbers),
    }


def main() -> None:
    """Run a small real multi-process demo on localhost."""
    payloads = [
        {"label": "task-a", "numbers": [1, 2, 3, 4]},
        {"label": "task-b", "numbers": [5, 6, 7]},
        {"label": "task-c", "numbers": [2, 4, 6, 8]},
        {"label": "task-d", "numbers": [10, 20]},
        {"label": "task-e", "numbers": [3, 9, 12]},
        {"label": "task-f", "numbers": [11, 12, 13]},
    ]
    started_at = time.perf_counter()

    print("QES REAL DISTRIBUTED DEMO (localhost only)")
    print("=" * 48)

    with Coordinator(
        {"summarize_numbers": summarize_numbers},
        worker_count=4,
        heartbeat_interval=0.2,
        heartbeat_timeout=0.8,
        startup_timeout=8.0,
        task_timeout=4.0,
    ) as coordinator:
        results = coordinator.map("summarize_numbers", payloads, timeout=4.0)
        for item in results:
            print(
                f"{item.task_name} {item.task_id} executed by PID {item.worker_pid} "
                f"({item.worker_id}) -> {item.result}"
            )
        snapshot = coordinator.worker_snapshot()
        print("-" * 48)
        print(f"ready workers: {sorted(snapshot)}")
        print(f"completed tasks: {len(coordinator.results())}")

    elapsed = time.perf_counter() - started_at
    print(f"wall-clock time: {elapsed:.4f} seconds")
    print("Measured on this one machine only; no cluster or scale claims are implied.")


if __name__ == "__main__":
    main()

