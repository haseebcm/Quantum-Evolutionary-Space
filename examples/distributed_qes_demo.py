"""Phase 11 demo: in-process distributed coordination for QES.

This demo is intentionally honest: workers are simulated as threads within a
single process; the coordination logic (heartbeats, work stealing, fault
tolerance, autoscaling policy) is real and would generalize to real
distributed workers behind a real transport, but no real network/multi-machine
execution occurs here.
"""
from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np

from qes.distributed import (
    Controller,
    Worker,
    autoscaling_policy,
    distributed_checkpoint,
    restore_from_checkpoint,
)
from qes.room import Room


def make_room(index: int, *, requires_gpu: bool) -> Room:
    base = float(index) / 10.0
    room = Room(
        x=np.asarray([base, base + 0.2, base + 0.4], dtype=float),
        x_star=np.ones(3, dtype=float) * 0.5,
        lower=-np.ones(3, dtype=float),
        upper=np.ones(3, dtype=float),
        activation=np.ones(3, dtype=float),
        compute={"compute": 1.0},
    )
    room.tag("index", index)
    room.tag("requires_gpu", requires_gpu)
    room.tag("crash_target", index == 0)
    return room


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("QES PHASE 11: DISTRIBUTED QES (IN-PROCESS SIMULATION)")
    print("=" * 60)

    controller = Controller(
        max_in_flight=3,
        lease_duration=0.08,
        heartbeat_timeout=0.08,
        dispatch_interval=0.005,
    )
    controller.start()

    crash_started = False
    crash_observed = False

    def make_process_fn(worker_name: str, *, slow_crash_target: bool = False):
        def process(room: Room) -> dict[str, Any]:
            if slow_crash_target and bool(room.get_tag("crash_target", False)):
                nonlocal crash_started
                crash_started = True
                time.sleep(0.12)
            else:
                time.sleep(0.01 if room.get_tag("requires_gpu", False) else 0.015)
            return {
                "worker": worker_name,
                "room_id": room.id,
                "requires_gpu": bool(room.get_tag("requires_gpu", False)),
                "score": round(float(np.sum(room.effective_state())), 4),
            }

        return process

    workers = [
        Worker(
            "worker-1",
            controller,
            make_process_fn("worker-1", slow_crash_target=True),
            heartbeat_interval=0.02,
        ),
        Worker("worker-2", controller, make_process_fn("worker-2"), has_gpu=True, heartbeat_interval=0.02),
        Worker("worker-3", controller, make_process_fn("worker-3"), heartbeat_interval=0.02),
    ]

    try:
        workers[0].start()
        crash_item_id = controller.submit(make_room(0, requires_gpu=False))

        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and not crash_started:
            time.sleep(0.005)

        workers[1].start()
        workers[2].start()
        if crash_started:
            workers[0].simulate_crash()
            crash_observed = True

        item_ids = [crash_item_id]
        for index in range(1, 12):
            requires_gpu = index % 4 == 0
            item_ids.append(
                controller.submit(make_room(index, requires_gpu=requires_gpu), requires_gpu=requires_gpu)
            )

        if not controller.wait_for_all(timeout=2.0):
            raise RuntimeError("distributed demo did not complete in time")

        snapshot = controller.worker_snapshot()
        print(f"[controller]   registered workers: {list(snapshot)}")
        print(
            f"[status]       work counts={controller.status_counts()} "
            f"results={len(controller.results())} crash_reclaimed={crash_observed}"
        )
        for worker_id, state in snapshot.items():
            age = time.monotonic() - float(state["last_heartbeat"])
            print(
                f"[heartbeat]    {worker_id}: alive={state['alive']} has_gpu={state['has_gpu']} "
                f"completed={state['completed_count']} heartbeat_age={age:.3f}s"
            )

        crash_result = controller.results()[crash_item_id]
        print(f"[fault_tol]    crash-target {crash_item_id} completed on {crash_result['worker']}")

        gpu_items = [
            item_id
            for item_id in item_ids
            if bool(controller.to_dict()["work_items"][item_id]["requires_gpu"])
        ]
        gpu_workers = {controller.results()[item_id]["worker"] for item_id in gpu_items}
        print(f"[gpu_sched]    GPU-hinted items={len(gpu_items)} assigned workers={sorted(gpu_workers)}")

        checkpoint = distributed_checkpoint(controller)
        restored = restore_from_checkpoint(checkpoint)
        restored_matches = restored.results() == controller.results()
        print(f"[checkpoint]   round-trip preserved aggregated results: {restored_matches}")

        scale_delta = autoscaling_policy(restored, target_utilization=0.75)
        print(f"[autoscaling]  recommendation delta workers: {scale_delta:+d}")

        elapsed = time.perf_counter() - t_start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("=" * 60)
        print("REAL MEASURED COST (this is ordinary CPU/RAM, not free compute)")
        print(f"  wall time            : {elapsed:.4f} s")
        print(f"  peak Python heap     : {peak / 1024:.1f} KB")
        print(f"  completed work items : {len(controller.results())}")
        print(
            "  workers are simulated as threads within a single process; the "
            "coordination logic (heartbeats, work stealing, fault tolerance, "
            "autoscaling policy) is real and would generalize to real "
            "distributed workers behind a real transport, but no real "
            "network/multi-machine execution occurs here."
        )
    finally:
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=1.0)
        controller.stop()
        controller.join(timeout=1.0)


if __name__ == "__main__":
    main()
