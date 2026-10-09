"""Repeatable load, crash, drain and backup-restore acceptance scenarios.

Run with an installed wheel or PYTHONPATH=src. --docker-image uses real non-root
containers with CPU/memory constraints on the same local filesystem instead of
native worker processes. Results describe this synthetic workload, not an SLA.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from qes._version import __version__
from qes.monitor import health_report
from qes.task_queue import SQLiteTaskQueue


class Workers:
    def __init__(self, directory: Path, image: str | None):
        self.directory = directory
        self.image = image
        self.processes: list[tuple[subprocess.Popen, str | None]] = []
        self.logs = []

    def start(self, *, claimer: bool = False) -> subprocess.Popen:
        name = None
        if claimer:
            code = ("import time; from qes.task_queue import SQLiteTaskQueue; "
                    "q=SQLiteTaskQueue('" + str('/app/state/tasks.sqlite' if self.image else self.directory/'tasks.sqlite') + "'); "
                    "q.heartbeat('crash-worker'); q.claim('crash-worker',lease_seconds=2); time.sleep(60)")
            args = ["-c", code]
        else:
            args = ["-m", "qes.worker", "--task-queue",
                    str('/app/state/tasks.sqlite' if self.image else self.directory/'tasks.sqlite'),
                    "--interval", "0.05", "--lease-seconds", "30"]
        if self.image:
            name = f"qes-acceptance-{os.getpid()}-{len(self.processes)}"
            command = ["docker", "run", "--name", name, "--cpus", "1", "--memory", "256m",
                       "--read-only", "--tmpfs", "/tmp", "--mount",
                       f"type=bind,source={self.directory},target=/app/state", "--entrypoint", "python",
                       self.image, *args]
        else:
            command = [sys.executable, *args]
        log = open(self.directory/f"worker-{len(self.processes)}.log", "w")
        self.logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append((process, name))
        return process

    def kill(self, process: subprocess.Popen) -> None:
        name = next(name for candidate, name in self.processes if candidate is process)
        if name:
            subprocess.run(["docker", "kill", name], check=True, capture_output=True, timeout=15)
        else:
            process.kill()
        process.wait(timeout=15)

    def drain(self) -> None:
        for process, name in self.processes:
            if process.poll() is not None:
                continue
            if name:
                subprocess.run(["docker", "stop", "--time", "10", name], check=True,
                               capture_output=True, timeout=20)
            else:
                process.terminate()
            assert process.wait(timeout=15) == 0, "worker did not drain cleanly"

    def close(self) -> None:
        for process, name in self.processes:
            if process.poll() is None:
                if name:
                    subprocess.run(["docker", "kill", name], capture_output=True, timeout=15)
                else:
                    process.kill()
                process.wait(timeout=15)
            if name:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        for log in self.logs:
            log.close()


def wait_for(predicate, *, timeout: float = 60) -> None:
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("acceptance scenario timed out")


def validate(image: str | None, task_count: int) -> dict:
    report = {"qes_version": __version__, "platform": platform.platform(), "python": platform.python_version(),
              "numpy": np.__version__, "executor": "docker" if image else "native",
              "task_count": task_count, "workers": 2}
    with tempfile.TemporaryDirectory(prefix="qes-acceptance-") as temporary:
        directory = Path(temporary)
        directory.chmod(0o777)  # Disposable fixture shared with the non-root image.
        queue = SQLiteTaskQueue(directory/"tasks.sqlite")
        workers = Workers(directory, image)
        try:
            ids = []
            for index in range(task_count):
                dimension = (2, 8, 16)[index % 3]
                ids.append(queue.submit("qes.search", {
                    "lower": [-5]*dimension, "upper": [5]*dimension,
                    "objective": "sphere" if index % 2 else "rastrigin",
                    "population": 12, "steps": 20, "max_evaluations": 200, "seed": index,
                }))
            if image:
                for path in directory.glob("tasks.sqlite*"):
                    path.chmod(0o666)
            started = time.monotonic()
            workers.start()
            workers.start()
            wait_for(lambda: queue.operational_snapshot()["counts"]["completed"] == task_count)
            elapsed = time.monotonic()-started
            records = [queue.get(task_id) for task_id in ids]
            assert all(record["attempts"] == 1 for record in records), "duplicate load claims"
            assert all(record["result"]["status"] == "success" for record in records)
            assert all(record["result"]["evaluations"] <= 200 for record in records)
            latencies = sorted(record["updated_at"]-record["created_at"] for record in records)
            report["load"] = {"elapsed_seconds": elapsed, "tasks_per_second": task_count/elapsed,
                              "latency_p50_seconds": float(np.percentile(latencies, 50)),
                              "latency_p95_seconds": float(np.percentile(latencies, 95)),
                              "queue_bytes_after_load": health_report(queue)["metrics"]["database_bytes"]}
            if image:
                name = workers.processes[-1][1]
                stats = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
                                       capture_output=True, text=True, check=True, timeout=15)
                report["container_resources"] = json.loads(stats.stdout)
                subprocess.run(["docker", "exec", name, "qes-status", "--task-queue",
                                "/app/state/tasks.sqlite", "--require-worker"],
                               check=True, capture_output=True, timeout=15)
                configuration = json.loads(subprocess.run(
                    ["docker", "inspect", name], check=True, capture_output=True,
                    text=True, timeout=15).stdout)[0]
                assert configuration["Config"]["User"] == "qes"
                assert configuration["HostConfig"]["ReadonlyRootfs"]
                assert configuration["HostConfig"]["Memory"] == 256 * 1024 * 1024
                report["container_health_and_limits"] = "passed"
            else:
                import resource

                report["worker_peak_rss_kib"] = queue.operational_snapshot()["worker_peak_rss_kib"]
            workers.drain()
            if not image:
                usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                report["worker_cpu_seconds"] = usage.ru_utime+usage.ru_stime
            assert queue.operational_snapshot()["live_workers"] == 0
            report["graceful_shutdown"] = "passed"

            crash_id = queue.submit("qes.search", {"max_evaluations": 20})
            crashed = workers.start(claimer=True)
            wait_for(lambda: queue.get(crash_id)["status"] == "running")
            workers.kill(crashed)
            expired_at = queue.get(crash_id)["expires_at"]
            wait_for(lambda: time.time() > expired_at)
            report["crash_alert"] = health_report(queue)["alerts"]
            assert "expired_leases" in report["crash_alert"]
            workers.start()
            wait_for(lambda: queue.get(crash_id)["status"] == "completed")
            assert queue.get(crash_id)["attempts"] == 2
            report["kill_and_restart"] = "passed"
            invalid_id = queue.submit("qes.search", {"objective": "invalid"})
            wait_for(lambda: queue.get(invalid_id)["status"] == "failed")
            assert queue.get(invalid_id)["attempts"] == 3
            assert "failed_tasks" in health_report(queue)["alerts"]
            report["failure_attempt_cap_and_alert"] = "passed"
            workers.drain()
            queue.backup(directory/"backup.sqlite")
            restored = SQLiteTaskQueue(directory/"backup.sqlite")
            try:
                assert restored.get(crash_id)["result"] == queue.get(crash_id)["result"]
                assert restored.operational_snapshot()["counts"]["completed"] == task_count+1
            finally:
                restored.close()
            report["backup_restore"] = "passed"
            report["final_metrics"] = queue.operational_snapshot()
        finally:
            workers.close()
            queue.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-image")
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    if not 1 <= args.tasks <= 1000:
        parser.error("tasks must lie in [1,1000]")
    report = validate(args.docker_image, args.tasks)
    Path(args.json).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
