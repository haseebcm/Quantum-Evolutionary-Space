"""Real Redis 7 / PostgreSQL 16 durability, concurrency and outage acceptance.

Creates isolated Docker services on loopback with disposable persistent volumes.
No production credentials or database contents are used. Failures fail the gate.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from qes._version import __version__
from qes.runtime import PostgresRuntimeStore, RedisRuntimeStore, RuntimeScheduler


def docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True,
                          text=True, timeout=120).stdout.strip()


def store_for(kind: str, address: str):
    return (RedisRuntimeStore(address, key_prefix="qes:test", socket_timeout=1)
            if kind == "redis" else PostgresRuntimeStore(address, table="qes_test"))


def write_process(kind: str, address: str, index: int) -> None:
    store = store_for(kind, address)
    try:
        for number in range(20):
            store.put(f"worker-{index}-{number}", {"index": index, "number": number})
    finally:
        store.close()


def ready(kind: str, address: str) -> None:
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        try:
            store = store_for(kind, address)
            try:
                store.snapshot()
            finally:
                store.close()
            return
        except Exception:
            time.sleep(0.5)
    raise AssertionError(f"{kind} service did not become ready")


def validate(kind: str) -> dict:
    name = "qes-store-"+uuid4().hex[:12]
    volume = name+"-data"
    docker("volume", "create", volume)
    try:
        if kind == "redis":
            docker("run", "-d", "--name", name, "-p", "127.0.0.1::6379", "-v", volume+":/data",
                   "redis:7-alpine", "redis-server", "--appendonly", "yes", "--appendfsync", "always")
            port = docker("port", name, "6379/tcp").rsplit(":", 1)[1]
            address = f"redis://127.0.0.1:{port}/0"
        else:
            password = uuid4().hex  # Disposable fixture credential; never a production account.
            docker("run", "-d", "--name", name, "-p", "127.0.0.1::5432", "-v",
                   volume+":/var/lib/postgresql/data", "-e", "POSTGRES_PASSWORD="+password,
                   "postgres:16-alpine")
            port = docker("port", name, "5432/tcp").rsplit(":", 1)[1]
            address = f"postgresql://postgres:{password}@127.0.0.1:{port}/postgres"
        ready(kind, address)
        store = store_for(kind, address)
        report = {"service": kind, "image": "redis:7-alpine" if kind == "redis" else "postgres:16-alpine"}
        try:
            with ProcessPoolExecutor(3, mp_context=multiprocessing.get_context("spawn")) as pool:
                list(pool.map(write_process, [kind]*3, [address]*3, range(3)))
            assert len(store.snapshot()) == 60
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(lambda i: store.put(f"thread-{i}", {"i": i}), range(40)))
            assert len(store.snapshot()) == 100
            store.put("stable", {"committed": True})
            store.put("stable", {"committed": True})
            assert len(store.snapshot()) == 101
            report["process_and_thread_concurrency"] = "passed"
            docker("kill", name)
            started = time.monotonic()
            try:
                store.put("outage", {"must_not_be_cached": True})
            except Exception:
                report["outage_failure_seconds"] = time.monotonic()-started
            else:
                raise AssertionError("offline write unexpectedly succeeded")
            assert report["outage_failure_seconds"] < 15
            effects = []
            scheduler = RuntimeScheduler(store=store)
            job = scheduler.submit("persist-failure", lambda: effects.append(1) or {"ok": True})
            try:
                scheduler.run_all()
            except Exception:
                pass
            assert effects == [1] and job.status == "persist_failed"
            assert job.result == {"ok": True}
            report["persistence_failure_no_handler_replay"] = "passed"
            docker("start", name)
            ready(kind, address)
            assert store.get("stable") == {"committed": True}
            assert store.get("outage") is None
            assert len(store.snapshot()) == 101
            store.put(job.job_id, job.result)
            assert store.get(job.job_id) == job.result and effects == [1]
            report["forced_restart_durability_and_reconnect"] = "passed"
            if kind == "postgres":
                docker("exec", name, "pg_dump", "-U", "postgres", "-d", "postgres",
                       "--table", "qes_test", "--file", "/tmp/backup.sql")
                docker("exec", name, "psql", "-U", "postgres", "-d", "postgres", "-c", "DROP TABLE qes_test")
                docker("exec", name, "psql", "-v", "ON_ERROR_STOP=1", "-U", "postgres", "-d", "postgres",
                       "-f", "/tmp/backup.sql")
                assert store.get("stable") == {"committed": True}
                report["pg_dump_restore"] = "passed"
            else:
                docker("exec", name, "redis-cli", "SAVE")
                docker("exec", name, "cp", "/data/dump.rdb", "/data/backup.rdb")
                store.clear()
                assert store.snapshot() == {}
                # Restore RDB on the same disposable volume; disable AOF only for
                # this restore scenario so the saved RDB is the authoritative input.
                docker("stop", "--time", "5", name)
                docker("rm", name)
                docker("run", "--rm", "-v", volume+":/data", "redis:7-alpine", "sh", "-c",
                       "cp /data/backup.rdb /data/dump.rdb")
                docker("run", "-d", "--name", name, "-p", f"127.0.0.1:{port}:6379", "-v", volume+":/data",
                       "redis:7-alpine", "redis-server", "--appendonly", "no")
                ready(kind, address)
                store.close()
                store = store_for(kind, address)
                assert store.get("stable") == {"committed": True}
                report["rdb_restore"] = "passed"
            report["final_records"] = len(store.snapshot())
        finally:
            store.close()
        return report
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        subprocess.run(["docker", "volume", "rm", volume], capture_output=True, timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    report = {"qes_version": __version__, "results": [validate("redis"), validate("postgres")]}
    Path(args.json).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
