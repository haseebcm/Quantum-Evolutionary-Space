import sqlite3
import subprocess
import sys
import time

from qes.task_queue import SQLiteTaskQueue


def wait_for(predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("child process did not reach the recovery checkpoint")


def test_crash_after_effect_replay_uses_stable_idempotency_key(tmp_path):
    path = tmp_path / "tasks.sqlite"
    effects = tmp_path / "effects.sqlite"
    with sqlite3.connect(effects) as conn:
        conn.execute("CREATE TABLE effects(task_id TEXT PRIMARY KEY)")
    queue = SQLiteTaskQueue(path)
    task_id = queue.submit("effect", {})
    code = """
import sqlite3,sys,time
from qes.task_queue import SQLiteTaskQueue,RegisteredTaskWorker
q=SQLiteTaskQueue(sys.argv[1])
def handler(payload,task_id):
    with sqlite3.connect(sys.argv[2]) as conn:
        conn.execute('INSERT OR IGNORE INTO effects VALUES(?)',(task_id,))
    if sys.argv[3]=='crash':
        time.sleep(60)
    return {'task_id':task_id}
RegisteredTaskWorker(q,{'effect':handler},owner=sys.argv[3]).run_one(lease_seconds=1)
q.close()
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(path), str(effects), "crash"])

    def effect_applied():
        with sqlite3.connect(effects) as conn:
            return conn.execute("SELECT COUNT(*) FROM effects").fetchone()[0] == 1

    try:
        wait_for(effect_applied)
        process.kill()
        process.wait(timeout=5)
        expiry = queue.get(task_id)["expires_at"]
        wait_for(lambda: time.time() > expiry)
        subprocess.run([sys.executable, "-c", code, str(path), str(effects), "recovery"],
                       check=True, timeout=15)
        assert queue.get(task_id)["status"] == "completed"
        assert queue.get(task_id)["attempts"] == 2
        with sqlite3.connect(effects) as conn:
            assert conn.execute("SELECT task_id FROM effects").fetchall() == [(task_id,)]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        queue.close()


def test_sigterm_drains_inflight_registered_handler(tmp_path):
    path = tmp_path / "tasks.sqlite"
    marker = tmp_path / "started"
    queue = SQLiteTaskQueue(path)
    task_id = queue.submit("qes.search", {"max_evaluations": 10})
    code = """
import sys,time
from pathlib import Path
import qes.worker as worker
original=worker.search_task
def slow(payload,task_id):
    Path(sys.argv[2]).write_text('started')
    time.sleep(0.5)
    return original(payload,task_id)
worker.search_task=slow
worker.run_registered(sys.argv[1],cycles=0,interval_seconds=0.1,lease_seconds=5)
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(path), str(marker)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        wait_for(marker.exists)
        process.terminate()
        _, error = process.communicate(timeout=5)
        assert process.returncode == 0, error.decode()
        assert queue.get(task_id)["status"] == "completed"
        assert queue.operational_snapshot()["live_workers"] == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        queue.close()
