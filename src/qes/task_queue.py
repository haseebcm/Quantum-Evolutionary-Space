"""Durable single-host registered-task intake with atomic leases and fencing.

Delivery is at least once. Handlers must make external effects idempotent using
the task ID. SQLite provides persistence/coordination, not OS callback isolation
or distributed consensus. Use one local filesystem, not a network-mounted DB.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class TaskLease:
    task_id: str
    tenant_id: str
    task_name: str
    payload: Any
    owner: str
    lease_token: str
    expires_at: float
    attempt: int


class StaleLeaseError(RuntimeError):
    """The lease has expired, completed, or been superseded by a newer claim."""


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
    return value


def _positive_seconds(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError("lease_seconds must be finite and positive")
    return float(value)


class SQLiteTaskQueue:
    """Data-only task records committed before submission returns.

    max_records bounds total retained rows, including completed history; purge
    terminal records deliberately to reclaim capacity. Every operation is an
    atomic transaction. A lease token fences result writes from stale workers.
    """

    def __init__(self, path: str | Path, *, max_records: int = 10000, max_payload_bytes: int = 1048576):
        for name, value in (("max_records", max_records), ("max_payload_bytes", max_payload_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_records = max_records
        self.max_payload_bytes = max_payload_bytes
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2):
            self._conn.close()
            raise ValueError("unsupported durable queue schema version")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, task_name TEXT NOT NULL,
            payload TEXT NOT NULL, idempotency_key TEXT NOT NULL, status TEXT NOT NULL,
            priority INTEGER NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            owner TEXT, lease_token TEXT, expires_at REAL, attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL, result TEXT, error TEXT,
            UNIQUE(tenant_id, idempotency_key))""")
        self._conn.execute("CREATE INDEX IF NOT EXISTS tasks_claim ON tasks(status, priority, created_at)")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS workers (
            owner TEXT PRIMARY KEY, last_seen REAL NOT NULL, status TEXT NOT NULL,
            peak_rss_kib INTEGER NOT NULL DEFAULT 0)""")
        self._conn.execute("PRAGMA user_version=2")
        self._conn.commit()

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _json(self, value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf8")) > self.max_payload_bytes:
            raise ValueError("task payload/result exceeds configured size limit")
        return encoded

    def submit(self, task_name: str, payload: Any, *, tenant_id: str = "local",
               idempotency_key: str | None = None, priority: int = 0, max_attempts: int = 3) -> str:
        _identifier(task_name, "task_name")
        _identifier(tenant_id, "tenant_id")
        key = _identifier(uuid4().hex if idempotency_key is None else idempotency_key, "idempotency_key")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise ValueError("priority must be an integer")
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        encoded = self._json(payload)
        now = time.time()
        with self._transaction() as conn:
            previous = conn.execute("SELECT task_id,task_name,payload FROM tasks WHERE tenant_id=? AND idempotency_key=?",
                                    (tenant_id, key)).fetchone()
            if previous is not None:
                if previous["task_name"] != task_name or previous["payload"] != encoded:
                    raise ValueError("idempotency key already belongs to different task content")
                return str(previous["task_id"])
            count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            if count >= self.max_records:
                raise OverflowError("durable queue record capacity exceeded")
            task_id = uuid4().hex
            conn.execute("""INSERT INTO tasks(task_id,tenant_id,task_name,payload,idempotency_key,
                         status,priority,created_at,updated_at,max_attempts) VALUES(?,?,?,?,?,'queued',?,?,?,?)""",
                         (task_id, tenant_id, task_name, encoded, key, priority, now, now, max_attempts))
            return task_id

    def claim(self, owner: str, *, lease_seconds: float = 30) -> TaskLease | None:
        _identifier(owner, "owner")
        seconds = _positive_seconds(lease_seconds)
        now = time.time()
        with self._transaction() as conn:
            conn.execute("""UPDATE tasks SET status='failed',error='attempt limit exhausted',updated_at=?
                         WHERE status='running' AND expires_at<=? AND attempts>=max_attempts""", (now, now))
            row = conn.execute("""SELECT * FROM tasks WHERE (status='queued' OR
                               (status='running' AND expires_at<=?)) AND attempts<max_attempts
                               ORDER BY priority DESC, created_at,task_id LIMIT 1""", (now,)).fetchone()
            if row is None:
                return None
            token = uuid4().hex
            expires = now + seconds
            conn.execute("""UPDATE tasks SET status='running',owner=?,lease_token=?,expires_at=?,
                         attempts=attempts+1,updated_at=? WHERE task_id=?""",
                         (owner, token, expires, now, row["task_id"]))
            return TaskLease(str(row["task_id"]), str(row["tenant_id"]), str(row["task_name"]),
                             json.loads(row["payload"]), owner, token, expires, int(row["attempts"]) + 1)

    def renew(self, lease: TaskLease, *, lease_seconds: float = 30) -> float:
        expires = time.time() + _positive_seconds(lease_seconds)
        with self._transaction() as conn:
            cursor = conn.execute("""UPDATE tasks SET expires_at=?,updated_at=? WHERE task_id=? AND
                                  status='running' AND lease_token=? AND owner=? AND expires_at>?""",
                                  (expires, time.time(), lease.task_id, lease.lease_token, lease.owner, time.time()))
            if cursor.rowcount != 1:
                raise StaleLeaseError("cannot renew stale lease")
        return expires

    def complete(self, lease: TaskLease, result: Any) -> None:
        encoded = self._json(result)
        self._finish(lease, "completed", encoded, None)

    def fail(self, lease: TaskLease, error: str, *, retry: bool = True) -> None:
        _identifier(error[:256], "error")
        status = "queued" if retry else "failed"
        self._finish(lease, status, None, error[:4096])

    def _finish(self, lease: TaskLease, status: str, result: str | None, error: str | None) -> None:
        now = time.time()
        with self._transaction() as conn:
            cursor = conn.execute("""UPDATE tasks SET status=CASE WHEN ?='queued' AND attempts>=max_attempts
                                  THEN 'failed' ELSE ? END,result=?,error=?,updated_at=?,owner=NULL,
                                  lease_token=NULL,expires_at=NULL WHERE task_id=? AND status='running'
                                  AND lease_token=? AND owner=? AND expires_at>?""",
                                  (status, status, result, error, now, lease.task_id, lease.lease_token, lease.owner, now))
            if cursor.rowcount != 1:
                raise StaleLeaseError("cannot finish a stale or expired lease")

    def get(self, task_id: str, *, tenant_id: str = "local") -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE task_id=? AND tenant_id=?",
                                     (task_id, tenant_id)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["payload"] = json.loads(record["payload"])
        record["result"] = None if record["result"] is None else json.loads(record["result"])
        return record

    def purge_terminal(self, *, before: float) -> int:
        if not np.isfinite(before):
            raise ValueError("before must be finite")
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE status IN ('completed','failed') AND updated_at<?", (before,))
            return cursor.rowcount

    def heartbeat(self, owner: str, *, status: str = "idle", peak_rss_kib: int = 0) -> None:
        """Record worker liveness; old terminal identities are retained for one day."""
        _identifier(owner, "owner")
        if status not in {"idle", "running", "stopped"}:
            raise ValueError("invalid worker status")
        if not isinstance(peak_rss_kib, int) or peak_rss_kib < 0:
            raise ValueError("peak_rss_kib must be nonnegative")
        now = time.time()
        with self._transaction() as conn:
            conn.execute("DELETE FROM workers WHERE last_seen<?", (now - 86400,))
            if conn.execute("SELECT COUNT(*) FROM workers").fetchone()[0] >= 1000 and (
                conn.execute("SELECT 1 FROM workers WHERE owner=?", (owner,)).fetchone() is None
            ):
                raise OverflowError("worker identity capacity exceeded")
            conn.execute("""INSERT INTO workers VALUES(?,?,?,?) ON CONFLICT(owner) DO UPDATE
                         SET last_seen=excluded.last_seen,status=excluded.status,
                         peak_rss_kib=MAX(workers.peak_rss_kib,excluded.peak_rss_kib)""",
                         (owner, now, status, peak_rss_kib))

    def operational_snapshot(self, *, worker_timeout: float = 60) -> dict[str, Any]:
        """Aggregate task/liveness metrics without exposing tenant payloads."""
        _positive_seconds(worker_timeout)
        now = time.time()
        with self._transaction() as conn:
            counts = {row[0]: row[1] for row in conn.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status")}
            oldest = conn.execute("SELECT MIN(created_at) FROM tasks WHERE status='queued'").fetchone()[0]
            expired = conn.execute("SELECT COUNT(*) FROM tasks WHERE status='running' AND expires_at<=?",
                                   (now,)).fetchone()[0]
            retries = conn.execute("SELECT COALESCE(SUM(MAX(attempts-1,0)),0) FROM tasks").fetchone()[0]
            live = conn.execute("SELECT COUNT(*) FROM workers WHERE status!='stopped' AND last_seen>?",
                                (now-worker_timeout,)).fetchone()[0]
            peak = conn.execute("SELECT COALESCE(MAX(peak_rss_kib),0) FROM workers").fetchone()[0]
        return {"counts": {key: counts.get(key, 0) for key in ("queued", "running", "completed", "failed")},
                "retained_records": sum(counts.values()), "record_capacity": self.max_records,
                "oldest_queued_seconds": 0.0 if oldest is None else max(0.0, now-oldest),
                "expired_leases": expired, "retry_attempts": retries, "live_workers": live,
                "worker_peak_rss_kib": peak}

    def backup(self, path: str | Path) -> None:
        """Create a consistent SQLite backup under the queue lock."""
        destination = Path(path)
        if destination.resolve() == self.path.resolve():
            raise ValueError("backup destination must differ from the active queue")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(str(destination)) as target:
            self._conn.backup(target)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class RegisteredTaskWorker:
    """Trusted task dispatch; handlers receive payload and stable operation ID.

    Completion persistence failure is surfaced, never retried by re-running the
    handler here. A later lease reclaim may repeat work: use idempotent handlers.
    Callers renew leases for long-running work or choose an adequate lease size.
    """

    def __init__(self, queue: SQLiteTaskQueue, handlers: dict[str, Callable[[Any, str], Any]], *, owner: str):
        if not handlers or any(not callable(handler) for handler in handlers.values()):
            raise ValueError("handlers must be a nonempty registered callable mapping")
        self.queue = queue
        self.handlers = dict(handlers)
        self.owner = _identifier(owner, "owner")

    def run_one(self, *, lease_seconds: float = 30) -> bool:
        lease = self.queue.claim(self.owner, lease_seconds=lease_seconds)
        if lease is None:
            return False
        handler = self.handlers.get(lease.task_name)
        if handler is None:
            self.queue.fail(lease, f"unregistered task: {lease.task_name}", retry=False)
            return True
        try:
            result = handler(lease.payload, lease.task_id)
        except Exception as exc:
            self.queue.fail(lease, f"{type(exc).__name__}: {exc}")
            return True
        self.queue.complete(lease, result)
        return True
