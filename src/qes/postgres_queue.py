"""PostgreSQL-authoritative task coordination for independent network workers.

At-least-once delivery. Database time controls leases; fenced completion prevents
stale results. External effects still require task-ID idempotency. No peer election.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import threading
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from qes.task_queue import StaleLeaseError, TaskLease, _identifier, _positive_seconds


class PostgresTaskQueue:
    def __init__(self, dsn: str, *, namespace: str = "qes_queue", max_records: int = 10000,
                 max_payload_bytes: int = 1048576):
        if not isinstance(namespace, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,39}", namespace) is None:
            raise ValueError("namespace must be a lowercase SQL identifier of at most 40 characters")
        for value in (max_records, max_payload_bytes):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("queue limits must be positive integers")
        self.max_records, self.max_payload_bytes = max_records, max_payload_bytes
        self._driver: Any = importlib.import_module("psycopg2")
        self._dsn = dsn
        self._lock = threading.Lock()
        self._closed = False
        self._conn = self._connect()
        self._tasks, self._workers, self._meta = (namespace+s for s in ("_tasks", "_workers", "_meta"))
        self._lock_key = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big", signed=True)
        try:
            with self._transaction() as cursor:
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key,))
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {self._meta}(id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL)")
                cursor.execute(f"SELECT version FROM {self._meta} WHERE id=1")
                row = cursor.fetchone()
                if row is not None and row[0] != 1:
                    raise ValueError("unsupported PostgreSQL task schema version")
                cursor.execute(f"""CREATE TABLE IF NOT EXISTS {self._tasks}(
                    task_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, task_name TEXT NOT NULL,
                    payload JSONB NOT NULL, idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                    priority INTEGER NOT NULL, created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL, owner TEXT, lease_token TEXT,
                    expires_at DOUBLE PRECISION, attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL, result JSONB, error TEXT,
                    UNIQUE(tenant_id,idempotency_key))""")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {self._tasks}_claim ON {self._tasks}(status,priority DESC,created_at)")
                cursor.execute(f"""CREATE TABLE IF NOT EXISTS {self._workers}(
                    owner TEXT PRIMARY KEY,last_seen DOUBLE PRECISION NOT NULL,status TEXT NOT NULL,
                    peak_rss_kib BIGINT NOT NULL DEFAULT 0)""")
                cursor.execute(f"INSERT INTO {self._meta} VALUES(1,1) ON CONFLICT(id) DO NOTHING")
        except BaseException:
            self._conn.close()
            raise

    def _connect(self):
        return self._driver.connect(self._dsn, connect_timeout=3, options="-c statement_timeout=5000 -c lock_timeout=3000",
                                    tcp_user_timeout=5000, keepalives_idle=5, keepalives_interval=1, keepalives_count=2)

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("PostgreSQL task queue is closed")
            if self._conn.closed:
                self._conn = self._connect()
            try:
                with self._conn.cursor() as cursor:
                    yield cursor
                self._conn.commit()
            except BaseException:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise

    def _now(self, cursor) -> float:
        cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
        return float(cursor.fetchone()[0])

    def _json(self, value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > self.max_payload_bytes:
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
        with self._transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key,))
            cursor.execute(f"SELECT task_id,task_name,payload::text FROM {self._tasks} WHERE tenant_id=%s AND idempotency_key=%s", (tenant_id,key))
            old = cursor.fetchone()
            if old:
                if old[1] != task_name or json.loads(old[2]) != json.loads(encoded):
                    raise ValueError("idempotency key already belongs to different task content")
                return str(old[0])
            cursor.execute(f"SELECT COUNT(*) FROM {self._tasks}")
            if cursor.fetchone()[0] >= self.max_records:
                raise OverflowError("durable queue record capacity exceeded")
            task_id = uuid4().hex
            now = self._now(cursor)
            cursor.execute(f"""INSERT INTO {self._tasks}(task_id,tenant_id,task_name,payload,idempotency_key,
                status,priority,created_at,updated_at,max_attempts) VALUES(%s,%s,%s,%s::jsonb,%s,'queued',%s,%s,%s,%s)""",
                (task_id,tenant_id,task_name,encoded,key,priority,now,now,max_attempts))
            return task_id

    def claim(self, owner: str, *, lease_seconds: float = 30) -> TaskLease | None:
        _identifier(owner, "owner")
        seconds = _positive_seconds(lease_seconds)
        with self._transaction() as cursor:
            now = self._now(cursor)
            cursor.execute(f"""UPDATE {self._tasks} SET status='failed',error='attempt limit exhausted',updated_at=%s
                WHERE status='running' AND expires_at<=%s AND attempts>=max_attempts""", (now,now))
            cursor.execute(f"""SELECT task_id,tenant_id,task_name,payload::text,attempts FROM {self._tasks}
                WHERE (status='queued' OR (status='running' AND expires_at<=%s)) AND attempts<max_attempts
                ORDER BY priority DESC,created_at,task_id LIMIT 1 FOR UPDATE SKIP LOCKED""", (now,))
            row = cursor.fetchone()
            if row is None:
                return None
            token, expires = uuid4().hex, self._now(cursor)+seconds
            cursor.execute(f"""UPDATE {self._tasks} SET status='running',owner=%s,lease_token=%s,
                expires_at=%s,attempts=attempts+1,updated_at=%s WHERE task_id=%s""", (owner,token,expires,now,row[0]))
            return TaskLease(row[0],row[1],row[2],json.loads(row[3]),owner,token,expires,row[4]+1)

    def renew(self, lease: TaskLease, *, lease_seconds: float = 30) -> float:
        seconds = _positive_seconds(lease_seconds)
        with self._transaction() as cursor:
            now = self._now(cursor)
            expires = now+seconds
            cursor.execute(f"""UPDATE {self._tasks} SET expires_at=%s,updated_at=%s WHERE task_id=%s
                AND tenant_id=%s AND status='running' AND owner=%s AND lease_token=%s AND expires_at>%s""",
                (expires,now,lease.task_id,lease.tenant_id,lease.owner,lease.lease_token,now))
            if cursor.rowcount != 1:
                raise StaleLeaseError("cannot renew stale lease")
            return expires

    def complete(self, lease: TaskLease, result: Any) -> None:
        self._finish(lease,"completed",self._json(result),None)

    def fail(self, lease: TaskLease, error: str, *, retry: bool = True) -> None:
        self._finish(lease,"queued" if retry else "failed",None,str(error)[:4096])

    def _finish(self, lease: TaskLease, status: str, result: str | None, error: str | None) -> None:
        with self._transaction() as cursor:
            now = self._now(cursor)
            cursor.execute(f"""UPDATE {self._tasks} SET status=CASE WHEN %s='queued' AND attempts>=max_attempts
                THEN 'failed' ELSE %s END,result=%s::jsonb,error=%s,updated_at=%s,owner=NULL,lease_token=NULL,expires_at=NULL
                WHERE task_id=%s AND tenant_id=%s AND status='running' AND owner=%s AND lease_token=%s AND expires_at>%s""",
                (status,status,result,error,now,lease.task_id,lease.tenant_id,lease.owner,lease.lease_token,now))
            if cursor.rowcount != 1:
                raise StaleLeaseError("cannot finish stale or expired lease")

    def get(self, task_id: str, *, tenant_id: str = "local") -> dict[str, Any] | None:
        with self._transaction() as cursor:
            cursor.execute(f"SELECT *,payload::text AS payload_json,result::text AS result_json FROM {self._tasks} WHERE task_id=%s AND tenant_id=%s", (task_id,tenant_id))
            row = cursor.fetchone()
            if row is None:
                return None
            record = dict(zip([column[0] for column in cursor.description],row,strict=True))
            record['payload'] = json.loads(record.pop('payload_json'))
            encoded = record.pop('result_json')
            record['result'] = None if encoded is None else json.loads(encoded)
            return record

    def heartbeat(self, owner: str, *, status: str = "idle", peak_rss_kib: int = 0) -> None:
        _identifier(owner,"owner")
        if status not in {'idle','running','stopped'} or not isinstance(peak_rss_kib,int) or peak_rss_kib<0:
            raise ValueError("invalid worker heartbeat")
        with self._transaction() as cursor:
            now = self._now(cursor)
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key,))
            cursor.execute(f"DELETE FROM {self._workers} WHERE last_seen<%s", (now-86400,))
            cursor.execute(f"SELECT COUNT(*) FROM {self._workers}")
            if cursor.fetchone()[0] >= 1000:
                cursor.execute(f"SELECT 1 FROM {self._workers} WHERE owner=%s", (owner,))
                if cursor.fetchone() is None:
                    raise OverflowError("worker identity capacity exceeded")
            cursor.execute(f"""INSERT INTO {self._workers} VALUES(%s,%s,%s,%s) ON CONFLICT(owner) DO UPDATE
                SET last_seen=excluded.last_seen,status=excluded.status,
                peak_rss_kib=GREATEST({self._workers}.peak_rss_kib,excluded.peak_rss_kib)""", (owner,now,status,peak_rss_kib))

    def operational_snapshot(self, *, worker_timeout: float = 60) -> dict[str, Any]:
        seconds = _positive_seconds(worker_timeout)
        with self._transaction() as cursor:
            now = self._now(cursor)
            cursor.execute(f"SELECT status,COUNT(*) FROM {self._tasks} GROUP BY status")
            counts = dict(cursor.fetchall())
            cursor.execute(f"SELECT MIN(created_at),COALESCE(SUM(GREATEST(attempts-1,0)),0) FROM {self._tasks}")
            _, retries = cursor.fetchone()
            cursor.execute(f"SELECT MIN(created_at) FROM {self._tasks} WHERE status='queued'")
            oldest = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {self._tasks} WHERE status='running' AND expires_at<=%s", (now,))
            expired = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {self._workers} WHERE status!='stopped' AND last_seen>%s", (now-seconds,))
            live = cursor.fetchone()[0]
            cursor.execute(f"SELECT COALESCE(MAX(peak_rss_kib),0) FROM {self._workers}")
            peak = cursor.fetchone()[0]
        return {'counts':{key:counts.get(key,0) for key in ('queued','running','completed','failed')},
                'retained_records':sum(counts.values()),'record_capacity':self.max_records,
                'oldest_queued_seconds':0.0 if oldest is None else max(0.0,now-oldest),
                'expired_leases':expired,'retry_attempts':retries,'live_workers':live,'worker_peak_rss_kib':peak}

    def purge_terminal(self, *, before: float) -> int:
        import math
        if not math.isfinite(before):
            raise ValueError("before must be finite")
        with self._transaction() as cursor:
            cursor.execute(f"DELETE FROM {self._tasks} WHERE status IN ('completed','failed') AND updated_at<%s", (before,))
            return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._conn.close()
