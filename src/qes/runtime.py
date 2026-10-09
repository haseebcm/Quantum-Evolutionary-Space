"""Production-oriented runtime primitives for scheduling QES work.

The runtime layer intentionally sits above the core QES search primitives and
provides a small but practical execution environment: priority scheduling,
thread-based workers, persistent store serialization, telemetry, and retry-aware
failure handling. It is structured to support deployment-style execution without
committing to any single external infrastructure provider.
"""
from __future__ import annotations

import hmac
import itertools
import json
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

try:  # pragma: no cover - exercised indirectly via optional-dependency tests
    import redis as _redis_module
except ImportError:  # pragma: no cover
    _redis_module = None

try:  # pragma: no cover - exercised indirectly via optional-dependency tests
    import psycopg2 as _psycopg2_module
except ImportError:  # pragma: no cover
    _psycopg2_module = None

_id_counter = itertools.count(1)


def _next_job_id() -> str:
    return f"job-{uuid4().hex}"


@dataclass
class RuntimeTelemetry:
    """Summary snapshot for the scheduler queue and recent execution outcomes."""

    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    total: int = 0
    success_rate: float = 0.0


@dataclass
class RuntimeJob:
    """A runnable unit of work scheduled by the production runtime."""

    name: str
    handler: Callable[[], Any]
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=_next_job_id)
    sequence: int = field(default_factory=lambda: next(_id_counter))
    status: str = "queued"
    attempts: int = 0
    result: Any = None
    error: str | None = None

    def execute(self) -> Any:
        """Run the job and retain its outcome."""
        self.status = "running"
        self.attempts += 1
        try:
            self.result = self.handler()
            self.status = "completed"
            self.error = None
            return self.result
        except Exception as exc:  # pragma: no cover - exercised by caller tests
            self.status = "failed"
            self.error = str(exc)
            raise


class RuntimeStore:
    """Minimal in-memory persistence for runtime results and task metadata."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(initial or {})

    def put(self, key: str, value: Any) -> Any:
        self._data[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._data)

    def clear(self) -> None:
        self._data.clear()


class FileRuntimeStore(RuntimeStore):
    """A JSON-backed runtime store suitable for durable task metadata."""

    def __init__(self, path: str | Path, initial: dict[str, Any] | None = None):
        super().__init__(initial)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        return dict(self._data)

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        return dict(self._data)


class SQLiteRuntimeStore(RuntimeStore):
    """Durable SQLite-backed runtime store for deployment-oriented workloads.

    Safe to share across threads (e.g. a ``RuntimeScheduler`` dispatching jobs
    via a ``ThreadPoolExecutor``): the underlying connection is opened with
    ``check_same_thread=False`` and all access is serialized behind a lock,
    since SQLite connections are not otherwise safe for concurrent use from
    multiple threads.
    """

    def __init__(self, path: str | Path, initial: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        super().__init__(initial)
        if initial:
            for key, value in initial.items():
                self.put(key, value)

    def put(self, key: str, value: Any) -> Any:
        payload = json.dumps(value, sort_keys=True)
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(job_id, payload) VALUES (?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET payload = excluded.payload",
                (key, payload),
            )
            self._conn.commit()
            self._data[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT payload FROM jobs WHERE job_id = ?", (key,)).fetchone()
        if row is None:
            return default
        value = json.loads(row[0])
        self._data[key] = value
        return value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute("SELECT job_id, payload FROM jobs ORDER BY job_id").fetchall()
        data = {job_id: json.loads(payload) for job_id, payload in rows}
        self._data = data
        return dict(data)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs")
            self._conn.commit()
            self._data.clear()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class RedisRuntimeStore(RuntimeStore):
    """Redis-backed runtime store for shared, multi-process/multi-node deployments.

    Unlike ``SQLiteRuntimeStore`` (single-file, single-node durability), this
    store is suitable for a horizontally scaled worker pool: any number of
    ``ProductionRuntime``/``DistributedRuntime`` processes can share the same
    Redis instance and observe each other's job results. Requires the
    optional ``redis`` package (``pip install redis``); a clear ``ImportError``
    is raised otherwise so the dependency stays optional for users who only
    need in-memory, file, or SQLite durability.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        key_prefix: str = "qes:runtime:jobs",
        client: Any = None,
    ):
        if client is None:
            if _redis_module is None:
                raise ImportError(
                    "RedisRuntimeStore requires the optional 'redis' package. "
                    "Install it with `pip install redis`."
                )
            client = _redis_module.Redis.from_url(url, decode_responses=True)
        self._client = client
        self._key_prefix = key_prefix
        super().__init__()

    def put(self, key: str, value: Any) -> Any:
        self._client.hset(self._key_prefix, key, json.dumps(value, sort_keys=True))
        self._data[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        raw = self._client.hget(self._key_prefix, key)
        if raw is None:
            return default
        value = json.loads(raw)
        self._data[key] = value
        return value

    def snapshot(self) -> dict[str, Any]:
        raw_items = self._client.hgetall(self._key_prefix) or {}
        data = {key: json.loads(payload) for key, payload in raw_items.items()}
        self._data = data
        return dict(data)

    def clear(self) -> None:
        self._client.delete(self._key_prefix)
        self._data.clear()


class PostgresRuntimeStore(RuntimeStore):
    """Postgres-backed durable runtime store for production deployments.

    Provides the same interface as ``SQLiteRuntimeStore``/``RedisRuntimeStore``
    but targets a shared Postgres database, appropriate for multi-node worker
    pools that need transactional durability with SQL query access to job
    history. Requires the optional ``psycopg2`` (or ``psycopg2-binary``)
    package; a clear ``ImportError`` is raised otherwise.
    """

    def __init__(
        self,
        dsn: str = "",
        *,
        table: str = "qes_runtime_jobs",
        connection: Any = None,
    ):
        self._table = table
        self._lock = threading.Lock()
        if connection is None:
            if _psycopg2_module is None:
                raise ImportError(
                    "PostgresRuntimeStore requires the optional 'psycopg2' package. "
                    "Install it with `pip install psycopg2-binary`."
                )
            connection = _psycopg2_module.connect(dsn)
        self._conn = connection
        with self._lock, self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    job_id TEXT PRIMARY KEY,
                    payload JSONB NOT NULL
                )
                """
            )
        self._conn.commit()
        super().__init__()

    def put(self, key: str, value: Any) -> Any:
        payload = json.dumps(value, sort_keys=True)
        with self._lock, self._conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self._table} (job_id, payload) VALUES (%s, %s)
                ON CONFLICT (job_id) DO UPDATE SET payload = EXCLUDED.payload
                """,
                (key, payload),
            )
        self._conn.commit()
        self._data[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock, self._conn.cursor() as cursor:
            cursor.execute(f"SELECT payload FROM {self._table} WHERE job_id = %s", (key,))
            row = cursor.fetchone()
        if row is None:
            return default
        payload = row[0]
        value = json.loads(payload) if isinstance(payload, str) else payload
        self._data[key] = value
        return value

    def snapshot(self) -> dict[str, Any]:
        with self._lock, self._conn.cursor() as cursor:
            cursor.execute(f"SELECT job_id, payload FROM {self._table} ORDER BY job_id")
            rows = cursor.fetchall()
        data = {}
        for job_id, payload in rows:
            data[job_id] = json.loads(payload) if isinstance(payload, str) else payload
        self._data = data
        return dict(data)

    def clear(self) -> None:
        with self._lock, self._conn.cursor() as cursor:
            cursor.execute(f"DELETE FROM {self._table}")
        self._conn.commit()
        self._data.clear()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class RuntimeScheduler:
    """Prioritized scheduler for QES jobs with optional thread-based dispatch."""

    def __init__(self, *, max_workers: int = 1, store: RuntimeStore | None = None):
        self.max_workers = max_workers if max_workers > 0 else 1
        self.store = store or RuntimeStore()
        self._queue: list[RuntimeJob] = []
        self._state_lock = threading.Lock()
        self._running = 0

    @property
    def queue(self) -> list[RuntimeJob]:
        return list(self._queue)

    def submit(
        self,
        name: str,
        handler: Callable[[], Any],
        *,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
        retries: int = 0,
    ) -> RuntimeJob:
        """Queue a new job and return it for inspection.

        Raises:
            ValueError: if ``name`` is empty/non-string, ``priority``/``retries``
                are negative, or ``metadata`` is not JSON-serializable (jobs are
                persisted as JSON by every built-in store, so this fails fast
                at submission time rather than deep inside a worker thread).
            TypeError: if ``handler`` is not callable.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("job name must be a non-empty string")
        if not callable(handler):
            raise TypeError("job handler must be callable")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        metadata = metadata or {}
        if not isinstance(metadata, dict):
            raise ValueError("job metadata must be a dict")
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"job metadata must be JSON-serializable: {exc}") from exc

        job = RuntimeJob(
            name=name,
            handler=handler,
            priority=priority,
            metadata=dict(metadata),
        )
        job.metadata.setdefault("retries", retries)
        self._queue.append(job)
        return job

    def _sorted_jobs(self) -> list[RuntimeJob]:
        return sorted(self._queue, key=lambda item: (-item.priority, item.sequence))

    def telemetry(self) -> RuntimeTelemetry:
        queued = len(self._queue)
        records = self.store.snapshot().values()
        completed = sum(1 for rec in records if rec.get("status") == "completed")
        failed = sum(1 for rec in records if rec.get("status") == "failed")
        with self._state_lock:
            running = self._running
        total = queued + running + completed + failed
        success_rate = 0.0 if total == 0 else completed / max(total, 1)
        return RuntimeTelemetry(
            queued=queued,
            running=running,
            completed=completed,
            failed=failed,
            total=total,
            success_rate=success_rate,
        )

    def _run_single_job(self, job: RuntimeJob) -> None:
        with self._state_lock:
            self._running += 1
        try:
            self._execute_and_persist(job)
        finally:
            with self._state_lock:
                self._running -= 1

    def _execute_and_persist(self, job: RuntimeJob) -> None:
        max_retries = int(job.metadata.get("retries", 0))
        # Only handler failures may retry execution. Persistence failures must
        # never replay a handler that may already have performed side effects.
        while True:
            try:
                result = job.execute()
                record = {
                    "name": job.name, "status": job.status,
                    "priority": job.priority, "metadata": job.metadata,
                    "result": result, "attempts": job.attempts,
                }
                break
            except Exception as exc:
                if job.attempts > max_retries:
                    record = {
                        "name": job.name, "status": "failed",
                        "priority": job.priority, "metadata": job.metadata,
                        "error": f"{type(exc).__name__}: {exc}",
                        "attempts": job.attempts,
                    }
                    break
                job.status = "retrying"
                job.error = str(exc)

        for persistence_attempt in range(max_retries + 1):
            try:
                self.store.put(job.job_id, record)
                return
            except Exception as exc:
                if persistence_attempt == max_retries:
                    job.status = "persist_failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    raise

    def run_ready(self, *, limit: int | None = None) -> list[RuntimeJob]:
        """Execute the highest-priority queued jobs and persist their results.

        Raises:
            ValueError: if ``limit`` is provided and negative.
        """
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        ready = self._sorted_jobs()
        if limit is not None:
            ready = ready[:limit]
        if not ready:
            return []

        for job in ready:
            self._queue.remove(job)

        with ThreadPoolExecutor(max_workers=min(len(ready), self.max_workers)) as executor:
            futures = {executor.submit(self._run_single_job, job): job for job in ready}
            for future in futures:
                future.result()

        return ready

    def run_all(self) -> list[RuntimeJob]:
        """Drain the queue by processing all jobs in priority order."""
        jobs: list[RuntimeJob] = []
        while self._queue:
            jobs.extend(self.run_ready())
        return jobs


class RuntimeAccessError(PermissionError):
    """Raised when a caller fails runtime access control (missing/invalid API key)."""


class ApiKeyGuard:
    """Minimal API-key access control for a production runtime.

    Deployments that expose ``ProductionRuntime``/``DistributedRuntime`` behind
    a service boundary (the ``qes.worker`` container, an HTTP wrapper, etc.)
    can pass an ``ApiKeyGuard`` so that job submission and queue draining
    require a valid key. This intentionally stays minimal (constant-time
    comparison, no external dependency) rather than prescribing a specific
    auth provider; swap in an OAuth/JWT-based guard for stronger deployments
    by implementing the same ``authorize`` interface.
    """

    def __init__(self, valid_keys: set[str] | list[str] | None = None):
        self._valid_keys = set(valid_keys or [])

    def add_key(self, api_key: str) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string")
        self._valid_keys.add(api_key)

    def revoke_key(self, api_key: str) -> None:
        self._valid_keys.discard(api_key)

    def authorize(self, api_key: str | None) -> None:
        """Raise ``RuntimeAccessError`` unless ``api_key`` is a known key."""
        if api_key is None or not any(
            hmac.compare_digest(api_key, known) for known in self._valid_keys
        ):
            raise RuntimeAccessError("invalid or missing API key")


class ProductionRuntime:
    """User-facing runtime for orchestrating QES search tasks in production-like flows."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        store: RuntimeStore | None = None,
        access_guard: ApiKeyGuard | None = None,
    ):
        self.scheduler = RuntimeScheduler(max_workers=max_workers, store=store)
        self.access_guard = access_guard

    def _check_access(self, api_key: str | None) -> None:
        if self.access_guard is not None:
            self.access_guard.authorize(api_key)

    def submit(
        self,
        name: str,
        handler: Callable[[], Any],
        *,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
        retries: int = 0,
        api_key: str | None = None,
    ) -> RuntimeJob:
        self._check_access(api_key)
        return self.scheduler.submit(
            name,
            handler,
            priority=priority,
            metadata=metadata,
            retries=retries,
        )

    def run_ready(self, *, limit: int | None = None, api_key: str | None = None) -> list[RuntimeJob]:
        self._check_access(api_key)
        return self.scheduler.run_ready(limit=limit)

    def run_all(self, *, api_key: str | None = None) -> list[RuntimeJob]:
        self._check_access(api_key)
        return self.scheduler.run_all()

    def telemetry(self) -> RuntimeTelemetry:
        return self.scheduler.telemetry()

    def snapshot(self) -> dict[str, Any]:
        return self.scheduler.store.snapshot()


class DistributedRuntime(ProductionRuntime):
    """Deployment-oriented runtime facade for multi-worker execution."""

    def __init__(
        self,
        *,
        worker_count: int = 2,
        store: RuntimeStore | None = None,
        access_guard: ApiKeyGuard | None = None,
    ):
        super().__init__(max_workers=worker_count, store=store, access_guard=access_guard)
        self.worker_count = worker_count

    def status(self) -> dict[str, Any]:
        telemetry = self.telemetry()
        return {
            "workers": self.worker_count,
            "queued": telemetry.queued,
            "completed": telemetry.completed,
            "failed": telemetry.failed,
            "success_rate": telemetry.success_rate,
        }


__all__ = [
    "ApiKeyGuard",
    "DistributedRuntime",
    "FileRuntimeStore",
    "PostgresRuntimeStore",
    "ProductionRuntime",
    "RedisRuntimeStore",
    "RuntimeAccessError",
    "RuntimeJob",
    "RuntimeScheduler",
    "RuntimeStore",
    "RuntimeTelemetry",
    "SQLiteRuntimeStore",
]
