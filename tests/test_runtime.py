import numpy as np
import pytest

import qes.runtime as runtime_module
from qes.runtime import (
    ApiKeyGuard,
    DistributedRuntime,
    FileRuntimeStore,
    PostgresRuntimeStore,
    ProductionRuntime,
    RedisRuntimeStore,
    RuntimeAccessError,
    RuntimeScheduler,
    RuntimeStore,
    SQLiteRuntimeStore,
)


class _FakeRedisClient:
    """Minimal in-memory stand-in for a redis.Redis client's hash operations."""

    def __init__(self):
        self._hashes: dict[str, dict[str, str]] = {}

    def hset(self, key, field, value):
        self._hashes.setdefault(key, {})[field] = value

    def hget(self, key, field):
        return self._hashes.get(key, {}).get(field)

    def hgetall(self, key):
        return dict(self._hashes.get(key, {}))

    def delete(self, key):
        self._hashes.pop(key, None)


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, sql, params=None):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("CREATE TABLE"):
            return
        if sql_norm.startswith("INSERT INTO"):
            job_id, payload = params
            self._conn.rows[job_id] = payload
        elif sql_norm.startswith("SELECT payload FROM"):
            job_id = params[0]
            payload = self._conn.rows.get(job_id)
            self._last = [(payload,)] if payload is not None else []
        elif sql_norm.startswith("SELECT job_id, payload FROM"):
            self._last = sorted(self._conn.rows.items())
        elif sql_norm.startswith("DELETE FROM"):
            self._conn.rows.clear()

    def fetchone(self):
        rows = getattr(self, "_last", [])
        return rows[0] if rows else None

    def fetchall(self):
        return getattr(self, "_last", [])


class _FakePostgresConnection:
    """Minimal in-memory stand-in for a psycopg2 connection."""

    def __init__(self):
        self.rows: dict[str, str] = {}
        self.committed = False
        self.closed = False

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_redis_runtime_store_round_trip():
    store = RedisRuntimeStore(client=_FakeRedisClient())
    assert store.snapshot() == {}
    store.put("job-1", {"status": "queued"})
    assert store.get("job-1") == {"status": "queued"}
    assert store.get("missing", {"status": "none"}) == {"status": "none"}
    assert store.snapshot()["job-1"]["status"] == "queued"
    store.clear()
    assert store.snapshot() == {}


def test_redis_runtime_store_requires_dependency_without_client(monkeypatch):
    import qes.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_redis_module", None)
    try:
        RedisRuntimeStore()
    except ImportError as exc:
        assert "redis" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ImportError when redis is unavailable")


def test_postgres_runtime_store_round_trip():
    store = PostgresRuntimeStore(connection=_FakePostgresConnection())
    assert store.snapshot() == {}
    store.put("job-1", {"status": "queued"})
    assert store.get("job-1") == {"status": "queued"}
    assert store.snapshot()["job-1"]["status"] == "queued"
    store.clear()
    assert store.snapshot() == {}
    store.close()


def test_postgres_runtime_store_requires_dependency_without_connection(monkeypatch):
    import qes.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_psycopg2_module", None)
    try:
        PostgresRuntimeStore()
    except ImportError as exc:
        assert "psycopg2" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ImportError when psycopg2 is unavailable")


def test_runtime_store_round_trip():
    store = RuntimeStore()
    assert store.snapshot() == {}
    store.put("alpha", {"status": "ok"})
    assert store.get("alpha") == {"status": "ok"}
    assert store.snapshot()["alpha"]["status"] == "ok"


def test_runtime_store_file_persistence(tmp_path):
    path = tmp_path / "runtime.json"
    store = FileRuntimeStore(path)
    store.put("beta", {"status": "persisted"})
    store.save()
    reopened = FileRuntimeStore(path)
    assert reopened.get("beta") == {"status": "persisted"}


def test_sqlite_runtime_store_round_trip(tmp_path):
    path = tmp_path / "runtime.sqlite"
    store = SQLiteRuntimeStore(path)
    store.put("job-1", {"status": "queued"})
    assert store.get("job-1") == {"status": "queued"}
    snapshot = store.snapshot()
    assert snapshot["job-1"]["status"] == "queued"
    store.close()


def test_scheduler_prioritizes_highest_priority_jobs():
    scheduler = RuntimeScheduler(max_workers=2)
    scheduler.submit("low", lambda: 1, priority=1)
    scheduler.submit("high", lambda: 2, priority=5)
    scheduler.submit("mid", lambda: 3, priority=3)

    jobs = scheduler.run_ready(limit=2)
    assert [job.name for job in jobs] == ["high", "mid"]
    assert jobs[0].status == "completed"
    assert jobs[0].result == 2


def test_scheduler_retries_and_reports_telemetry():
    scheduler = RuntimeScheduler(max_workers=2)
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("transient")
        return "ok"

    scheduler.submit("retry", flaky, priority=1, retries=1)
    scheduler.run_ready()
    telemetry = scheduler.telemetry()
    assert telemetry.completed == 1
    assert telemetry.failed == 0
    assert telemetry.success_rate == 1.0
    result = scheduler.store.snapshot()
    assert next(iter(result.values()))["result"] == "ok"


def test_production_runtime_persists_job_results():
    runtime = ProductionRuntime(max_workers=2)
    runtime.submit("seed", lambda: np.array([1.0, 2.0]), priority=2)
    runtime.submit("branch", lambda: {"score": 9}, priority=1)

    jobs = runtime.run_all()
    assert len(jobs) == 2
    data = runtime.snapshot()
    assert any(value["status"] == "completed" for value in data.values())
    assert any("result" in value for value in data.values())
    assert runtime.telemetry().total >= 2


def test_distributed_runtime_reports_worker_status():
    runtime = DistributedRuntime(worker_count=3)
    runtime.submit("one", lambda: 1, priority=2)
    runtime.submit("two", lambda: 2, priority=3)
    runtime.run_all()
    assert runtime.status()["workers"] == 3
    assert runtime.status()["completed"] >= 2


def test_scheduler_submit_rejects_invalid_job_definitions():
    scheduler = RuntimeScheduler()
    with pytest.raises(ValueError):
        scheduler.submit("", lambda: None)
    with pytest.raises(TypeError):
        scheduler.submit("job", "not-callable")
    with pytest.raises(ValueError):
        scheduler.submit("job", lambda: None, retries=-1)
    with pytest.raises(ValueError):
        scheduler.submit("job", lambda: None, metadata={"bad": object()})


def test_api_key_guard_authorizes_known_keys_only():
    guard = ApiKeyGuard(["secret-key"])
    guard.authorize("secret-key")
    with pytest.raises(RuntimeAccessError):
        guard.authorize("wrong-key")
    with pytest.raises(RuntimeAccessError):
        guard.authorize(None)

    guard.add_key("second-key")
    guard.authorize("second-key")
    guard.revoke_key("second-key")
    with pytest.raises(RuntimeAccessError):
        guard.authorize("second-key")


def test_production_runtime_enforces_access_guard_when_configured():
    guard = ApiKeyGuard(["worker-key"])
    runtime = ProductionRuntime(access_guard=guard)

    with pytest.raises(RuntimeAccessError):
        runtime.submit("job", lambda: 1, api_key="wrong")

    job = runtime.submit("job", lambda: 1, api_key="worker-key")
    assert job.name == "job"

    with pytest.raises(RuntimeAccessError):
        runtime.run_all(api_key="wrong")

    jobs = runtime.run_all(api_key="worker-key")
    assert len(jobs) == 1


def test_production_runtime_without_guard_ignores_api_key():
    runtime = ProductionRuntime()
    runtime.submit("job", lambda: 1)
    jobs = runtime.run_all()
    assert len(jobs) == 1


def test_runtime_store_and_file_store_cover_clear_and_load_paths(tmp_path):
    store = RuntimeStore({"alpha": {"status": "queued"}})
    store.clear()
    assert store.snapshot() == {}

    path = tmp_path / "runtime.json"
    file_store = FileRuntimeStore(path, initial={"job-1": {"status": "queued"}})
    assert file_store.save()["job-1"]["status"] == "queued"
    file_store.put("job-2", {"status": "done"})
    reloaded = FileRuntimeStore(path)
    assert reloaded.load()["job-1"]["status"] == "queued"
    assert reloaded.get("missing", {"status": "none"}) == {"status": "none"}

    empty_path = tmp_path / "missing.json"
    assert FileRuntimeStore(empty_path).load() == {}


def test_sqlite_runtime_store_supports_initial_data_defaults_and_clear(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite", initial={"seed": {"status": "queued"}})
    assert store.get("seed") == {"status": "queued"}
    assert store.get("missing", {"status": "none"}) == {"status": "none"}
    store.clear()
    assert store.snapshot() == {}
    store.close()


def test_postgres_runtime_store_connects_via_optional_dependency(monkeypatch):
    fake_conn = _FakePostgresConnection()

    class _FakePsycopg2:
        @staticmethod
        def connect(dsn, **kwargs):
            assert dsn == "postgresql://example"
            return fake_conn

    monkeypatch.setattr(runtime_module, "_psycopg2_module", _FakePsycopg2())
    store = PostgresRuntimeStore(dsn="postgresql://example")
    try:
        assert store.get("missing", {"status": "none"}) == {"status": "none"}
        assert store.snapshot() == {}
    finally:
        store.close()
    assert fake_conn.closed is True


def test_redis_runtime_store_connects_via_optional_dependency(monkeypatch):
    fake_client = _FakeRedisClient()

    class _FakeRedisModule:
        class Redis:
            @staticmethod
            def from_url(url, decode_responses, **kwargs):
                assert url == "redis://example"
                assert decode_responses is True
                return fake_client

    monkeypatch.setattr(runtime_module, "_redis_module", _FakeRedisModule())
    store = RedisRuntimeStore(url="redis://example")
    store.put("job-1", {"status": "queued"})
    assert store.snapshot()["job-1"]["status"] == "queued"


def test_scheduler_reports_failures_and_validates_metadata_and_limits():
    scheduler = RuntimeScheduler(max_workers=2)
    assert scheduler.queue == []

    with pytest.raises(ValueError, match="metadata"):
        scheduler.submit("job", lambda: None, metadata="bad")  # type: ignore[arg-type]

    attempts = {"count": 0}

    def always_fails():
        attempts["count"] += 1
        raise ValueError("boom")

    job = scheduler.submit("boom", always_fails, retries=1)
    assert scheduler.run_ready(limit=0) == []
    completed = scheduler.run_ready(limit=1)
    assert completed == [job]
    snapshot = scheduler.store.snapshot()[job.job_id]
    assert snapshot["status"] == "failed"
    assert snapshot["attempts"] == 2
    assert snapshot["error"] == "ValueError: boom"

    telemetry = scheduler.telemetry()
    assert telemetry.failed == 1
    assert telemetry.success_rate == 0.0

    with pytest.raises(ValueError, match="limit"):
        scheduler.run_ready(limit=-1)


def test_runtime_job_execute_failure_sets_status_and_error():
    job = runtime_module.RuntimeJob(name="bad", handler=lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    with pytest.raises(RuntimeError, match="nope"):
        job.execute()
    assert job.status == "failed"
    assert job.error == "nope"


def test_api_key_guard_and_runtime_run_ready_cover_access_paths():
    guard = ApiKeyGuard()
    with pytest.raises(ValueError, match="api_key"):
        guard.add_key("")

    guard.add_key("secret")
    runtime = ProductionRuntime(access_guard=guard)
    runtime.submit("job", lambda: 1, api_key="secret")
    assert [job.result for job in runtime.run_ready(api_key="secret")] == [1]
    assert runtime.snapshot()

    distributed = DistributedRuntime(worker_count=4)
    assert distributed.status()["workers"] == 4
