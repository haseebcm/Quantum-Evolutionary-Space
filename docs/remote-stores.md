# Redis and PostgreSQL result stores — 2.1.0

The supported adapters share **job results**, not durable task intake or
multi-host worker claims. The registered durable worker still uses the local
SQLite queue. Result-store promotion does not certify multi-host orchestration,
consensus, GPUs or the advanced quantum API.

Install `pip install 'qes[redis,postgres]'` from the 2.1.0 distribution. Supply
connection URLs/DSNs through deployment secrets, never committed source. Use
TLS and private network access for remote services. Give PostgreSQL the table
creation/CRUD permissions required by the adapter, and restrict Redis credentials
to the configured hash namespace. The plain loopback connections used in CI are
disposable fixtures, not deployment security configuration.

Redis creates a connection pool with bounded socket and connect waits (5 seconds
by default, configurable with `socket_timeout`). It reconnects using the client
pool. PostgreSQL-owned connections use a 3-second connect timeout and a 5-second
statement timeout; a connection closed by a disconnect is recreated on the next
operation. Injected connections stay caller-managed for recovery/configuration.
`close()` terminates adapter use and closes the supplied client/connection; do
not share that handle with consumers that outlive the adapter.

Operations surface connection, serialization and persistence errors. There is
no hidden transaction replay after an uncertain commit. Runtime persistence
retry is separate from handler retry and never reruns a completed handler in
the same process. Recover a retained `persist_failed` result by writing it under
the same job ID after service recovery. Repeated puts are replacement/upsert
operations. A crash after an external effect still needs downstream idempotency.

Both adapters reject non-finite JSON values. Redis snapshots decode byte keys
when an injected client does not enable response decoding. PostgreSQL serializes
transactions on its connection and rolls back failed transactions; a rollback
failure does not mask the original outage/commit exception.

## Durability and operations

For Redis durability, configure AOF and an appropriate fsync policy, persistent
storage, a memory limit and an explicit eviction policy. The acceptance fixture
uses `appendonly yes`, `appendfsync always`; arbitrary server defaults do not
provide equivalent guarantees. Retain consistent RDB/AOF backups and test the
exact restore mode. Do not combine an old RDB with a newer authoritative AOF and
expect the RDB to take precedence.

PostgreSQL durability depends on WAL, fsync/synchronous-commit configuration,
persistent volumes and your replication/failover policy. Validate `pg_dump` or
your managed backup service and regularly restore into an isolated database.
Neither adapter supplies database administration, automatic failover or quotas.
Monitor service health, connection errors, persistence failures, storage growth,
memory, backup age and restore success. Archive/retain results deliberately;
`clear()` deletes the configured result namespace/table contents.

## Acceptance gate

```bash
python benchmarks/validate_remote_stores.py --json remote-stores.json
```

This requires Docker and creates isolated Redis 7 / PostgreSQL 16 services,
loopback ports and disposable persistent volumes. It validates three independent
writer processes, four threads sharing an adapter, acknowledged-write survival
after SIGKILL, bounded offline failure, client recovery, no successful-handler
replay on persistence failure, PostgreSQL dump restore and Redis RDB restore.
The volumes are deleted in cleanup. No production database is addressed.

Main and release CI run this gate against actual services and retain its JSON
report. Publication requires all existing core/container gates plus this job.
Test your hosted service configuration separately before relying on its security,
replication, retention or failover behavior.
