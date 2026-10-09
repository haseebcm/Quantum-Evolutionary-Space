# Supported network worker coordination — 2.2.0

PostgreSQL 16 is the single authority for durable task state. Independent Linux
workers connect over TCP and need no shared filesystem. The existing classical
SDK, single-host SQLite workers, Redis/PostgreSQL result stores, monitoring and
recovery contracts remain supported. Peer election and arbitrary distributed
callbacks are not promoted by this release.

## Deploy

Install the 2.2.0 distribution with the `postgres` extra on each worker. Provision
one authoritative writable PostgreSQL service with persistent WAL-backed storage.
Use a private network and authenticated TLS (`sslmode=verify-full`, a trusted CA
and correct server hostname) in production. Supply the DSN through your secret
manager/environment, not source files or command-line history. The worker role
needs queue-table creation and CRUD privileges; run a migration/bootstrap role
first if your policy separates DDL privileges. Namespace isolates table sets;
`tenant_id` scopes lookups/fencing, but direct database access is trusted. Enforce
application/tenant authorization before allowing submission or DB access.

```python
import os
from qes import PostgresTaskQueue

queue = PostgresTaskQueue(os.environ['QES_POSTGRES_DSN'])
task_id = queue.submit('qes.search', {'max_evaluations': 200, 'seed': 7},
                       idempotency_key='search-request-001', tenant_id='local')
print(task_id)
queue.close()
```

On each host, configure `QES_POSTGRES_DSN` and run:

```bash
qes-worker --interval 1 --lease-seconds 300
qes-status --require-worker
qes-status --format prometheus
```

Use `--queue-namespace` consistently for workers/monitoring and `namespace=` for
submission. Do not also select a SQLite path. For the container, override its
SQLite default command with `--interval 1` and inject `QES_POSTGRES_DSN`; retain
non-root execution, CPU/memory limits, private networking, and supervision.
The image health check selects the PostgreSQL backend when that variable is set.
PostgreSQL metrics do not report database-host disk space as if it were local;
monitor that service's disk, WAL, replication, backups and latency separately.

## Delivery and recovery

Claims lock one eligible row with `FOR UPDATE SKIP LOCKED`, update its token and
attempt count and commit atomically. Database time controls all lease comparisons;
host clock skew cannot extend a lease. Completion/renewal check task, tenant,
owner, current UUID token and unexpired database lease. No worker can publish a
stale queue result after another worker reclaims it. Lease expiry does not stop
an already-running callback or undo its external effects.

Delivery is **at least once**. A stable task ID is passed to registered trusted
handlers; use it as downstream idempotency key or transactional/outbox identity.
A successful-handler completion failure is surfaced, never immediately replayed
by the same dispatch. An ambiguous submit acknowledgment is recovered by
resubmitting identical content with the same tenant/idempotency key. Recovery
can repeat callbacks after lease expiry, so external idempotency remains mandatory.

The built-in trusted search task has its existing bounds/evaluation/deadline
limits. Its deadline is cooperative. Default leases are 300 seconds; choose a
lease longer than worst-case trusted callback duration, or renew explicitly.
Do not change limits without workload measurements. Intake capacity is atomic
across clients using the same namespace and consistent limits. Completed/failed
records occupy capacity until deliberately purged; purging also removes their
submission-key deduplication protection.

A database outage stops new claims. Errors are surfaced and the worker exits;
supervision restarts it after service recovery. Owned closed connections reconnect
on the next operation. Linux transport keepalives/TCP user timeout, connection,
statement and lock waits are configured to detect failures; they do not provide
an OS hard timeout for arbitrary callbacks. SIGTERM stops new claims and drains
current work. Forced worker loss leaves leases recoverable up to the attempt cap.
Database uncertainty never triggers peer election or independent local claims.

Back up all namespace tasks/workers/meta tables consistently with PostgreSQL
backup tooling. Restore into a new isolated database, verify records/schema,
then direct supervised workers to it. The schema version is checked before use;
unknown versions are rejected. PostgreSQL HA is operator-managed: prevent multiple
writable divergent primaries, select synchronous durability/failover settings to
meet your acknowledged-task RPO, and test your actual managed service. This
library does not promise zero data loss under asynchronous replica promotion.

## Validation evidence and limits

`benchmarks/validate_coordination.py` runs actual PostgreSQL and two independently
networked non-root worker containers with separate filesystems. It gates main and
release CI on load without duplicate claims, health/drain, durable submission
idempotency, conflicting-content rejection, worker disconnect/rejoin with stale
write rejection, database-clock skew, lease renewal, tenant fencing, concurrent
capacity limits, attempt exhaustion, database kill/restart and full queue backup
restoration. JSON evidence is retained as the coordination CI artifact.

These are network failure simulations on one CI Docker host, not measurements
from physical hosts, multiple regions, your database HA provider or an arbitrary
workload. Supported production scope is PostgreSQL-authoritative bounded trusted
workers over a private Linux network. Validate your deployment's throughput,
latency, failure domains, supervision, TLS, capacity and backup/failover policy
before promising an application SLA. GPU and advanced quantum APIs remain
experimental.

PostgreSQL's queue-lock and connection parameters are documented at
https://www.postgresql.org/docs/16/sql-select.html and
https://www.postgresql.org/docs/16/libpq-connect.html.
