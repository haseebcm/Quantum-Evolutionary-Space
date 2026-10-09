# Trusted single-host deployment and recovery

Version 2.1.0 supports trusted classical computation and bounded registered
single-host tasks. This is a scoped production release; the experimental API
families listed below are not promoted by the version number.

## Supported operating envelope

Use the classical `QESClient`/`optimize` APIs with finite bounded states and
trusted callbacks. The container runs registered data-only `qes.search` tasks
through `SQLiteTaskQueue`. Multiple worker processes can share one SQLite file
on a local filesystem on the same host. Do not share it through a network mount.
Advanced quantum, GPU and multi-host leadership are experimental. Redis and
PostgreSQL result adapters have separate [supported contracts](remote-stores.md).
Container builds and remote integrations must be validated in your deployment.

## Submit and run

Install `pip install .` from the reviewed commit. Submission commits before it
returns, and the stable task ID identifies the operation across lease recovery:

```python
from qes import SQLiteTaskQueue

queue = SQLiteTaskQueue("state/tasks.sqlite")
task_id = queue.submit(
    "qes.search",
    {"lower": [-5, -5], "upper": [5, 5], "objective": "sphere",
     "population": 20, "steps": 50, "max_evaluations": 1000, "seed": 42},
    tenant_id="local", idempotency_key="search-2026-001",
)
print(task_id)
queue.close()
```

```bash
qes-worker --task-queue state/tasks.sqlite --interval 1 --lease-seconds 300
# Or use the persistent local named volume and non-root container:
docker compose up --build -d
```

Host submission and container submission must address the same queue file.
The default Compose named volume is container-managed; submit from an attached
container, or configure an appropriately owned local bind mount. Do not confuse
a host `state/tasks.sqlite` with the file inside the named volume.

The built-in task accepts only sphere/rastrigin objectives, 1–128 dimensions,
1–256 rooms, 1–1,000 steps and 1–10,000 objective evaluations. Its deadline is
30 seconds and cooperative. Read the complete record using
`queue.get(task_id, tenant_id="local")`; inspect status, attempts, result and
error. Result metadata includes feasibility status and stopping reason.

SIGTERM/SIGINT stop new claims and let the current trusted handler finish.
Configure an orchestrator shutdown grace period longer than the expected task
duration (for example 60 seconds for these bounded built-in workloads). A forced
kill leaves a running task recoverable after its lease expires. Choose leases
longer than expected callback execution, or explicitly renew long-running tasks.

## Delivery, capacity and backup

Delivery is **at least once**. A crash after an external effect but before saving
completion may repeat that effect. Registered custom handlers receive
`(payload, task_id)`; use the task ID as the downstream idempotency key or an
application transaction/outbox. Fencing blocks stale workers from writing queue
results; it cannot undo downstream effects. Unknown task names fail rather than
importing or evaluating code from input.

The queue limits retained rows and serialized payload/result size. Completed
and failed rows occupy capacity. Purge terminal records only after retention,
audit and idempotency requirements are satisfied; after purging, the submission
key no longer deduplicates. Disk quotas, CPU/memory limits and supervision belong
to the operator; configure them for the measured workload.

Create a consistent live backup with `queue.backup("backups/tasks.sqlite")`.
Do not copy only the live main file while WAL writes are occurring. To restore,
stop all workers, retain the original files, restore the backup to a new path,
open it with `SQLiteTaskQueue`, verify representative records, then start workers
against that path. Expired running leases are reclaimed up to the attempt cap.
Tests exercise reopen, backup/restore, independent process claims and stale writes.

## Local RPC and promotion gates

RPC servers bind only to loopback. Configure `token_issuer` to require signed
tokens with `rpc:<method>` scopes. Supplied `tenant_id` must match the subject or
an explicit `tenant:<id>` scope. Handlers must validate authorization for their
own application objects and require tenant identity where applicable. Omitting
`token_issuer` is trusted local mode. Never expose that mode through a proxy.
External access requires an authenticated TLS gateway and managed signing keys.

RPC retries reuse request IDs. A bounded process-local cache deduplicates matching
requests while retained; it does not promise deduplication across restarts or
eviction. Durable task IDs and idempotent handlers remain the recovery mechanism.
Lowest-ID election is not partition-safe consensus.

## Monitoring and alert response

```bash
qes-status --task-queue state/tasks.sqlite --require-worker
qes-status --task-queue state/tasks.sqlite --format prometheus
```

Exit status 0 means healthy, 1 means one or more operational alerts, and 2 means
the monitor could not inspect the queue or its configuration is invalid. It does
not create a missing queue. Run with the state-directory owner's permissions.
The monitor exports no task payloads and provides no public network endpoint.
Send JSON output to your log/alert collector, or atomically write the Prometheus
output to a textfile collector. Failed tasks remain alerted until investigated
and deliberately purged. A healthy exit does not establish objective optimality.

| Alert | Action |
|---|---|
| `no_live_workers` | Check process/container supervision and worker startup logs; restore workers |
| `expired_leases` | Check killed/stalled workers; permit fenced reclaim; investigate repeated expiry |
| `queue_age` | Investigate slow handlers or insufficient capacity; scale workers on the same host |
| `queue_capacity` | Investigate backlog and terminal retention; back up and purge eligible records |
| `failed_tasks` | Inspect stored errors and attempt counts; correct configuration/handler failures |
| `disk_space` | Restore free space; check WAL, backups and retention before accepting more work |

Defaults alert on queue age above 60 seconds, 90% retained record capacity and
less than 100 MiB free disk. Worker heartbeats expire after 60 seconds. They are
refreshed during polling and between trusted callbacks, not inside arbitrary
callbacks. Choose timeout and lease values to fit the maximum trusted callback
duration. Peak worker RSS is reported in KiB on Unix; zero means unavailable on
Windows. Configure `--max-records` consistently with the queue's intake limit.

The reference container has a health check, 1 CPU, 256 MiB memory, a read-only
root filesystem, writable local state and a 60-second shutdown grace period.
Docker health status does not automatically restart an unhealthy running
container: connect your supervisor or alert collector to that status. The
standard restart policy applies when a worker exits.

Worker messages include `worker_started`, `task_processed` with duration and
`worker_stopped`. Task/worker counts, oldest queued age, expired leases, retry
attempts, retained capacity, database bytes, disk space and peak RSS are exposed
by the monitor. Tenant-specific payload errors remain in the local protected DB.

## Repeatable acceptance workloads

```bash
python benchmarks/validate_single_host.py --tasks 60 --json native-operations.json
docker build -t qes-acceptance .
python benchmarks/validate_single_host.py --docker-image qes-acceptance --tasks 40 --json container-operations.json
```

The scenarios vary sphere/rastrigin objectives over 2, 8 and 16 dimensions,
12-room populations and 200 objective evaluations, using two workers. They
measure end-to-end throughput/latency, memory and local state size, exercise
SIGTERM drain and SIGKILL lease recovery, inject a failure to verify the attempt
cap and alert, and restore an online backup. The final failed record is an
intentional fault injection. A separate process test crashes after an external
effect and verifies replay using the stable task ID prevents duplicate effects.
These synthetic results are a reproducible starting envelope, not an application
SLA or evidence for arbitrary callbacks, much larger dimensions, network storage
or more hosts. Load-test your actual workload before changing limits.

Before releasing, require Python 3.10–3.12 CI, lint/types, full regression tests,
installed-wheel smoke/tests, all examples and the committed benchmark gate.
Before operating, measure workload latency, task failure/attempt rates, queue
capacity, disk usage and shutdown/recovery behavior. Promote experimental
integrations only after testing the actual hardware, service and failure modes.
