# Trusted single-host deployment and recovery

## Supported operating envelope

Use the classical `QESClient`/`optimize` APIs with finite bounded states and
trusted callbacks. The container runs registered data-only `qes.search` tasks
through `SQLiteTaskQueue`. Multiple worker processes can share one SQLite file
on a local filesystem on the same host. Do not share it through a network mount.
Advanced quantum, GPU, remote stores and multi-host leadership are experimental.
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

Before releasing, require Python 3.10–3.12 CI, lint/types, full regression tests,
installed-wheel smoke/tests, all examples and the committed benchmark gate.
Before operating, measure workload latency, task failure/attempt rates, queue
capacity, disk usage and shutdown/recovery behavior. Promote experimental
integrations only after testing the actual hardware, service and failure modes.
