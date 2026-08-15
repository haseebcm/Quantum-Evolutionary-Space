# QES deployment patterns

QES is a Python framework, but it can be deployed in several production-like patterns depending on the runtime model.

## 1. Local single-process runtime

Use this for development, local experimentation, and small supervised search loops.

Typical pattern:

- one machine
- one Python process
- one `Universe` with one or more `World`s
- one scheduling loop that advances the QES space

This is the simplest deployment and matches the repository examples most closely.

```python
from qes.universe import Universe
from qes.world import World

universe = Universe(dt=1.0)
world = World(name="local-world")
universe.add_world(world)

for _ in range(20):
    universe.step(event="tick")
```

## 2. Worker-based pipeline

Use this when each experiment or branch should run as an independent worker task.

Pattern:

- one orchestrator owns the master `Universe`
- worker processes create or evaluate `QESSpace` instances
- results are merged back into the central memory / pattern store

This is useful for:

- design-space search
- scenario simulation
- nested experiment generation
- branch evaluation before selection

Typical runtime composition:

```text
API / scheduler
    -> job queue
        -> worker 1: QESSpace search
        -> worker 2: QESSpace search
        -> worker N: QESSpace search
    -> result aggregator
    -> pattern memory update
```

## 3. Multi-world orchestration

Use this when different domains or scenarios need isolation.

Example:

- world 1 = structural design searches
- world 2 = control tuning
- world 3 = simulation and verification
- world 4 = nested experimental universe

Each world owns a different search task, while the top-level `Universe` governs policies, memory, and branching boundaries.

## 4. Event-driven digital-space service

For production-like use, wrap the runtime in a service boundary:

- receive a task or configuration
- materialize a `QESSpace`
- generate candidate rooms
- run a bounded search loop
- validate with closure checks
- emit the best candidate and metadata

This is usually the right fit for:

- design recommendation services
- autonomous policy tuning loops
- simulation orchestration engines
- digital-twin update pipelines

## 5. Containerized deployment pattern

The repository ships a real, working container setup rather than a hypothetical
sketch: `Dockerfile`, `docker-compose.yml`, and `src/qes/worker.py` (the
`python -m qes.worker` / `qes-worker` service entrypoint).

```bash
docker compose up --build
```

`docker-compose.yml` mounts a named volume at `/app/state` so the durable
SQLite-backed `ProductionRuntime`/`DistributedRuntime` job history survives
container restarts. Configure the worker via environment variables:

| Variable                      | Purpose                                      |
| ------------------------------ | --------------------------------------------- |
| `QES_RUNTIME_STATE_DIR`       | Directory for the durable SQLite store        |
| `QES_WORKER_CYCLES`           | Number of scheduling cycles (`0` = run forever) |
| `QES_WORKER_INTERVAL_SECONDS` | Delay between scheduling cycles               |
| `QES_LOG_LEVEL`               | Python logging level                          |

Replace `_demo_jobs()` in `src/qes/worker.py` with real job intake (an HTTP
endpoint, a message queue consumer, etc.) to move from the reference demo
workload to production traffic.

## 6. Durable, multi-node storage backends

`SQLiteRuntimeStore` is appropriate for a single node. For a horizontally
scaled worker pool spanning multiple containers/machines, use a shared
backend so every node observes the same job state:

- `RedisRuntimeStore` — low-latency shared job state (`pip install qes[redis]`)
- `PostgresRuntimeStore` — transactional durability with SQL query access to
  job history (`pip install qes[postgres]`)

Both implement the same `RuntimeStore` interface (`put`, `get`, `snapshot`,
`clear`), so they are drop-in replacements passed to `ProductionRuntime(store=...)`
or `DistributedRuntime(store=...)`.

## 7. Access control for exposed runtimes

If a `ProductionRuntime`/`DistributedRuntime` is reachable from outside its
own process (behind an HTTP wrapper, a shared queue consumer, etc.), pass an
`ApiKeyGuard` so `submit`/`run_ready`/`run_all` require a valid API key:

```python
from qes.runtime import ApiKeyGuard, ProductionRuntime

guard = ApiKeyGuard(["prod-key-1", "prod-key-2"])
runtime = ProductionRuntime(access_guard=guard)

runtime.submit("job", handler, api_key="prod-key-1")  # ok
runtime.submit("job", handler, api_key="wrong")  # raises RuntimeAccessError
```

Job submission also validates inputs eagerly (`name` non-empty, `handler`
callable, non-negative `retries`, JSON-serializable `metadata`) so malformed
payloads fail immediately at `submit()` rather than deep inside a worker
thread.

For a full Kubernetes deployment (manifests, scaling guidance, autoscaling
notes), see [`kubernetes-deployment.md`](kubernetes-deployment.md).

## Recommended production pattern

For most serious deployments, combine:

- a top-level `Universe` orchestrator
- one or more `World` partitions per domain
- one `QESSpace` per active search campaign
- `Agent`s for monitoring and adaptation
- a shared durable store (`RedisRuntimeStore`/`PostgresRuntimeStore`) for
  multi-node job history, or `SQLiteRuntimeStore` for a single node
- an `ApiKeyGuard` if the runtime is reachable from outside its own process
- validation / permission / closure gates before accepting outputs

This pattern preserves the framework’s main value: governed search in a digital universe rather than ad hoc script execution.
