# H¹¹ Quantum Evolutionary Space (QES)

[![CI](https://github.com/haseebcm/Quantum-Evolutionary-Space/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/haseebcm/Quantum-Evolutionary-Space/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A Python framework for governed search, bounded optimization, and durable task execution.**

QES explores candidate states inside explicit bounds, evaluates their admissibility,
tracks their evolution, and returns validated results. A candidate is represented
as a **room**: its state, reference target, constraints, lineage, and memory travel
together through the computation. You supply the problem and objective; QES
provides the population engine, permission gates, search SDK, and worker runtime.

The framework applies to design-space exploration, control-policy search,
parameter fitting, simulation, and resource allocation. Its quantum-inspired
research modules use classical mathematical models; quantum hardware is not
required for the supported core.

**Current release:** [QES 2.2.0](https://github.com/haseebcm/Quantum-Evolutionary-Space/releases/tag/v2.2.0) ·
**Author:** Mohamed Haseeb C M · **License:** [MIT](LICENSE)

## Supported production scope

Version 2.2.0 supports bounded trusted computation and registered workers, with
explicit contracts for failures, resource limits, delivery, and recovery.

| Capability | Supported scope |
|---|---|
| Classical search | `QESClient`, `quick_search`, and `optimize` with finite bounded states and trusted callbacks |
| Admissible results | Exact returned states pass non-mutating gate inspection; absent feasible results are explicit |
| Execution controls | Counted objective evaluations, cooperative deadlines/cancellation, and stopping reasons |
| Single-host workers | Durable SQLite intake, atomic leases, retries, stale-result fencing, and persistent local state |
| Network workers | PostgreSQL-authoritative task intake and coordination across independent Linux workers over TCP |
| Result storage | In-memory/local stores and real-service-tested Redis 7 / PostgreSQL 16 adapters |
| Operations | Worker heartbeats, queue health, JSON/Prometheus metrics, alerts, and shutdown/recovery checks |
| Packaging | Typed Python package, wheel/source downloads, and Python 3.10–3.12 CI |

GPU acceleration, advanced `qes.quantum_compute` primitives, peer-election
coordination, and S3 integration remain experimental. Architectural/research
modules are available for exploration; their presence does not extend the
production support contract. See [readiness and compatibility](docs/production-readiness.md).

## Install

Install the versioned GitHub release directly; this does not depend on PyPI
publication status:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
python -m pip install "qes @ git+https://github.com/haseebcm/Quantum-Evolutionary-Space.git@v2.2.0"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.
Alternatively, download the wheel from the [release assets](https://github.com/haseebcm/Quantum-Evolutionary-Space/releases/tag/v2.2.0)
and install it with `python -m pip install /path/to/qes-2.2.0-py3-none-any.whl`.
The core runtime dependency is NumPy. For PostgreSQL workers/result storage:

```bash
python -m pip install "qes[postgres] @ git+https://github.com/haseebcm/Quantum-Evolutionary-Space.git@v2.2.0"
```

Use `qes[redis,postgres]` in the same command to install both optional adapters.

## Run a bounded search

```python
from qes import quick_search

result = quick_search(
    bounds=([-5.0, -5.0], [5.0, 5.0]),
    objective=lambda x: float(x @ x),
    population=20,
    steps=50,
    max_evaluations=1000,
    max_wall_time=5.0,
    rng=42,
)

print(result.status, result.stopping_reason, result.evaluations)
if result.status == "success":
    print("State:", result.best_state)
    print("Objective:", result.best_score)
```

Objectives are minimized. `success` means a returned candidate is admissible;
it does not prove a global optimum. `no_feasible_solution` has no winning state
or score. The lower-level `optimize()` API raises `NoFeasibleSolutionError` when
it cannot return an admissible candidate.

Every objective call, including initialization and finite-difference probes,
counts against the evaluation cap. Deadlines and cancellation refuse new work;
they cannot interrupt a callback already running. SDK evaluation caps apply
across repeated runs of the same client. Survivor selection is opt-in, and
entropy/concentration alone does not establish objective convergence.

## Run durable workers

### Local SQLite queue

Submit a data-only registered task:

```python
from qes import SQLiteTaskQueue

queue = SQLiteTaskQueue("state/tasks.sqlite")
task_id = queue.submit(
    "qes.search",
    {"lower": [-5, -5], "upper": [5, 5], "objective": "sphere",
     "max_evaluations": 200, "seed": 7},
    idempotency_key="search-request-001",
)
print(task_id)
queue.close()
```

Start the worker, then inspect health from another terminal:

```bash
qes-worker --task-queue state/tasks.sqlite --interval 1
qes-status --task-queue state/tasks.sqlite --require-worker
qes-status --task-queue state/tasks.sqlite --format prometheus
```

Read the result with `queue.get(task_id)`. The built-in `qes.search` handler
accepts sphere/rastrigin objectives and bounded configuration, not executable
code from task input. Multiple local processes can share one SQLite file on a
local filesystem. See [deployment, limits, and recovery](docs/deployment.md).

### PostgreSQL network queue

Configure `QES_POSTGRES_DSN` through your deployment secret manager. Production
connections should use authenticated TLS and a private network.

```python
import os
from qes import PostgresTaskQueue

queue = PostgresTaskQueue(os.environ["QES_POSTGRES_DSN"])
task_id = queue.submit(
    "qes.search", {"max_evaluations": 200, "seed": 7},
    idempotency_key="network-search-001",
)
print(task_id)
queue.close()
```

On each worker host, with that environment configured:

```bash
qes-worker --interval 1 --lease-seconds 300
qes-status --require-worker
```

PostgreSQL controls atomic claims and lease time. Completion checks the current
owner, tenant, token, and lease expiry, including after row-lock waits. Workers
need no shared filesystem. Database outages stop claims; supervised workers
recover after service availability returns. See the
[network coordination contract](docs/multi-host-coordination.md).

**Delivery is at least once.** Stable task IDs support downstream idempotency;
fencing protects queue results but cannot undo external effects. Database HA,
replication/failover policy, secret management, and application authorization
remain deployment responsibilities.

### Containers

```bash
git clone https://github.com/haseebcm/Quantum-Evolutionary-Space.git
cd Quantum-Evolutionary-Space
git checkout v2.2.0
docker compose up --build -d
```

The reference Compose deployment uses non-root workers, persistent local SQLite
state, CPU/memory limits, a read-only root filesystem, health checks, and a
shutdown grace period. Its named volume is separate from a host-side `state/`
folder. PostgreSQL workers use the same image with a DSN and an overridden
worker command; follow the [deployment instructions](docs/multi-host-coordination.md).

## Architecture

The core lifecycle generates candidate rooms, executes state transitions,
measures divergence, applies permission gates, and records population telemetry.
Selection can retain survivors by a caller-defined signature. `World`, `Agent`,
and `Universe` compose these populations into larger models.

| Layer | Main APIs/modules | Purpose |
|---|---|---|
| Problem representation | `Room`, `MCCStateSpace`, `qes.domain` | States, targets, bounds, activation, and constraints |
| Population computation | `QESSpace`, `RealityGenerator`, `qes.dynamics` | Branch and evolve candidate populations |
| Governance | `GenesisPermission`, `AdaptivePermission`, `qes.selection` | Inspect admissibility and select survivors |
| Search | `QESClient`, `quick_search`, `optimize` | Bounded optimization with result and budget contracts |
| Composition | `World`, `Universe`, `Agent` | Model environments, agents, and nested spaces |
| Durable execution | `SQLiteTaskQueue`, `PostgresTaskQueue`, `RegisteredTaskWorker` | Intake, claims, leases, fencing, and registered dispatch |
| Operations | `qes.worker`, `qes.monitor`, `qes.runtime` | Worker lifecycle, health/alerts, and result persistence |
| Research extensions | Equation ASTs, causal models, digital twins, QSEE, quantum-inspired evolution | Explore additional domain and architecture models |

The [H¹¹ architecture specification](docs/QES-architecture.md) describes the
mathematical model. Architecture terminology is not a guarantee of physical
quantum execution, cloud provisioning, OS isolation, or formal safety certification.

## Validation and current status

The 2.2.0 release passed **1,502 tests** with one skipped, plus all hosted release
gates. CI includes:

- Python 3.10–3.12 tests, lint, and type checks.
- Installed-wheel validation and all 41 shipped examples.
- The committed performance regression gate.
- Native and constrained non-root container load, shutdown, kill/restart, alerts,
  and backup restoration.
- Real Redis/PostgreSQL concurrent clients, outages, reconnects, persistence
  failures without successful-handler replay, and backup restores.
- Networked worker load, disconnect/rejoin, stale-write fencing, clock skew,
  lease renewal, capacity races, attempt exhaustion, database restart, and full
  PostgreSQL queue restoration.

Operational network tests use independently networked containers on one CI host.
Physical-host/region behavior and your database provider's HA configuration need
validation in the target deployment. Test counts and historical benchmark data
are evidence for the tested workloads, not an application SLA or proof that
QES outperforms every alternative optimizer.

See [GitHub Actions](https://github.com/haseebcm/Quantum-Evolutionary-Space/actions/workflows/ci.yml),
[validation evidence](docs/readiness-validation.md), and the per-run operational
JSON artifacts. To reproduce from a checkout:

```bash
python -m pip install -e ".[dev,redis,postgres]"
python -m ruff check src tests examples
python -m mypy
python -m pytest -q --cov=qes
python benchmarks/run_benchmarks.py --repeat 5 --baseline benchmarks/qes_benchmarks.json --fail-on-regression
```

The Docker-backed acceptance scenarios are documented in the deployment,
[result-store](docs/remote-stores.md), and network-coordination guides.

## Examples and documentation

| Start here | Demonstrates |
|---|---|
| [SDK example](examples/sdk_demo.py) | `QESClient` and `quick_search` |
| [Durable queue example](examples/durable_queue_demo.py) | Submission, bounded search, result inspection, and backup |
| [Core loop](examples/basic_run.py) | Rooms, generation, permission gates, and telemetry |
| [Control-policy search](examples/control_policy_search.py) | A bounded control problem |
| [Design-space search](examples/design_space_search.py) | Candidate design exploration |
| [Scientific inference](examples/scientific_inference.py) | Parameter fitting from observations |
| [Domain packs](examples/domain_packs_demo.py) | Reusable domain setup |

Browse the [complete example index](docs/examples-index.md),
[API reference](docs/api-reference.md), [tutorials](docs/tutorials.md), and
[documentation index](docs/index.md). Quantum serialization migrations and other
compatibility changes are recorded in [production-readiness.md](docs/production-readiness.md)
and the [changelog](CHANGELOG.md). Binary serialization uses compressed data-only
JSON; legacy pickle payloads are rejected.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and
[RELEASING.md](RELEASING.md) for release procedures.

Released under the [MIT License](LICENSE), © 2026 Mohamed Haseeb C M.
