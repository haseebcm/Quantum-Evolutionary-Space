# Correctness fixes, compatibility and supported scope

This release candidate hardens the trusted classical SDK and single-host task
worker. It includes reproducible regressions, data-only durable intake, execution
budgets, local RPC authorization, resource limits and a deployment runbook.
GPU, external stores, multi-host consensus and advanced quantum APIs remain
experimental. Release tagging requires passing the CI matrix and review.

## Result and metric contracts

- `QESClient` returns only active candidates whose exact returned state passes
  non-mutating gate inspection. Historical optimizer states are revalidated.
  `SDKRunResult.status` is `success` or `no_feasible_solution`; the latter has
  no winning state/score and stores no winning pattern.
- `score_fn` and `objective` are minimized. Permission is feasibility evidence,
  not objective utility. The default `QESSpace.step()` does not rank/prune
  admissible rooms; explicit signature/survivor selection remains opt-in.
- `optimize()` raises `qes.intelligence.NoFeasibleSolutionError` if no candidate
  passes admission. A zero-population run may evaluate an admissible seed.
- Entropy normalizes finite nonnegative weights. Empty/all-zero populations
  have concentration zero. Equal positive weights have concentration zero for
  more than one room. Concentration does not establish geometric convergence
  or an objective optimum. Negative/non-finite weights now raise `ValueError`.
- The SDK and optimizer clip initial search populations to their bounds.
  Finite-difference probes stay within bounds; near a boundary they use the
  available secant interval. Adam beta values must be strictly less than one.

## Copying and checkpoint contracts

`Room.clone()` deep-copies supported mutable metadata. `QESSpace.clone()`
deep-copies its object graph, isolating cloneable strategies, gates and random
generators. In-memory snapshots include gate, domain, strategy and time-step
state. Restore invalidates active membership caches and remains compatible with
older snapshots that did not include these fields.

Python functions, closures and external resources are not automatically made
independent by `deepcopy`. Use stateless callbacks or stateful callable objects
with a correct copy protocol. External side effects are not rolled back.
Snapshots of arbitrary Python callables are in-memory checkpoints, not a
portable data-only durable checkpoint format. Non-copyable resource-owning
callbacks need a caller-supplied state protocol before isolated replay can be
promised. `Layer.checkpoint()` also retains the qubit vectors and nested history.

Active-room list results are copies of the membership list, not copies of the
rooms. Change lifecycle through the space's supported operations; direct edits
to lifecycle fields can invalidate cached membership. Duplicate IDs are rejected.

## Job retries and persistence

Execution retries apply to failed handlers. Saving a completed result is retried
separately and never replays that successful handler in the same process.
Permanent store failure raises the store exception, sets `job.status` to
`persist_failed` and retains `job.result` for inspection/recovery. Running-job
telemetry is tracked; generated job IDs use UUIDs to avoid collisions on restart.

This does not provide exactly-once external effects. A crash after an effect but
before persistence still requires idempotent handlers or an application-specific
transaction/outbox protocol. The legacy scheduler's pending queue remains in memory. Use `SQLiteTaskQueue`
for durable intake and atomic single-host claims; result stores alone do not
provide these guarantees. See [deployment and recovery](deployment.md).
Only trusted registered handlers should be submitted.

## Budgets, validation and tick failure policy

`QESClient` and `optimize` accept `max_evaluations` and `max_wall_time`.
Every objective invocation, including initialization and finite-difference probes,
counts before execution. SDK caps apply across repeated runs on one client.
Results report `evaluations` and `stopping_reason`; `cancel()` refuses new work.
Deadlines are cooperative and cannot interrupt an already-running callback.
A zero budget can produce no feasible solution without invoking the objective.

Room execution validates finite states, ordered matching bounds and couplings.
A tick stages all callback vectors before committing any vector writes. Callback
memory mutations and external effects use an explicit best-effort contract and
are not rolled back. Adaptive population evaluation inspects the whole batch at
one threshold and adapts once, avoiding admission changes caused by room order.

Default limits include 10,000 rooms, 1,000 space history entries, 256 agent inbox
messages, 10,000 events, 1,000 in-memory pending jobs, 10,000 durable records and
1 MiB serialized task payload/result. History evicts oldest entries; intake,
inboxes and event logs reject capacity overflow explicitly. Durable terminal
records count toward capacity until deliberately purged.

## Experimental quantum API and migration

The complete quantum API remains experimental. The new regression suite covers
physical-state diagnostics, selected circuit operations and serialization;
many evolution/synchronization primitives still need independent contract tests.

- `ComputationalQubit.primary` remains an arbitrary computational vector.
  Physical diagnostics normalize a finite nonzero vector without modifying it.
  Invalid/zero vectors raise `ValueError`; pure-state construction from a density
  matrix rejects mixed/nonphysical matrices rather than silently approximating.
- Built-in `GateOp` records store local matrices and target indices. Custom
  records with `targets=None` retain the full-matrix execution path. Code that
  inspected built-in matrices as expanded full-system matrices must migrate.
  Circuit run inputs must be finite one-dimensional vectors of the right size.
- Circuit allocation checks limit one complex128 state vector to 64 MiB by
  default via `max_state_bytes`. This is not a total process-memory guarantee;
  simulation can allocate additional working copies. Custom operators and
  `max_state_bytes` must be trusted or additionally constrained by a service.
- Binary serialization version 2 is compressed data-only JSON. It rejects
  legacy pickle payloads. To migrate trusted existing pickles, decode them only
  in a separately controlled legacy environment and re-encode supported data;
  never load untrusted legacy files.
- JSON wraps data in a versioned envelope. Complex numbers are tagged and
  decoded; arrays become lists. Unsupported Python objects and non-finite JSON
  values are rejected. Reserved complex-tag dictionaries are escaped.
- Diff envelopes now use `_patch_version: 2`, `set` and `delete`, so setting a
  value to `null` differs from deleting it. Legacy flat patches are rejected.
- Compressed payloads are capped at 16 MiB after decompression; malformed,
  truncated and trailing compressed data are rejected. These helpers are not a
  complete authenticated storage or adversarial-input isolation system.

These changes need compatibility review before publishing a release. The
single-source version is `2.0.0rc1`, reflecting the serialization and validation
compatibility changes. This branch does not create a release tag or publish a
package.

## Validation gates

CI runs lint/types, tests, every shipped example, and built-wheel installation
tests. The wheel includes `py.typed`, and all package version accessors read
`qes._version`. The publish workflow checks its tag against that same source.

Before promoting a library API, require regressions for its critical behavior,
installed-distribution validation on supported Python versions, accurate docs,
and a defined contract for invalid input, failures, budgets and state copying.
Local Python 3.12 success does not replace the CI compatibility matrix.

For an operated service, additionally require durable task intake/atomic
leases, duplicate-effect handling, actual network authorization, limits and
deadlines, clean shutdown, backup restoration, resource retention and measured
operational envelopes. Test actual GPUs/external stores/multi-host failures
before supporting those integrations. Reachability-based lowest-ID election is
not partition-safe consensus. Configuration validation is not OS sandboxing.

## Remaining readiness backlog

The item IDs correspond to the production-readiness plan. “Partial” means the
implemented work does not satisfy the entire acceptance section.

| Item | Status in this branch | Remaining work |
|---|---|---|
| C01 admissible results | Implemented for SDK/default optimizer | Broader pattern reuse must validate against its consuming application |
| C02 entropy/concentration | Implemented | Domain-specific geometric/objective stopping metrics |
| C03 selection semantics | Implemented opt-in SDK survivor policy | Domain-specific signature policy |
| C04 restore/cache | Implemented | Direct lifecycle edits remain outside the supported cache contract |
| C05 branch isolation | Implemented for copyable object graphs | Portable callback/resource state protocol |
| C06 numerical boundaries | Core validation and batch adaptive gate implemented | Experimental wider API contracts |
| C07 callback failures | Vector writes staged until all callbacks validate | Callback memory and external effects remain caller-owned |
| O01 bounded gradients | Implemented basic bounded secants | Analytic-gradient support and higher-order boundary schemes |
| O02 hard budgets | Evaluation cap, cooperative deadline/cancel and stopping reasons implemented | Preemptive isolation for untrusted/hanging callbacks is outside supported scope |
| O03 performance evidence | Open | Held-out workloads, strong baselines, domain ablations |
| Q01 physical states | Partial | Broader multi-qubit and density-matrix API validation |
| Q02 local operators | Implemented for built-in circuit gates | Benchmarks and tests for the remaining circuit helpers |
| Q03 serialization | Implemented migration | Broader schema-depth/resource and application authentication requirements |
| Q04 quantum tests | Partial | Independent tests across all exported primitives |
| R01 storage retries | Implemented | Crash-safe application idempotency for external effects |
| R02 durable intake | Local SQLite queue, atomic leases, fencing and at-least-once contract implemented | Multi-host broker integration is experimental |
| R03 retention/limits | Bounded intake, payloads, rooms, histories, inboxes, events and RPC connections | Operator disk/container quotas and retention scheduling |
| R04 shutdown/telemetry | Registered worker SIGTERM, deadline, result metadata and recovery tested | Hosted metrics/alert routing belongs to deployment |
| R05 durable stores | PostgreSQL identifier validation/transaction locking; queue schema and backup restore tested | Actual remote Postgres/Redis integration validation |
| S01 dispatch authorization | Local RPC signed-token verification, method scopes and supplied tenant checks implemented | TLS gateway, key rotation, application object authorization |
| S02 execution isolation | Loopback network binding and bounded concurrency; data-only registered worker | OS isolation for untrusted callbacks is outside supported scope |
| S03 leadership/retries | Local queue fencing and bounded process-local RPC deduplication | Multi-host consensus and partition safety remain experimental |
| P01 packaging/version | Typed installed wheel and 2.0.0rc1 release candidate implemented | Passing hosted CI matrix and release approval |
| P02 CI semantics | Partial | Feature-specific coverage gates and actual integration jobs |
| P03 contracts/docs | Supported core/worker contract and deployment/recovery runbook implemented | Experimental APIs need independent promotion evidence |

Unimplemented items must not be described as completed production guarantees.
