# Version 2.0.0 validation evidence

Local validation for the `2.0.0` production-readiness branch, 2026-10-09:

| Gate | Result |
|---|---|
| Python 3.12 source suite | 1,493 passed, 1 skipped |
| Branch coverage run | 91% combined statement/branch coverage |
| Installed wheel suite outside checkout | 1,493 passed, 1 skipped |
| Ruff | Passed |
| Mypy | Passed, 79 source files |
| Shipped examples | All 41 passed individually |
| Committed benchmark regression gate | Passed, five repetitions |
| Package version and typing marker | Installed metadata, main and quantum versions agree; `py.typed` present |
| Durable intake | Reopen, idempotent submission, independent connection/process claims tested |
| Recovery | Expired lease reclaim, stale fencing, attempt cap and backup restore tested |
| Shutdown | Idle registered worker exits successfully on SIGTERM |
| RPC dispatch | Socket tests cover missing/wrong scope, tenant denial and duplicate requests |

One test emits a runpy warning because it executes an already imported worker
module; this is recorded rather than suppressed. The skip is retained from the
existing suite. The previous release candidate passed hosted Python 3.10–3.12, examples, wheel
and benchmark checks. This release adds monitoring, worker liveness and operational
failure scenarios. Release 2.0.0 is tagged only after the new main CI, including
actual constrained non-root container operations, passes. Local measurements are
in [single-host-native.json](validation/single-host-native.json); hosted native and
container JSON reports are retained as CI artifacts.

The native scenario processed 60 mixed 2/8/16-dimensional searches with two
workers, each capped at 200 objective evaluations. End-to-end throughput was
77 tasks/second, p95 completion latency 0.76 seconds and maximum recorded worker
RSS 65,920 KiB. These include process startup and are one synthetic sample, not
an SLA. Shutdown, kill/reclaim, attempt caps, operational alerts and restoration
passed. Separate process tests validate in-flight SIGTERM drain and idempotent
replay after an effect was committed before a crash.

This does not establish real GPU, remote Redis/Postgres/S3, multi-host partitions or every experimental quantum primitive. See
[supported contracts](production-readiness.md) and [deployment](deployment.md).
