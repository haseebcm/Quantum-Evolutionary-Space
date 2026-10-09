# Release candidate validation evidence

Local validation for the `2.0.0rc1` production-readiness branch, 2026-10-09:

| Gate | Result |
|---|---|
| Python 3.12 source suite | 1,487 passed, 1 skipped |
| Branch coverage run | 91% combined statement/branch coverage |
| Installed wheel suite outside checkout | 1,487 passed, 1 skipped |
| Ruff | Passed |
| Mypy | Passed, 78 source files |
| Shipped examples | All 41 passed individually |
| Committed benchmark regression gate | Passed, five repetitions |
| Package version and typing marker | Installed metadata, main and quantum versions agree; `py.typed` present |
| Durable intake | Reopen, idempotent submission, independent connection/process claims tested |
| Recovery | Expired lease reclaim, stale fencing, attempt cap and backup restore tested |
| Shutdown | Idle registered worker exits successfully on SIGTERM |
| RPC dispatch | Socket tests cover missing/wrong scope, tenant denial and duplicate requests |

One test emits a runpy warning because it executes an already imported worker
module; this is recorded rather than suppressed. The skip is retained from the
existing suite. The initial hosted run passed Python 3.10–3.12, lint/types, examples and
installed-wheel validation but failed the space throughput benchmark. A follow-up
batches validated low-dimensional permission calculations and retains the original
performance thresholds. Scalar/batch parity and invalid-input regressions are
included. The follow-up hosted run must pass before merge.
CI runs Python 3.10, 3.11 and 3.12 on this branch; release workflows repeat gates
before distribution publication. No release tag or PyPI publication is included.

This does not establish real GPU, remote Redis/Postgres/S3, container execution,
multi-host partitions or every experimental quantum primitive. See
[supported contracts](production-readiness.md) and [deployment](deployment.md).
