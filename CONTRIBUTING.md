# Contributing to QES

Thanks for your interest in improving the QES framework. This project
implements the formal specification in
[`docs/QES-architecture.md`](docs/QES-architecture.md) as a reusable Python
framework (`src/qes/`). Contributions that add new domain-agnostic
capabilities, examples, tests, or documentation are welcome.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
```

This installs the framework in editable mode plus its dev tooling:
`pytest`, `pytest-cov`, `ruff`, and `mypy`.

## Before opening a pull request

Run the full local check suite and make sure everything is green:

```powershell
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\python -m mypy
.\.venv\Scripts\python -m pytest -q --cov=qes --cov-report=term-missing
```

These same checks run in CI (`.github/workflows/ci.yml`) across
Python 3.10–3.12 on every push and pull request.

## Guidelines

- **Keep modules domain-agnostic.** QES is a framework, not an application:
  new code should not hard-code assumptions about a specific domain
  (finance, physics, etc.). Domain-specific logic belongs in `examples/`.
- **Cite the spec.** If a change implements or modifies formal behavior,
  reference the relevant section number from `docs/QES-architecture.md` in
  the module docstring and/or PR description.
- **Add tests.** Every module in `src/qes/` has a matching `tests/test_*.py`
  file with unit tests. New modules or public functions should follow the
  same pattern.
- **Small, composable changes.** QES's modules are intentionally small and
  single-purpose (one module per doc section-group), wired together by
  `qes.space.QESSpace`. Prefer adding a new composable primitive over
  growing an existing one.
- **Don't break the public API** (`src/qes/__init__.py`) without updating
  the README and, if relevant, `CHANGELOG.md`.

## Reporting bugs / requesting features

Please use the issue templates under `.github/ISSUE_TEMPLATE/`.
