# Releasing

QES follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **MAJOR** — incompatible public API changes (anything exported from
  `src/qes/__init__.py`).
- **MINOR** — new backwards-compatible functionality (new modules, new
  optional parameters, new examples).
- **PATCH** — backwards-compatible bug fixes.

## Release process

1. Update `CHANGELOG.md`: move the relevant `[Unreleased]` entries under a
   new `## [X.Y.Z] - YYYY-MM-DD` heading.
2. Bump `version` in `pyproject.toml` to match.
3. Ensure CI is green on `main` (lint, type check, tests).
4. Commit the version bump: `git commit -m "Release vX.Y.Z"`.
5. Tag the release: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. Create a GitHub Release from the tag, using the corresponding
   `CHANGELOG.md` section as the release notes.
7. Pushing a `vX.Y.Z` tag automatically triggers
   `.github/workflows/publish.yml`, which builds the sdist/wheel, verifies
   the tag matches `pyproject.toml`, runs the test suite, and publishes to
   PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/)
   (no stored API token required). Configure the `pypi` GitHub Environment
   as a trusted publisher for the `qes` project on PyPI before the first
   release; after that, releases are published automatically on tag push.
   To publish manually instead: `python -m build && twine upload dist/*`.
