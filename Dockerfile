# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# Install the package in editable mode so the container always reflects the
# committed source tree; numpy is the only runtime dependency.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY examples ./examples
COPY docs ./docs

# Durable runtime state (SQLite / file-backed store) is written here; mount a
# volume at this path in production so job history survives container restarts.
ENV QES_RUNTIME_STATE_DIR=/app/state
RUN mkdir -p "$QES_RUNTIME_STATE_DIR"

ENTRYPOINT ["python", "-m", "qes.worker"]
CMD ["--worker-count", "4"]
