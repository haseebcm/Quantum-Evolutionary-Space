# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# Install the committed package; numpy is the only runtime dependency.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY examples ./examples
COPY docs ./docs

# Durable runtime state (SQLite / file-backed store) is written here; mount a
# volume at this path in production so job history survives container restarts.
ENV QES_RUNTIME_STATE_DIR=/app/state
RUN useradd --create-home --uid 10001 qes && mkdir -p "$QES_RUNTIME_STATE_DIR" && chown qes:qes "$QES_RUNTIME_STATE_DIR"
USER qes

ENTRYPOINT ["python", "-m", "qes.worker"]
CMD ["--task-queue", "/app/state/tasks.sqlite"]
