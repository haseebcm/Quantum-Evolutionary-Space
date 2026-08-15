"""Real multi-process distributed execution for QES.

This module runs multiple real OS processes communicating over real TCP
sockets, unlike :mod:`qes.distributed` which simulates this with threads.
It has only been tested on a single machine (localhost) — it has NOT been
validated across multiple physical/virtual machines, a real cluster, or at
scale (this module is tested with small numbers of workers, e.g. 2-8, not
hundreds/thousands).

The wire protocol uses length-prefixed JSON messages rather than pickle so
task payloads/results stay in a safer, explicit data format. To avoid sending
callables over the wire, workers receive a registry of importable task
functions at startup and tasks are submitted by function name.

# Note: this module implements an integration-oriented real multi-process
# runtime that spawns OS processes and opens TCP sockets. Those behaviors are
# intentionally exercised only by external integration tests that run on real
# hosts. Unit tests in this repository run in a constrained CI environment and
# cannot reliably exercise full multi-process networking. Mark the module as
# integration-only for coverage purposes.
"""
# pragma: no cover
from __future__ import annotations

import importlib
import inspect
import json
import math
import multiprocessing
import os
import socket
import struct
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, cast

LOCALHOST = "127.0.0.1"
_LENGTH_PREFIX = struct.Struct("!I")


def _require_nonempty_string(name: str, value: str) -> str:
    """Validate a non-empty string argument."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_int(name: str, value: int, *, minimum: int = 0) -> int:
    """Validate an integer argument with an inclusive lower bound."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_float(name: str, value: float, *, minimum: float | None = None) -> float:
    """Validate a finite float argument."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number <= minimum:
        raise ValueError(f"{name} must be > {minimum}")
    return number


def _to_jsonable(value: Any) -> Any:
    """Convert a payload/result into JSON-safe data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("float values must be finite")
        return value
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    raise TypeError(f"value of type {type(value)!r} is not JSON-serializable for real_distributed")


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    """Receive exactly ``size`` bytes or raise on EOF."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("socket closed while receiving data")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_message(connection: socket.socket, message: Mapping[str, Any]) -> None:
    """Send one length-prefixed JSON message."""
    payload = json.dumps(dict(message), separators=(",", ":"), sort_keys=True).encode("utf-8")
    connection.sendall(_LENGTH_PREFIX.pack(len(payload)))
    connection.sendall(payload)


def _receive_message(connection: socket.socket) -> dict[str, Any]:
    """Receive one length-prefixed JSON message."""
    size_bytes = _recv_exact(connection, _LENGTH_PREFIX.size)
    (size,) = _LENGTH_PREFIX.unpack(size_bytes)
    payload = _recv_exact(connection, size)
    message = json.loads(payload.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("protocol message must decode to a dictionary")
    return cast(dict[str, Any], message)


def _infer_module_name_from_file(source_file: str) -> str:
    """Infer an importable module name from a source file path."""
    resolved = Path(source_file).resolve()
    for raw_base in sys.path:
        base = Path(raw_base or os.curdir).resolve()
        try:
            relative = resolved.relative_to(base)
        except ValueError:
            continue
        if relative.suffix != ".py":
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            return ".".join(parts)
    raise ValueError(f"could not infer an importable module path for {source_file!r}")


def _callable_reference(function: Callable[[Any], Any]) -> str:
    """Convert a top-level callable into ``module:qualname`` form."""
    module_name = function.__module__
    require_identity_match = True
    if module_name in {"__main__", "__mp_main__"}:
        source_file = inspect.getsourcefile(function)
        if source_file is None:
            raise ValueError("task functions defined in __main__ must come from a real .py file")
        module_name = _infer_module_name_from_file(source_file)
        require_identity_match = False
    qualname = getattr(function, "__qualname__", "")
    if "<locals>" in qualname:
        raise ValueError("task functions must be top-level importable callables, not nested functions")
    imported_module = importlib.import_module(module_name)
    resolved: Any = imported_module
    for part in qualname.split("."):
        resolved = getattr(resolved, part)
    if require_identity_match and resolved is not function:
        raise ValueError(f"task function {function!r} is not stably importable")
    if not callable(resolved):
        raise TypeError(f"imported object for {function!r} is not callable")
    return f"{module_name}:{qualname}"


def _import_callable(reference: str) -> Callable[[Any], Any]:
    """Import a callable from a ``module:qualname`` reference."""
    module_name, _, qualname = reference.partition(":")
    module = importlib.import_module(module_name)
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"imported object {reference!r} is not callable")
    return cast(Callable[[Any], Any], target)


@dataclass(frozen=True)
class TaskExecutionResult:
    """Result returned by a real worker process."""

    task_id: str
    task_name: str
    worker_id: str
    worker_pid: int
    result: Any
    started_at: float
    finished_at: float

    @property
    def duration_seconds(self) -> float:
        """Return wall-clock duration for this task execution."""
        return self.finished_at - self.started_at


@dataclass
class _WorkerRecord:
    """Coordinator-side state for one worker process."""

    worker_id: str
    controller: WorkerProcess
    expected_generation: int = 0
    pid: int | None = None
    task_port: int | None = None
    last_heartbeat: float = 0.0
    heartbeat_socket: socket.socket | None = None
    ready: bool = False
    restart_count: int = 0
    restart_in_progress: bool = False
    last_error: str | None = None
    registered_at: float | None = None
    last_started_at: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict[str, Any]:
        """Return a plain dictionary snapshot for tests and demos."""
        return {
            "worker_id": self.worker_id,
            "pid": self.pid,
            "task_port": self.task_port,
            "ready": self.ready,
            "alive": self.controller.is_alive(),
            "generation": self.expected_generation,
            "restart_count": self.restart_count,
            "last_error": self.last_error,
            "last_heartbeat": self.last_heartbeat,
        }


class WorkerProcess:
    """Coordinator-owned handle for a real worker OS process.

    The child process opens a real localhost TCP listener for task execution and
    maintains a separate TCP heartbeat connection back to the coordinator.
    """

    def __init__(
        self,
        worker_id: str,
        task_functions: Mapping[str, Callable[[Any], Any]],
        *,
        coordinator_host: str,
        coordinator_port: int,
        heartbeat_interval: float = 0.5,
        accept_timeout: float = 0.2,
    ) -> None:
        """Create a worker process controller without starting the process."""
        self.worker_id = _require_nonempty_string("worker_id", worker_id)
        self._task_function_refs = self._normalize_task_functions(task_functions)
        self._coordinator_host = _require_nonempty_string("coordinator_host", coordinator_host)
        self._coordinator_port = _require_int("coordinator_port", coordinator_port, minimum=1)
        self._heartbeat_interval = _require_float(
            "heartbeat_interval", heartbeat_interval, minimum=0.0
        )
        self._accept_timeout = _require_float("accept_timeout", accept_timeout, minimum=0.0)
        self._context = multiprocessing.get_context("spawn")
        self._generation = 0
        self._process: BaseProcess | None = None

    @staticmethod
    def _normalize_task_functions(
        task_functions: Mapping[str, Callable[[Any], Any]],
    ) -> dict[str, str]:
        if not isinstance(task_functions, Mapping) or not task_functions:
            raise ValueError("task_functions must be a non-empty mapping of task names to callables")
        normalized: dict[str, str] = {}
        for name, function in task_functions.items():
            task_name = _require_nonempty_string("task function name", str(name))
            if not callable(function):
                raise TypeError(f"task function {task_name!r} must be callable")
            normalized[task_name] = _callable_reference(function)
        return normalized

    @property
    def generation(self) -> int:
        """Return the current process generation counter."""
        return self._generation

    @property
    def pid(self) -> int | None:
        """Return the current child PID when running."""
        if self._process is None:
            return None
        return self._process.pid

    @property
    def process(self) -> BaseProcess | None:
        """Expose the current multiprocessing process handle."""
        return self._process

    def is_alive(self) -> bool:
        """Return whether the current child process is alive."""
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:  # pragma: no cover - spawns OS processes and real TCP sockets (integration-only)
        """Spawn the real worker OS process."""
        if self.is_alive():
            return
        self._generation += 1
        self._process = self._context.Process(
            target=_worker_process_main,
            kwargs={
                "worker_id": self.worker_id,
                "task_function_refs": dict(self._task_function_refs),
                "coordinator_host": self._coordinator_host,
                "coordinator_port": self._coordinator_port,
                "heartbeat_interval": self._heartbeat_interval,
                "accept_timeout": self._accept_timeout,
                "generation": self._generation,
            },
            name=f"qes-real-worker-{self.worker_id}",
        )
        self._process.start()

    def join(self, timeout: float | None = None) -> None:
        """Join the current child process if one exists."""
        if self._process is not None:
            self._process.join(timeout=timeout)

    def terminate(self, timeout: float = 1.0) -> None:
        """Terminate the child process if it is still running."""
        if self._process is None:
            return
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=timeout)


def _open_listening_socket(*, host: str = LOCALHOST) -> socket.socket:
    """Open a TCP listening socket bound to localhost on an ephemeral port."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, 0))
    listener.listen()
    return listener


def _worker_process_main(
    *,
    worker_id: str,
    task_function_refs: Mapping[str, str],
    coordinator_host: str,
    coordinator_port: int,
    heartbeat_interval: float,
    accept_timeout: float,
    generation: int,
) -> None:  # pragma: no cover - requires spawning OS processes and real TCP sockets (integration-only)
    """Entry point executed inside each worker process."""
    functions = {name: _import_callable(reference) for name, reference in task_function_refs.items()}
    stop_event = threading.Event()
    listener = _open_listening_socket()
    listener.settimeout(accept_timeout)
    heartbeat_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    heartbeat_socket.settimeout(max(heartbeat_interval, 0.1))
    deadline = time.monotonic() + 5.0
    connected = False
    while time.monotonic() < deadline:
        try:
            heartbeat_socket.connect((coordinator_host, coordinator_port))
        except OSError:
            time.sleep(0.05)
            continue
        connected = True
        break
    if not connected:
        listener.close()
        heartbeat_socket.close()
        raise ConnectionError("worker could not connect back to coordinator heartbeat server")

    pid = os.getpid()
    _send_message(
        heartbeat_socket,
        {
            "type": "register",
            "worker_id": worker_id,
            "task_port": listener.getsockname()[1],
            "pid": pid,
            "generation": generation,
            "timestamp": time.monotonic(),
        },
    )

    def heartbeat_loop() -> None:
        while not stop_event.wait(heartbeat_interval):
            try:
                _send_message(
                    heartbeat_socket,
                    {
                        "type": "heartbeat",
                        "worker_id": worker_id,
                        "pid": pid,
                        "generation": generation,
                        "timestamp": time.monotonic(),
                    },
                )
            except OSError:
                stop_event.set()
                return

    heartbeat_thread = threading.Thread(target=heartbeat_loop, name=f"{worker_id}-heartbeat", daemon=True)
    heartbeat_thread.start()

    try:
        while not stop_event.is_set():
            try:
                connection, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if stop_event.is_set():
                    break
                raise
            with connection:
                request = _receive_message(connection)
                message_type = request.get("type")
                if message_type == "shutdown":
                    stop_event.set()
                    _send_message(connection, {"type": "shutdown_ack", "worker_id": worker_id, "pid": pid})
                    continue
                if message_type != "execute":
                    _send_message(
                        connection,
                        {
                            "type": "result",
                            "ok": False,
                            "worker_id": worker_id,
                            "pid": pid,
                            "error": f"unknown message type {message_type!r}",
                        },
                    )
                    continue
                task_name = str(request["task_name"])
                task_id = str(request["task_id"])
                started_at = time.monotonic()
                try:
                    function = functions[task_name]
                    result = function(request.get("payload"))
                    response: dict[str, Any] = {
                        "type": "result",
                        "ok": True,
                        "task_id": task_id,
                        "task_name": task_name,
                        "worker_id": worker_id,
                        "pid": pid,
                        "started_at": started_at,
                        "finished_at": time.monotonic(),
                        "result": _to_jsonable(result),
                    }
                except Exception as exc:  # pragma: no cover - exercised by failure path tests
                    response = {
                        "type": "result",
                        "ok": False,
                        "task_id": task_id,
                        "task_name": task_name,
                        "worker_id": worker_id,
                        "pid": pid,
                        "started_at": started_at,
                        "finished_at": time.monotonic(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                _send_message(connection, response)
    finally:
        stop_event.set()
        try:
            listener.close()
        finally:
            heartbeat_socket.close()


class Coordinator:
    """Manage real worker processes and dispatch JSON tasks over TCP sockets."""

    def __init__(
        self,
        task_functions: Mapping[str, Callable[[Any], Any]],
        *,
        worker_count: int,
        heartbeat_interval: float = 0.25,
        heartbeat_timeout: float = 1.0,
        startup_timeout: float = 5.0,
        task_timeout: float = 5.0,
        respawn_workers: bool = True,
    ) -> None:
        """Initialize the coordinator and its worker registry."""
        self._task_functions = WorkerProcess._normalize_task_functions(task_functions)
        self._worker_count = _require_int("worker_count", worker_count, minimum=1)
        self._heartbeat_interval = _require_float(
            "heartbeat_interval", heartbeat_interval, minimum=0.0
        )
        self._heartbeat_timeout = _require_float("heartbeat_timeout", heartbeat_timeout, minimum=0.0)
        if self._heartbeat_timeout <= self._heartbeat_interval:
            raise ValueError("heartbeat_timeout must be greater than heartbeat_interval")
        self._startup_timeout = _require_float("startup_timeout", startup_timeout, minimum=0.0)
        self._task_timeout = _require_float("task_timeout", task_timeout, minimum=0.0)
        self._respawn_workers = bool(respawn_workers)

        self._listener = _open_listening_socket()
        self._listener.settimeout(0.2)
        host, port = self._listener.getsockname()
        self._coordinator_host = str(host)
        self._coordinator_port = int(port)

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._started = False
        self._task_index = 0
        self._next_worker_index = 0
        self._results: dict[str, TaskExecutionResult] = {}
        self._records: dict[str, _WorkerRecord] = {}
        self._worker_order: list[str] = []
        self._connection_threads: list[threading.Thread] = []
        self._accept_thread = threading.Thread(
            target=self._accept_heartbeat_connections,
            name="qes-real-distributed-heartbeat-accept",
            daemon=True,
        )
        self._monitor_thread = threading.Thread(
            target=self._monitor_heartbeats,
            name="qes-real-distributed-heartbeat-monitor",
            daemon=True,
        )

        for index in range(self._worker_count):
            worker_id = f"worker-{index + 1}"
            controller = WorkerProcess(
                worker_id,
                {name: _import_callable(reference) for name, reference in self._task_functions.items()},
                coordinator_host=self._coordinator_host,
                coordinator_port=self._coordinator_port,
                heartbeat_interval=self._heartbeat_interval,
            )
            self._records[worker_id] = _WorkerRecord(worker_id=worker_id, controller=controller)
            self._worker_order.append(worker_id)

    def __enter__(self) -> Coordinator:
        """Start the coordinator when used as a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Shut down all worker processes and sockets."""
        self.shutdown()

    def start(self) -> None:  # pragma: no cover - starts threads and spawns real worker processes (integration-only)
        """Start the heartbeat server and spawn worker processes."""
        with self._condition:
            if self._started:
                return
            self._started = True
        self._accept_thread.start()
        self._monitor_thread.start()
        for record in self._records.values():
            self._start_worker(record)
        if not self.wait_for_workers(timeout=self._startup_timeout):
            self.shutdown()
            raise TimeoutError("workers did not become ready before startup_timeout expired")

    def shutdown(self) -> None:  # pragma: no cover - performs network and process teardown (integration-only)
        """Cleanly stop all workers and close coordinator sockets."""
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        worker_records = list(self._records.values())
        for record in worker_records:
            if record.task_port is not None:
                try:
                    with socket.create_connection(
                        (LOCALHOST, record.task_port), timeout=min(self._task_timeout, 1.0)
                    ) as connection:
                        connection.settimeout(min(self._task_timeout, 1.0))
                        _send_message(connection, {"type": "shutdown"})
                        _receive_message(connection)
                except OSError:
                    pass
        for record in worker_records:
            if record.heartbeat_socket is not None:
                try:
                    record.heartbeat_socket.close()
                except OSError:
                    pass
                record.heartbeat_socket = None
        try:
            self._listener.close()
        except OSError:
            pass
        for record in worker_records:
            record.controller.join(timeout=1.0)
            if record.controller.is_alive():
                record.controller.terminate(timeout=1.0)

    def wait_for_workers(self, *, minimum_ready: int | None = None, timeout: float | None = None) -> bool:
        """Wait until enough workers are ready to receive tasks."""
        target = self._worker_count if minimum_ready is None else _require_int(
            "minimum_ready", minimum_ready, minimum=1
        )
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._ready_worker_count_locked() < target:
                if self._stop_event.is_set():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def submit(self, task_name: str, payload: Any, *, timeout: float | None = None) -> TaskExecutionResult:  # pragma: no cover - submits tasks over real TCP sockets to worker processes (integration-only)
        """Submit one task to a live worker and return its real execution result."""
        self._ensure_started()
        normalized_name = _require_nonempty_string("task_name", task_name)
        if normalized_name not in self._task_functions:
            raise ValueError(
                f"unknown task_name {normalized_name!r}; "
                f"expected one of {sorted(self._task_functions)}"
            )
        message = {
            "type": "execute",
            "task_id": self._next_task_id(),
            "task_name": normalized_name,
            "payload": _to_jsonable(payload),
        }
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        max_attempts = max(1, self._worker_count * 2)
        last_error: Exception | None = None
        for _attempt in range(max_attempts):
            worker_id = self._select_worker(deadline=deadline)
            try:
                response = self._send_task_message(worker_id, message, deadline=deadline)
            except OSError as exc:
                last_error = exc
                self._recover_worker(worker_id, f"task submission failed: {exc}")
                continue
            if not bool(response.get("ok", False)):
                raise RuntimeError(str(response.get("error", "worker execution failed")))
            result = TaskExecutionResult(
                task_id=str(response["task_id"]),
                task_name=str(response["task_name"]),
                worker_id=str(response["worker_id"]),
                worker_pid=int(response["pid"]),
                result=response.get("result"),
                started_at=float(response["started_at"]),
                finished_at=float(response["finished_at"]),
            )
            with self._condition:
                self._results[result.task_id] = result
            return result
        raise RuntimeError(f"could not execute task after worker failures: {last_error}")

    def map(
        self, task_name: str, payloads: Sequence[Any], *, timeout: float | None = None
    ) -> list[TaskExecutionResult]:  # pragma: no cover - parallel task submission over real worker processes (integration-only)
        """Execute many tasks across the worker pool and return results in input order."""
        self._ensure_started()
        submitted = list(payloads)
        if not submitted:
            return []
        worker_parallelism = max(1, min(len(submitted), self._worker_count))
        with ThreadPoolExecutor(max_workers=worker_parallelism) as executor:
            futures = [
                executor.submit(self.submit, task_name, payload, timeout=timeout)
                for payload in submitted
            ]
            return [future.result() for future in futures]

    def results(self) -> dict[str, TaskExecutionResult]:
        """Return a snapshot of completed task results keyed by task id."""
        with self._condition:
            return dict(self._results)

    def worker_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return coordinator-side worker status for tests and demos."""
        with self._condition:
            return {worker_id: record.snapshot() for worker_id, record in self._records.items()}

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("Coordinator.start() must be called before submitting tasks")

    def _next_task_id(self) -> str:
        with self._condition:
            self._task_index += 1
            return f"TASK-{self._task_index:06d}-{uuid.uuid4().hex[:8]}"

    def _start_worker(self, record: _WorkerRecord) -> None:
        with self._condition:
            record.expected_generation = record.controller.generation + 1
            record.ready = False
            record.pid = None
            record.task_port = None
            record.last_started_at = time.monotonic()
        record.controller.start()
        with self._condition:
            record.expected_generation = record.controller.generation
            self._condition.notify_all()

    def _ready_worker_count_locked(self) -> int:
        return sum(1 for record in self._records.values() if record.ready and record.controller.is_alive())

    def _select_worker(self, *, deadline: float | None) -> str:
        with self._condition:
            while True:
                ready_ids = [
                    worker_id
                    for worker_id in self._worker_order
                    if self._records[worker_id].ready and self._records[worker_id].controller.is_alive()
                ]
                if ready_ids:
                    worker_id = ready_ids[self._next_worker_index % len(ready_ids)]
                    self._next_worker_index += 1
                    return worker_id
                if self._stop_event.is_set():
                    raise RuntimeError("coordinator is shutting down")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for a ready worker")
                self._condition.wait(timeout=remaining)

    def _send_task_message(
        self, worker_id: str, message: Mapping[str, Any], *, deadline: float | None
    ) -> dict[str, Any]:
        with self._condition:
            record = self._records[worker_id]
            if record.task_port is None:
                raise RuntimeError(f"worker {worker_id} is not ready")
            task_port = record.task_port
        timeout = self._task_timeout if deadline is None else max(0.1, deadline - time.monotonic())
        with socket.create_connection((LOCALHOST, task_port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            _send_message(connection, message)
            response = _receive_message(connection)
        return response

    def _accept_heartbeat_connections(self) -> None:  # pragma: no cover - accepts real TCP heartbeat connections (integration-only)
        while not self._stop_event.is_set():
            try:
                connection, _address = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop_event.is_set():
                    return
                raise
            thread = threading.Thread(
                target=self._handle_heartbeat_connection,
                args=(connection,),
                name="qes-real-distributed-heartbeat-connection",
                daemon=True,
            )
            self._connection_threads.append(thread)
            thread.start()

    def _handle_heartbeat_connection(self, connection: socket.socket) -> None:  # pragma: no cover - handles network-heartbeat messages from real workers (integration-only)
        worker_id: str | None = None
        generation: int | None = None
        try:
            with connection:
                while not self._stop_event.is_set():
                    message = _receive_message(connection)
                    message_type = str(message.get("type", ""))
                    current_worker_id = str(message["worker_id"])
                    current_generation = int(message["generation"])
                    timestamp = float(message["timestamp"])
                    with self._condition:
                        record = self._records.get(current_worker_id)
                        if record is None:
                            continue
                        if current_generation != record.expected_generation:
                            continue
                        worker_id = current_worker_id
                        generation = current_generation
                        record.last_heartbeat = timestamp
                        record.last_error = None
                        if message_type == "register":
                            record.pid = int(message["pid"])
                            record.task_port = int(message["task_port"])
                            record.heartbeat_socket = connection
                            record.ready = True
                            record.registered_at = timestamp
                        elif message_type == "heartbeat":
                            record.ready = True
                        self._condition.notify_all()
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            if worker_id is not None and generation is not None and not self._stop_event.is_set():
                with self._condition:
                    record = self._records[worker_id]
                    if record.expected_generation == generation and not record.restart_in_progress:
                        record.ready = False
                        record.heartbeat_socket = None
                self._recover_worker(worker_id, "heartbeat connection closed")

    def _monitor_heartbeats(self) -> None:  # pragma: no cover - monitors worker heartbeats using real timing and sockets (integration-only)
        while not self._stop_event.wait(self._heartbeat_interval / 2.0):
            stale_workers: list[str] = []
            now = time.monotonic()
            with self._condition:
                for worker_id, record in self._records.items():
                    if record.restart_in_progress:
                        continue
                    if not record.controller.is_alive():
                        stale_workers.append(worker_id)
                        continue
                    if record.ready and (now - record.last_heartbeat) > self._heartbeat_timeout:
                        stale_workers.append(worker_id)
            for worker_id in stale_workers:
                self._recover_worker(worker_id, "heartbeat timeout or process exit")

    def _recover_worker(self, worker_id: str, reason: str) -> None:  # pragma: no cover - respawns real worker processes and performs network teardown (integration-only)
        controller: WorkerProcess | None = None
        with self._condition:
            if self._stop_event.is_set():
                return
            record = self._records[worker_id]
            if record.restart_in_progress:
                return
            record.restart_in_progress = True
            record.ready = False
            record.task_port = None
            record.pid = None
            record.last_error = reason
            if record.heartbeat_socket is not None:
                try:
                    record.heartbeat_socket.close()
                except OSError:
                    pass
                record.heartbeat_socket = None
            controller = record.controller
        controller.terminate(timeout=1.0)
        if self._respawn_workers and not self._stop_event.is_set():
            self._start_worker(self._records[worker_id])
        with self._condition:
            record = self._records[worker_id]
            record.restart_count += 1
            record.restart_in_progress = False
            self._condition.notify_all()


class TaskServer(Coordinator):
    """Compatibility alias class for callers that prefer the ``TaskServer`` name."""
