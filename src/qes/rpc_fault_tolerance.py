"""Real localhost RPC and fault-tolerance utilities for QES.

This module uses real TCP sockets, real JSON messages, and real wall-clock
timing via :func:`time.monotonic` throughout. It is intentionally small and
honest: the implementation has been validated on localhost with a handful of
peers (roughly 2-5) and short-lived tests, not across real multi-machine
networks, WAN latency, packet loss, or large-scale production deployments.

QES uses quantum/universe language elsewhere in the project as a metaphor for
classical possibility-space search. This module is likewise classical systems
code: ordinary sockets, threads, heartbeats, retries, and leader selection.
"""
from __future__ import annotations

import json
import random
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_RESERVED_METHODS = {"__heartbeat__", "__ping__"}
_T = TypeVar("_T")


def _require_nonempty_string(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _require_positive_int(name: str, value: int, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = ">=" if allow_zero else ">"
        threshold = 0 if allow_zero else 0
        raise ValueError(f"{name} must be {comparator} {threshold}")
    return value


def _require_positive_float(name: str, value: float, *, allow_zero: bool = False) -> float:
    number = float(value)
    if number < 0.0 or (not allow_zero and number == 0.0):
        comparator = ">=" if allow_zero else ">"
        threshold = 0.0
        raise ValueError(f"{name} must be {comparator} {threshold}")
    if not (number < float("inf")):
        raise ValueError(f"{name} must be finite")
    return number


def _require_host(host: str) -> str:
    return _require_nonempty_string("host", host)


def _require_port(port: int) -> int:
    validated = _require_positive_int("port", port, allow_zero=True)
    if validated > 65535:
        raise ValueError("port must be <= 65535")
    return validated


def _require_peer_id(peer_id: int) -> int:
    return _require_positive_int("peer_id", peer_id)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise EOFError("socket closed before the full message was received")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_message(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(data) > _MAX_MESSAGE_BYTES:
        raise ValueError("message exceeds maximum supported size")
    sock.sendall(len(data).to_bytes(4, byteorder="big", signed=False))
    sock.sendall(data)


def _recv_message(sock: socket.socket) -> dict[str, Any]:
    raw_length = _recv_exact(sock, 4)
    length = int.from_bytes(raw_length, byteorder="big", signed=False)
    if length <= 0 or length > _MAX_MESSAGE_BYTES:
        raise ValueError("received invalid message length")
    raw_payload = _recv_exact(sock, length)
    payload = json.loads(raw_payload.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("received payload must be a JSON object")
    return payload


class RPCError(RuntimeError):
    """Base class for RPC-related failures."""


class RPCConnectionError(RPCError):
    """Raised when a real socket connection, read, or write operation fails."""


class RPCRemoteError(RPCError):
    """Raised when the remote RPC server returns an explicit error response."""


class FailureDetector:
    """Real heartbeat failure detector based on :func:`time.monotonic`.

    The detector tracks peer liveness in-memory and marks a peer as failed when
    no heartbeat arrives within ``heartbeat_timeout`` seconds. It has only been
    exercised on localhost with a few peers and short timeouts.
    """

    def __init__(self, heartbeat_timeout: float, *, check_interval: float = 0.05) -> None:
        """Initialize a detector with real timeout-based failure marking."""
        self.heartbeat_timeout = _require_positive_float("heartbeat_timeout", heartbeat_timeout)
        self.check_interval = _require_positive_float("check_interval", check_interval)
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}
        self._failed: set[str] = set()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background failure-marking thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._monitor_loop, name="qes-failure-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and wait for it to exit."""
        self._running.clear()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(self.check_interval * 4.0, 0.2))
        self._thread = None

    def register_peer(self, peer_id: str) -> None:
        """Register a peer that is expected to send future heartbeats."""
        normalized = _require_nonempty_string("peer_id", peer_id)
        now = time.monotonic()
        with self._lock:
            self._last_seen.setdefault(normalized, now)
            self._failed.discard(normalized)

    def record_heartbeat(self, peer_id: str) -> None:
        """Record a newly received heartbeat for ``peer_id``."""
        normalized = _require_nonempty_string("peer_id", peer_id)
        now = time.monotonic()
        with self._lock:
            self._last_seen[normalized] = now
            self._failed.discard(normalized)

    def peer_failed(self, peer_id: str) -> bool:
        """Return whether ``peer_id`` is currently marked failed."""
        normalized = _require_nonempty_string("peer_id", peer_id)
        self.check_failures()
        with self._lock:
            return normalized in self._failed

    def last_heartbeat(self, peer_id: str) -> float | None:
        """Return the last recorded heartbeat time for ``peer_id``."""
        normalized = _require_nonempty_string("peer_id", peer_id)
        with self._lock:
            return self._last_seen.get(normalized)

    def failed_peers(self) -> tuple[str, ...]:
        """Return all currently failed peers as a sorted tuple."""
        self.check_failures()
        with self._lock:
            return tuple(sorted(self._failed))

    def alive_peers(self) -> tuple[str, ...]:
        """Return all peers not currently marked failed."""
        self.check_failures()
        with self._lock:
            alive = [peer_id for peer_id in self._last_seen if peer_id not in self._failed]
        return tuple(sorted(alive))

    def snapshot(self) -> dict[str, dict[str, float | bool]]:
        """Return a plain-dict snapshot of tracked liveness state."""
        self.check_failures()
        with self._lock:
            snapshot: dict[str, dict[str, float | bool]] = {}
            for peer_id, last_seen in self._last_seen.items():
                snapshot[peer_id] = {
                    "last_heartbeat": last_seen,
                    "failed": peer_id in self._failed,
                }
        return snapshot

    def check_failures(self) -> tuple[str, ...]:
        """Refresh failure state immediately and return failed peer IDs."""
        now = time.monotonic()
        with self._lock:
            failed = {
                peer_id
                for peer_id, last_seen in self._last_seen.items()
                if (now - last_seen) > self.heartbeat_timeout
            }
            self._failed = failed
            return tuple(sorted(failed))

    def _monitor_loop(self) -> None:
        while self._running.is_set():
            self.check_failures()
            time.sleep(self.check_interval)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff policy for retrying real RPC transport failures.

    The policy sleeps using real wall-clock time between attempts. It retries
    socket-level failures and timeouts, not successful RPC responses that
    merely contain remote application errors.
    """

    max_attempts: int = 3
    base_delay: float = 0.05
    backoff_multiplier: float = 2.0
    jitter: float = 0.0

    def __post_init__(self) -> None:
        _require_positive_int("max_attempts", self.max_attempts)
        _require_positive_float("base_delay", self.base_delay, allow_zero=True)
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")
        _require_positive_float("jitter", self.jitter, allow_zero=True)

    def compute_delay(self, attempt_number: int) -> float:
        """Compute the post-failure delay before the next attempt."""
        _require_positive_int("attempt_number", attempt_number)
        delay = self.base_delay * (self.backoff_multiplier ** (attempt_number - 1))
        if self.jitter > 0.0:
            delay += random.uniform(0.0, self.jitter)
        return delay

    def execute(self, operation: Callable[[], _T]) -> _T:
        """Execute ``operation`` with retries for transport failures."""
        last_error: RPCConnectionError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except RPCConnectionError as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.compute_delay(attempt))
        if last_error is None:
            raise RuntimeError("retry execution failed without capturing an error")
        raise last_error


class RPCServer:
    """Minimal real JSON-over-TCP RPC server for localhost experiments.

    Each client request is sent over a real TCP connection using length-prefixed
    framing. The server also exposes a built-in heartbeat endpoint:
    clients send ``__heartbeat__`` messages to this server, and the internal
    :class:`FailureDetector` marks them failed if heartbeats stop arriving.

    This server has only been exercised on localhost with a few peers and is
    not a production-ready distributed runtime.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        server_id: str | None = None,
        socket_timeout: float = 0.2,
        heartbeat_timeout: float = 0.3,
        detector_check_interval: float = 0.05,
    ) -> None:
        """Configure a server that will bind when :meth:`start` is called."""
        self._host = _require_host(host)
        self._port = _require_port(port)
        self.server_id = server_id or f"{self._host}:{self._port or 'ephemeral'}"
        self.socket_timeout = _require_positive_float("socket_timeout", socket_timeout)
        self.failure_detector = FailureDetector(
            heartbeat_timeout,
            check_interval=detector_check_interval,
        )
        self._handlers: dict[str, Callable[[Any], Any]] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._server_socket: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._connection_threads: list[threading.Thread] = []
        self._client_sockets: set[socket.socket] = set()

    @property
    def address(self) -> tuple[str, int]:
        """Return the bound address after the server has started."""
        server_socket = self._server_socket
        if server_socket is None:
            raise RuntimeError("server has not been started")
        host, port = server_socket.getsockname()[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        """Bind, listen, and start the accept loop."""
        if self._running.is_set():
            return self.address
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self._host, self._port))
        server_socket.listen()
        server_socket.settimeout(self.socket_timeout)
        self._server_socket = server_socket
        self._running.set()
        self.failure_detector.start()
        self._accept_thread = threading.Thread(target=self._accept_loop, name="qes-rpc-accept", daemon=True)
        self._accept_thread.start()
        return self.address

    def stop(self) -> None:
        """Stop accepting new work and close all active sockets."""
        self._running.clear()
        self.failure_detector.stop()
        server_socket = self._server_socket
        self._server_socket = None
        if server_socket is not None:
            try:
                server_socket.close()
            except OSError:
                pass
        with self._lock:
            client_sockets = list(self._client_sockets)
        for client_socket in client_sockets:
            try:
                client_socket.close()
            except OSError:
                pass
        accept_thread = self._accept_thread
        if accept_thread is not None:
            accept_thread.join(timeout=max(self.socket_timeout * 4.0, 0.2))
        self._accept_thread = None
        with self._lock:
            connection_threads = list(self._connection_threads)
            self._connection_threads.clear()
        for thread in connection_threads:
            thread.join(timeout=max(self.socket_timeout * 4.0, 0.2))

    def register_handler(self, name: str, handler: Callable[[Any], Any]) -> None:
        """Register a named request handler."""
        normalized = _require_nonempty_string("name", name)
        if normalized in _RESERVED_METHODS:
            raise ValueError(f"{normalized!r} is reserved for internal server use")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handlers[normalized] = handler

    def register_client(self, client_id: str) -> None:
        """Register a client that is expected to send heartbeats."""
        self.failure_detector.register_peer(client_id)

    def client_snapshot(self) -> dict[str, dict[str, float | bool]]:
        """Return the current client heartbeat snapshot."""
        return self.failure_detector.snapshot()

    def _accept_loop(self) -> None:
        while self._running.is_set():
            server_socket = self._server_socket
            if server_socket is None:
                break
            try:
                client_socket, _address = server_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._running.is_set():
                    continue
                break
            client_socket.settimeout(self.socket_timeout)
            with self._lock:
                self._client_sockets.add(client_socket)
            thread = threading.Thread(
                target=self._handle_connection,
                args=(client_socket,),
                name="qes-rpc-client",
                daemon=True,
            )
            with self._lock:
                self._connection_threads = [item for item in self._connection_threads if item.is_alive()]
                self._connection_threads.append(thread)
            thread.start()

    def _handle_connection(self, client_socket: socket.socket) -> None:
        try:
            request = _recv_message(client_socket)
            response = self._dispatch_request(request)
            _send_message(client_socket, response)
        except (ConnectionError, OSError, EOFError, TimeoutError, ValueError, json.JSONDecodeError):
            return
        finally:
            with self._lock:
                self._client_sockets.discard(client_socket)
            try:
                client_socket.close()
            except OSError:
                pass

    def _dispatch_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id", ""))
        method = request.get("method")
        params = request.get("params")
        if not isinstance(method, str) or not method.strip():
            return self._error_response(request_id, "ValueError", "request method must be a non-empty string")
        normalized = method.strip()
        try:
            if normalized == "__heartbeat__":
                return self._handle_heartbeat(request_id, params)
            if normalized == "__ping__":
                return self._ok_response(
                    request_id,
                    {"server_id": self.server_id, "monotonic": time.monotonic()},
                )
            handler = self._handlers.get(normalized)
            if handler is None:
                raise LookupError(f"unknown RPC method {normalized!r}")
            result = handler(params)
            return self._ok_response(request_id, result)
        except Exception as exc:  # pragma: no cover - exercised in tests via public call path.
            return self._error_response(request_id, type(exc).__name__, str(exc))

    def _handle_heartbeat(self, request_id: str, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise TypeError("heartbeat params must be an object containing client_id")
        client_id = params.get("client_id")
        if not isinstance(client_id, str):
            raise TypeError("heartbeat params must include string client_id")
        self.failure_detector.record_heartbeat(client_id)
        return self._ok_response(
            request_id,
            {"server_id": self.server_id, "heartbeat_recorded": True},
        )

    @staticmethod
    def _ok_response(request_id: str, result: Any) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "status": "ok",
            "result": result,
        }

    @staticmethod
    def _error_response(request_id: str, error_type: str, message: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "status": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }


class RPCClient:
    """Real TCP RPC client for localhost call/response testing.

    Each RPC call opens a real TCP connection, sends one length-prefixed JSON
    request, reads one JSON response, and closes the socket. Optional heartbeat
    threads send real heartbeat calls to a server over the same transport.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        client_id: str = "rpc-client",
        timeout: float = 0.2,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Initialize a client for the given host/port."""
        self.host = _require_host(host)
        self.port = _require_port(port)
        self.client_id = _require_nonempty_string("client_id", client_id)
        self.timeout = _require_positive_float("timeout", timeout)
        self.retry_policy = retry_policy
        self._request_counter = 0
        self._counter_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def call(self, method: str, params: Any = None, *, timeout: float | None = None) -> Any:
        """Call a named RPC method and return the decoded JSON result."""
        normalized = _require_nonempty_string("method", method)
        effective_timeout = self.timeout if timeout is None else _require_positive_float("timeout", timeout)

        def operation() -> Any:
            return self._call_once(normalized, params, timeout=effective_timeout)

        if self.retry_policy is None:
            return operation()
        return self.retry_policy.execute(operation)

    def send_heartbeat(self) -> None:
        """Send one real heartbeat RPC to the server."""
        self._call_once("__heartbeat__", {"client_id": self.client_id}, timeout=self.timeout)

    def start_heartbeats(self, interval: float) -> None:
        """Start a background thread that periodically sends heartbeats."""
        validated_interval = _require_positive_float("interval", interval)
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            raise RuntimeError("heartbeats are already running")
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(validated_interval,),
            name="qes-rpc-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop_heartbeats(self) -> None:
        """Stop the background heartbeat thread."""
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None:
            thread.join(timeout=max(self.timeout * 4.0, 0.2))
        self._heartbeat_thread = None

    def _call_once(self, method: str, params: Any, *, timeout: float) -> Any:
        request_id = self._next_request_id()
        payload = {
            "request_id": request_id,
            "method": method,
            "params": params,
            "client_id": self.client_id,
        }
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                _send_message(sock, payload)
                response = _recv_message(sock)
        except (ConnectionError, OSError, EOFError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise RPCConnectionError(
                f"RPC transport failure calling {method!r} on {(self.host, self.port)}: {exc}"
            ) from exc

        status = response.get("status")
        if status == "ok":
            return response.get("result")
        if status == "error":
            error = response.get("error")
            if isinstance(error, dict):
                error_type = error.get("type", "RPCRemoteError")
                message = error.get("message", "remote RPC error")
            else:
                error_type = "RPCRemoteError"
                message = "remote RPC error"
            raise RPCRemoteError(f"{error_type}: {message}")
        raise RPCConnectionError("RPC response did not include a valid status field")

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._heartbeat_stop.is_set():
            try:
                self.send_heartbeat()
            except RPCError:
                pass
            self._heartbeat_stop.wait(interval)

    def _next_request_id(self) -> str:
        with self._counter_lock:
            self._request_counter += 1
            counter = self._request_counter
        return f"{self.client_id}-{counter}-{time.monotonic_ns()}"


class LeaderElection:
    """Simple lowest-ID-wins leader election for small localhost peer groups.

    Each peer periodically probes the others using real RPC ``__ping__`` calls.
    The smallest reachable peer ID is treated as leader. When the current
    leader stops responding for longer than ``heartbeat_timeout``, a new leader
    is chosen from the remaining reachable peers.

    This has only been validated on a single machine with a handful of peers.
    """

    def __init__(
        self,
        local_peer_id: int,
        peer_addresses: Mapping[int, tuple[str, int]],
        *,
        heartbeat_timeout: float = 0.3,
        check_interval: float = 0.05,
        rpc_timeout: float = 0.1,
    ) -> None:
        """Initialize an election participant."""
        self.local_peer_id = _require_peer_id(local_peer_id)
        self.heartbeat_timeout = _require_positive_float("heartbeat_timeout", heartbeat_timeout)
        self.check_interval = _require_positive_float("check_interval", check_interval)
        self.rpc_timeout = _require_positive_float("rpc_timeout", rpc_timeout)
        normalized_peers: dict[int, tuple[str, int]] = {}
        for peer_id, address in peer_addresses.items():
            normalized_peer_id = _require_peer_id(peer_id)
            if (
                not isinstance(address, tuple)
                or len(address) != 2
                or not isinstance(address[0], str)
                or not isinstance(address[1], int)
            ):
                raise TypeError("peer addresses must be (host, port) tuples")
            normalized_peers[normalized_peer_id] = (_require_host(address[0]), _require_port(address[1]))
        if self.local_peer_id not in normalized_peers:
            raise ValueError("local_peer_id must exist in peer_addresses")
        self.peer_addresses = dict(sorted(normalized_peers.items()))
        self.failure_detector = FailureDetector(
            self.heartbeat_timeout,
            check_interval=self.check_interval,
        )
        for peer_id in self.peer_addresses:
            self.failure_detector.register_peer(str(peer_id))
        self._leader_lock = threading.Lock()
        self._current_leader: int | None = None
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start probing peers and tracking the current leader."""
        self.failure_detector.start()
        self.determine_leader()
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._monitor_loop, name="qes-leader-election", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop peer monitoring and failure detection."""
        self._running.clear()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(self.check_interval * 4.0, 0.2))
        self._thread = None
        self.failure_detector.stop()

    def determine_leader(self) -> int | None:
        """Probe peers immediately and return the current leader ID."""
        for peer_id, address in self.peer_addresses.items():
            if peer_id == self.local_peer_id:
                self.failure_detector.record_heartbeat(str(peer_id))
                continue
            if self._probe_peer(address):
                self.failure_detector.record_heartbeat(str(peer_id))
        self.failure_detector.check_failures()
        leader = self._select_lowest_alive_peer()
        with self._leader_lock:
            self._current_leader = leader
        return leader

    def current_leader(self) -> int | None:
        """Return the most recently determined leader ID."""
        with self._leader_lock:
            return self._current_leader

    def leader_address(self) -> tuple[str, int] | None:
        """Return the current leader's address, if one is known."""
        leader = self.current_leader()
        if leader is None:
            return None
        return self.peer_addresses[leader]

    def wait_for_leader(self, expected_leader: int, timeout: float) -> bool:
        """Wait until ``expected_leader`` becomes the current leader."""
        validated_expected = _require_peer_id(expected_leader)
        validated_timeout = _require_positive_float("timeout", timeout)
        deadline = time.monotonic() + validated_timeout
        while time.monotonic() < deadline:
            if self.current_leader() == validated_expected:
                return True
            time.sleep(min(self.check_interval, 0.01))
        return self.current_leader() == validated_expected

    def _monitor_loop(self) -> None:
        while self._running.is_set():
            self.determine_leader()
            time.sleep(self.check_interval)

    def _probe_peer(self, address: tuple[str, int]) -> bool:
        client = RPCClient(
            address[0],
            address[1],
            client_id=f"leader-{self.local_peer_id}",
            timeout=self.rpc_timeout,
            retry_policy=None,
        )
        try:
            client.call("__ping__")
            return True
        except RPCError:
            return False

    def _select_lowest_alive_peer(self) -> int | None:
        failed = set(self.failure_detector.failed_peers())
        for peer_id in sorted(self.peer_addresses):
            if str(peer_id) not in failed:
                return peer_id
        return None
