"""Tests for real localhost RPC and fault-tolerance helpers."""
from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Mapping
from typing import Any

import pytest

from qes.rpc_fault_tolerance import (
    LeaderElection,
    RetryPolicy,
    RPCClient,
    RPCRemoteError,
    RPCServer,
)

_MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def _send_message(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(data) > _MAX_MESSAGE_BYTES:
        raise ValueError("message exceeds maximum supported size")
    sock.sendall(len(data).to_bytes(4, byteorder="big", signed=False))
    sock.sendall(data)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise EOFError("socket closed before full payload arrived")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_message(sock: socket.socket) -> dict[str, Any]:
    size = int.from_bytes(_recv_exact(sock, 4), byteorder="big", signed=False)
    raw_payload = _recv_exact(sock, size)
    payload = json.loads(raw_payload.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    return payload


def wait_until(predicate: Any, *, timeout: float = 1.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class FlakyTransportServer:
    """Real socket server that drops the first N connections before succeeding."""

    def __init__(self, failures_before_success: int, *, socket_timeout: float = 0.05) -> None:
        self.failures_before_success = failures_before_success
        self.socket_timeout = socket_timeout
        self.attempts = 0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(("127.0.0.1", 0))
        self._server_socket.listen()
        self._server_socket.settimeout(self.socket_timeout)
        self._thread = threading.Thread(target=self._serve, name="flaky-transport-server", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server_socket.getsockname()[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._running.set()
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        try:
            self._server_socket.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)
        assert not self._thread.is_alive()

    def _serve(self) -> None:
        while self._running.is_set():
            try:
                client_socket, _address = self._server_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with client_socket:
                client_socket.settimeout(self.socket_timeout)
                request = _recv_message(client_socket)
                with self._lock:
                    self.attempts += 1
                    attempt_number = self.attempts
                if attempt_number <= self.failures_before_success:
                    continue
                _send_message(
                    client_socket,
                    {
                        "request_id": request.get("request_id", ""),
                        "status": "ok",
                        "result": {"attempt": attempt_number, "ok": True},
                    },
                )


@pytest.fixture
def rpc_server() -> Any:
    server = RPCServer(
        server_id="test-server",
        socket_timeout=0.05,
        heartbeat_timeout=0.12,
        detector_check_interval=0.02,
    )
    server.start()
    try:
        yield server
    finally:
        server.stop()


def test_rpc_round_trip_returns_real_result(rpc_server: RPCServer) -> None:
    rpc_server.register_handler("add", lambda params: params["a"] + params["b"])
    host, port = rpc_server.address
    client = RPCClient(host, port, client_id="roundtrip-client", timeout=0.1)

    result = client.call("add", {"a": 2, "b": 5})

    assert result == 7


def test_rpc_round_trip_returns_real_error_response(rpc_server: RPCServer) -> None:
    def explode(_params: Any) -> Any:
        raise ValueError("bad payload")

    rpc_server.register_handler("explode", explode)
    host, port = rpc_server.address
    client = RPCClient(host, port, client_id="error-client", timeout=0.1)

    with pytest.raises(RPCRemoteError, match="ValueError: bad payload"):
        client.call("explode", {"x": 1})


def test_server_failure_detector_marks_client_failed_after_heartbeats_stop(rpc_server: RPCServer) -> None:
    rpc_server.register_client("heartbeat-client")
    host, port = rpc_server.address
    client = RPCClient(host, port, client_id="heartbeat-client", timeout=0.05)
    client.start_heartbeats(0.03)
    try:
        assert wait_until(
            lambda: "heartbeat-client" in rpc_server.client_snapshot()
            and rpc_server.failure_detector.peer_failed("heartbeat-client") is False,
            timeout=0.4,
        )
    finally:
        client.stop_heartbeats()

    assert wait_until(
        lambda: rpc_server.failure_detector.peer_failed("heartbeat-client"),
        timeout=0.5,
    )


def test_retry_policy_retries_real_socket_failures_with_backoff() -> None:
    flaky_server = FlakyTransportServer(failures_before_success=2)
    flaky_server.start()
    try:
        host, port = flaky_server.address
        client = RPCClient(
            host,
            port,
            client_id="retry-client",
            timeout=0.05,
            retry_policy=RetryPolicy(
                max_attempts=4,
                base_delay=0.02,
                backoff_multiplier=2.0,
                jitter=0.0,
            ),
        )

        started_at = time.monotonic()
        result = client.call("unstable", {"payload": "hello"})
        elapsed = time.monotonic() - started_at
    finally:
        flaky_server.stop()

    assert result == {"attempt": 3, "ok": True}
    assert flaky_server.attempts == 3
    assert elapsed >= 0.06


def test_leader_election_fails_over_to_next_lowest_live_peer() -> None:
    servers = [
        RPCServer(
            port=0,
            server_id="1",
            socket_timeout=0.05,
            heartbeat_timeout=0.12,
            detector_check_interval=0.02,
        ),
        RPCServer(
            port=0,
            server_id="2",
            socket_timeout=0.05,
            heartbeat_timeout=0.12,
            detector_check_interval=0.02,
        ),
        RPCServer(
            port=0,
            server_id="3",
            socket_timeout=0.05,
            heartbeat_timeout=0.12,
            detector_check_interval=0.02,
        ),
    ]
    elections: list[LeaderElection] = []
    try:
        addresses: dict[int, tuple[str, int]] = {}
        for index, server in enumerate(servers, start=1):
            addresses[index] = server.start()
        elections = [
            LeaderElection(
                peer_id,
                addresses,
                heartbeat_timeout=0.12,
                check_interval=0.03,
                rpc_timeout=0.05,
            )
            for peer_id in sorted(addresses)
        ]
        for election in elections:
            election.start()

        assert wait_until(lambda: all(election.current_leader() == 1 for election in elections), timeout=0.6)

        elections[0].stop()
        servers[0].stop()

        assert wait_until(
            lambda: all(election.current_leader() == 2 for election in elections[1:]),
            timeout=0.8,
        )
    finally:
        for election in reversed(elections):
            election.stop()
        for server in reversed(servers):
            server.stop()
