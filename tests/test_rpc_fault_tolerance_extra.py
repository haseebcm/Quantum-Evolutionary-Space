import time

from qes.rpc_fault_tolerance import (
    FailureDetector,
    RetryPolicy,
    RPCClient,
    RPCConnectionError,
    RPCServer,
)


def test_failure_detector_register_and_check():
    detector = FailureDetector(heartbeat_timeout=0.05, check_interval=0.01)
    detector.register_peer("peer1")
    # initially not failed
    assert not detector.peer_failed("peer1")
    # simulate old last seen to force failure
    now = time.monotonic()
    with detector._lock:
        detector._last_seen["peer1"] = now - 1.0
    failed = detector.check_failures()
    assert "peer1" in failed
    snapshot = detector.snapshot()
    assert "peer1" in snapshot


def test_retry_policy_execute_with_transient_errors():
    calls = {"count": 0}

    def operation():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RPCConnectionError("transient")
        return "success"

    policy = RetryPolicy(max_attempts=5, base_delay=0.0, backoff_multiplier=1.0, jitter=0.0)
    result = policy.execute(operation)
    assert result == "success"
    assert calls["count"] == 3

    # exhausting attempts raises the original RPCConnectionError
    def always_fail():
        raise RPCConnectionError("dead")

    policy2 = RetryPolicy(max_attempts=2, base_delay=0.0, backoff_multiplier=1.0, jitter=0.0)
    import pytest

    with pytest.raises(RPCConnectionError):
        policy2.execute(always_fail)


def test_rpc_server_client_call_and_heartbeat():
    server = RPCServer(socket_timeout=0.05, heartbeat_timeout=0.05, detector_check_interval=0.01)
    try:
        addr = server.start()
        host, port = addr

        # simple echo handler
        server.register_handler("echo", lambda params: {"echo": params})

        client = RPCClient(host, port, client_id="test-client", timeout=0.1)
        # call the echo handler
        result = client.call("echo", {"x": 1})
        assert result == {"echo": {"x": 1}}

        # heartbeat handling
        server.register_client("test-client")
        client.send_heartbeat()
        # small sleep to allow server to process
        time.sleep(0.12)
        snap = server.client_snapshot()
        assert "test-client" in snap
        # heartbeat processing timing is best-effort; ensure a last_heartbeat exists
        assert snap["test-client"]["last_heartbeat"] is not None
    finally:
        server.stop()
