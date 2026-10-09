import pytest

from qes.rpc_fault_tolerance import RPCClient, RPCRemoteError, RPCServer
from qes.security import TokenIssuer


def test_real_rpc_authorizes_before_dispatch():
    issuer = TokenIssuer(b"test-signing-key-32-bytes-long!!!!")
    server = RPCServer(token_issuer=issuer)
    calls = []
    server.register_handler("echo", lambda params: calls.append(params) or params)
    host, port = server.start()
    try:
        with pytest.raises(RPCRemoteError):
            RPCClient(host, port).call("echo", {"x": 1})
        wrong_scope = issuer.issue("alice", 30, ["rpc:other"])
        with pytest.raises(RPCRemoteError):
            RPCClient(host, port, token=wrong_scope).call("echo", {})
        token = issuer.issue("alice", 30, ["rpc:echo"])
        client = RPCClient(host, port, token=token)
        with pytest.raises(RPCRemoteError):
            client.call("echo", {"tenant_id": "bob"})
        assert calls == []
        assert client.call("echo", {"tenant_id": "alice", "x": 1})["x"] == 1
        assert len(calls) == 1
    finally:
        server.stop()


def test_request_retries_deduplicate_within_live_server_cache():
    server = RPCServer()
    effects = []
    server.register_handler("effect", lambda p: effects.append(p) or {"result": p})
    request = {"client_id": "trusted", "request_id": "id", "method": "effect", "params": 1}
    assert server._dispatch_request(request)["status"] == "ok"
    assert server._dispatch_request(request)["status"] == "ok"
    assert effects == [1]
    conflict = dict(request, params=2)
    assert server._dispatch_request(conflict)["status"] == "error"
    assert effects == [1]


def test_rpc_refuses_unprotected_remote_binding():
    with pytest.raises(ValueError, match="loopback"):
        RPCServer(host="0.0.0.0")
