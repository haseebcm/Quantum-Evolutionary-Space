"""Demo for real localhost RPC, retries, and leader failover in QES.

This example uses real TCP sockets and real wall-clock timing on one machine.
It demonstrates behavior that has been tested only on localhost with a handful
of peers; it does not claim multi-machine or large-scale production readiness.
"""
from __future__ import annotations

import threading
import time

from qes.rpc_fault_tolerance import LeaderElection, RetryPolicy, RPCClient, RPCServer


def wait_until(predicate, *, timeout: float, interval: float = 0.01) -> bool:
    """Poll a predicate until it becomes true or a timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def run_rpc_demo() -> None:
    """Start a real RPC server and make several real client calls."""
    server = RPCServer(server_id="demo-primary", socket_timeout=0.05, heartbeat_timeout=0.2)
    server.register_handler("add", lambda params: params["a"] + params["b"])
    server.register_handler("echo", lambda params: {"echo": params})
    server.start()
    host, port = server.address
    client = RPCClient(host, port, client_id="demo-client", timeout=0.08)

    print("RPC demo")
    print("-" * 40)
    print("add(2, 5) ->", client.call("add", {"a": 2, "b": 5}))
    print("echo({'message': 'hello'}) ->", client.call("echo", {"message": "hello"}))

    replacement_ready = threading.Event()
    replacement_started = threading.Event()

    def restart_server() -> None:
        time.sleep(0.12)
        replacement = RPCServer(
            host=host,
            port=port,
            server_id="demo-replacement",
            socket_timeout=0.05,
            heartbeat_timeout=0.2,
        )
        replacement.register_handler("add", lambda params: params["a"] + params["b"])
        replacement.start()
        replacement_started.set()
        replacement_ready.wait(1.0)
        replacement.stop()

    retry_thread = threading.Thread(target=restart_server, name="demo-server-restart", daemon=True)
    retry_thread.start()

    server.stop()
    resilient_client = RPCClient(
        host,
        port,
        client_id="demo-client",
        timeout=0.05,
        retry_policy=RetryPolicy(max_attempts=6, base_delay=0.03, backoff_multiplier=1.6, jitter=0.0),
    )
    started_at = time.monotonic()
    recovered_result = resilient_client.call("add", {"a": 10, "b": 7})
    elapsed = time.monotonic() - started_at
    print(f"retry after server restart -> {recovered_result} (elapsed {elapsed:.3f}s)")

    replacement_ready.set()
    retry_thread.join(timeout=1.0)
    if retry_thread.is_alive() or not replacement_started.is_set():
        raise RuntimeError("replacement server thread did not finish cleanly")


def run_leader_demo() -> None:
    """Run a three-peer localhost leader-election scenario."""
    servers = [
        RPCServer(port=0, server_id="1", socket_timeout=0.05, heartbeat_timeout=0.12),
        RPCServer(port=0, server_id="2", socket_timeout=0.05, heartbeat_timeout=0.12),
        RPCServer(port=0, server_id="3", socket_timeout=0.05, heartbeat_timeout=0.12),
    ]
    elections: list[LeaderElection] = []
    try:
        addresses = {index: server.start() for index, server in enumerate(servers, start=1)}
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

        if not wait_until(lambda: all(election.current_leader() == 1 for election in elections), timeout=0.6):
            raise RuntimeError("initial leader election did not converge")
        print("\nLeader election demo")
        print("-" * 40)
        print("initial leader:", elections[1].current_leader())

        failover_started = time.monotonic()
        elections[0].stop()
        servers[0].stop()

        if not wait_until(
            lambda: all(election.current_leader() == 2 for election in elections[1:]),
            timeout=0.8,
        ):
            raise RuntimeError("leader failover did not converge")
        failover_elapsed = time.monotonic() - failover_started
        print(f"leader after peer 1 stops: {elections[1].current_leader()} (elapsed {failover_elapsed:.3f}s)")
    finally:
        for election in reversed(elections):
            election.stop()
        for server in reversed(servers):
            server.stop()


def main() -> None:
    """Run both the RPC/retry and leader-election demos."""
    run_rpc_demo()
    run_leader_demo()


if __name__ == "__main__":
    main()
