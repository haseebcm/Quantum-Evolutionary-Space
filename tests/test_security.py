import threading
import time

import pytest

import qes.security as security_module
from qes.security import (
    AccessControlList,
    InvalidTokenError,
    Permission,
    PermissionDeniedError,
    QuotaExceededError,
    QuotaManager,
    Role,
    SandboxPolicy,
    SandboxViolationError,
    TokenExpiredError,
    TokenIssuer,
)


def test_token_issue_and_verify_round_trip():
    issuer = TokenIssuer(secret_key=b"unit-test-secret-key")
    token = issuer.issue(subject="tenant-a", ttl_seconds=60.0, scopes=["read", "write"])

    verified = issuer.verify(token)

    assert verified.subject == "tenant-a"
    assert verified.scopes == ("read", "write")
    assert verified.expires_at > verified.issued_at
    assert not verified.is_expired(now=verified.issued_at)


def test_token_tampering_is_rejected():
    issuer = TokenIssuer(secret_key=b"unit-test-secret-key")
    token = issuer.issue(subject="tenant-a", ttl_seconds=60.0, scopes=["read"])
    payload_hex, signature_hex = token.split(".")
    tampered_payload = ("0" if payload_hex[0] != "0" else "1") + payload_hex[1:]

    with pytest.raises(InvalidTokenError, match="signature mismatch"):
        issuer.verify(f"{tampered_payload}.{signature_hex}")


def test_expired_token_is_rejected():
    issuer = TokenIssuer(secret_key=b"unit-test-secret-key")
    token = issuer.issue(subject="tenant-a", ttl_seconds=0.05, scopes=["read"])

    time.sleep(0.08)

    with pytest.raises(TokenExpiredError, match="expired"):
        issuer.verify(token)


def test_quota_manager_tracks_usage_and_rejects_overage():
    manager = QuotaManager({"tenant-a": {"compute_seconds": 10.0, "memory_bytes": 1024.0}})

    manager.charge("tenant-a", "compute_seconds", 3.0)
    manager.charge("tenant-a", "memory_bytes", 512.0)
    report = manager.usage_report("tenant-a")

    assert report["consumed"] == {"compute_seconds": 3.0, "memory_bytes": 512.0}
    assert report["remaining"] == {"compute_seconds": 7.0, "memory_bytes": 512.0}

    with pytest.raises(QuotaExceededError, match="Quota exceeded"):
        manager.charge("tenant-a", "compute_seconds", 8.0)


def test_quota_manager_charge_is_thread_safe():
    n_threads = 8
    charges_per_thread = 200
    amount = 1.0
    expected_total = float(n_threads * charges_per_thread)
    manager = QuotaManager({"tenant-a": {"compute_seconds": expected_total}})
    start_event = threading.Event()
    errors: list[Exception] = []

    def worker() -> None:
        start_event.wait()
        try:
            for _ in range(charges_per_thread):
                manager.charge("tenant-a", "compute_seconds", amount)
        except Exception as exc:  # pragma: no cover - assertion captures unexpected errors.
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    start_event.set()
    for thread in threads:
        thread.join()

    assert errors == []
    report = manager.usage_report("tenant-a")
    assert report["consumed"]["compute_seconds"] == expected_total
    assert report["remaining"]["compute_seconds"] == 0.0


def test_sandbox_policy_accepts_valid_configuration():
    policy = SandboxPolicy(
        numeric_limits={"n_rooms": (1.0, 32.0), "array_size": (1.0, 1024.0), "iterations": (1.0, 5000.0)},
        forbidden_keys={"exec", "code"},
    )

    validated = policy.validate({"n_rooms": 8, "array_size": 128, "iterations": 2000, "label": "safe"})

    assert validated["n_rooms"] == 8
    assert validated["label"] == "safe"


def test_sandbox_policy_rejects_forbidden_key():
    policy = SandboxPolicy(numeric_limits={"n_rooms": (1.0, 32.0)}, forbidden_keys={"exec"})

    with pytest.raises(SandboxViolationError, match="forbidden"):
        policy.validate({"n_rooms": 8, "exec": "rm -rf ."})


def test_sandbox_policy_rejects_numeric_limit_violation():
    policy = SandboxPolicy(numeric_limits={"iterations": (1.0, 5000.0)})

    with pytest.raises(SandboxViolationError, match="violates bounds"):
        policy.validate({"iterations": 100000})


def test_sandbox_policy_rejects_non_numeric_value_for_numeric_key():
    policy = SandboxPolicy(numeric_limits={"array_size": (1.0, 1024.0)})

    with pytest.raises(SandboxViolationError, match="must be numeric"):
        policy.validate({"array_size": "large"})


def test_sandbox_policy_rejects_nested_container_values():
    policy = SandboxPolicy(numeric_limits={"n_rooms": (1.0, 32.0)})

    with pytest.raises(SandboxViolationError, match="scalar JSON value"):
        policy.validate({"n_rooms": 4, "plugins": ["unsafe-extension"]})


def test_access_control_list_allows_and_denies_expected_permissions():
    acl = AccessControlList(
        [
            Role(name="viewer", permissions=frozenset({"experiments:read"})),
            Role(name="operator", permissions=frozenset({"experiments:read", "experiments:run"})),
            Role(
                name="admin",
                permissions=frozenset({"experiments:read", "experiments:run", "admin:manage"}),
            ),
        ]
    )

    assert acl.check(["viewer"], "experiments:read")
    assert not acl.check(["viewer"], "experiments:run")
    assert acl.check(["operator"], "experiments:run")
    assert not acl.check(["operator"], "admin:manage")
    assert acl.check(["admin"], "admin:manage")


def test_access_control_list_require_raises_for_denied_permission():
    acl = AccessControlList([Role(name="viewer", permissions=frozenset({"experiments:read"}))])

    with pytest.raises(PermissionDeniedError, match="do not grant permission"):
        acl.require(["viewer"], "experiments:run")


def test_token_issuer_validates_secret_subject_ttl_and_scope_inputs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(security_module.secrets, "token_bytes", lambda size: b"k" * size)
    issuer = TokenIssuer()
    token = issuer.issue(subject="tenant-a", ttl_seconds=1.0, scopes=["read"])
    assert isinstance(token, str)

    with pytest.raises(TypeError, match="secret_key"):
        TokenIssuer(secret_key="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="secret_key"):
        TokenIssuer(secret_key=b"")
    with pytest.raises(TypeError, match="subject"):
        issuer.issue(subject=1, ttl_seconds=1.0, scopes=["read"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="subject"):
        issuer.issue(subject=" ", ttl_seconds=1.0, scopes=["read"])
    with pytest.raises(ValueError, match="ttl_seconds"):
        issuer.issue(subject="tenant-a", ttl_seconds=float("inf"), scopes=["read"])
    with pytest.raises(TypeError, match="scopes"):
        issuer.issue(subject="tenant-a", ttl_seconds=1.0, scopes="read")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scope"):
        issuer.issue(subject="tenant-a", ttl_seconds=1.0, scopes=["read", 1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="scope"):
        issuer.issue(subject="tenant-a", ttl_seconds=1.0, scopes=[""])


def test_token_verify_rejects_malformed_and_invalid_claims(monkeypatch: pytest.MonkeyPatch):
    issuer = TokenIssuer(secret_key=b"unit-test-secret-key")

    with pytest.raises(TypeError, match="token"):
        issuer.verify(123)  # type: ignore[arg-type]
    with pytest.raises(InvalidTokenError, match="must not be empty"):
        issuer.verify("")
    with pytest.raises(InvalidTokenError, match="separator"):
        issuer.verify("missing-separator")
    with pytest.raises(InvalidTokenError, match="payload"):
        issuer.verify("zz.00")

    payload_hex = b'{"sub":"tenant","iat":1.0,"exp":2.0,"scopes":["read"],"nonce":"n"}'.hex()
    with pytest.raises(InvalidTokenError, match="signature"):
        issuer.verify(f"{payload_hex}.zz")

    def signed(raw: bytes) -> str:
        return f"{raw.hex()}.{issuer._sign(raw).hex()}"

    with pytest.raises(InvalidTokenError, match="valid JSON"):
        issuer.verify(signed(b"\xff"))
    with pytest.raises(InvalidTokenError, match="object"):
        issuer.verify(signed(b"[]"))
    with pytest.raises(InvalidTokenError, match="subject"):
        issuer.verify(signed(b'{"sub":"","iat":1.0,"exp":2.0,"scopes":["read"],"nonce":"n"}'))
    with pytest.raises(InvalidTokenError, match="issued-at"):
        issuer.verify(signed(b'{"sub":"tenant","iat":"x","exp":2.0,"scopes":["read"],"nonce":"n"}'))
    with pytest.raises(InvalidTokenError, match="expiry claim"):
        issuer.verify(signed(b'{"sub":"tenant","iat":1.0,"exp":"x","scopes":["read"],"nonce":"n"}'))
    with pytest.raises(InvalidTokenError, match="later than issued-at"):
        issuer.verify(signed(b'{"sub":"tenant","iat":2.0,"exp":1.0,"scopes":["read"],"nonce":"n"}'))
    with pytest.raises(InvalidTokenError, match="scopes"):
        issuer.verify(signed(b'{"sub":"tenant","iat":1.0,"exp":2.0,"scopes":[1],"nonce":"n"}'))
    with pytest.raises(InvalidTokenError, match="nonce"):
        issuer.verify(signed(b'{"sub":"tenant","iat":1.0,"exp":2.0,"scopes":["read"],"nonce":""}'))

    monkeypatch.setattr(security_module.time, "time", lambda: 100.0)
    token = issuer.issue(subject="tenant-a", ttl_seconds=1.0, scopes=["read"])
    verified = issuer.verify(token)
    assert verified.is_expired(now=101.0)


def test_quota_manager_and_sandbox_policy_cover_validation_edges():
    manager = QuotaManager()
    with pytest.raises(TypeError, match="resource_limits"):
        manager.configure_tenant("tenant-a", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tenant_id"):
        manager.configure_tenant(1, {"cpu": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resource"):
        manager.configure_tenant("tenant-a", {" ": 1.0})
    with pytest.raises(ValueError, match="quota limit"):
        manager.configure_tenant("tenant-a", {"cpu": -1.0})
    with pytest.raises(TypeError, match="quota limit"):
        manager.configure_tenant("tenant-a", {"cpu": "bad"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="quota limit"):
        manager.configure_tenant("tenant-a", {"cpu": True})  # type: ignore[dict-item]

    manager.configure_tenant("tenant-a", {"cpu": 1.0})
    with pytest.raises(QuotaExceededError, match="No quota configured for tenant"):
        manager.charge("tenant-b", "cpu", 0.5)
    with pytest.raises(QuotaExceededError, match="resource"):
        manager.charge("tenant-a", "memory", 0.5)
    with pytest.raises(TypeError, match="amount"):
        manager.charge("tenant-a", "cpu", "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="amount"):
        manager.charge("tenant-a", "cpu", 0.0)
    with pytest.raises(KeyError, match="Unknown tenant"):
        manager.usage_report("tenant-b")

    with pytest.raises(TypeError, match="numeric_limits"):
        SandboxPolicy(numeric_limits=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        SandboxPolicy(numeric_limits={"cpu": [0.0, 1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="min > max"):
        SandboxPolicy(numeric_limits={"cpu": (2.0, 1.0)})

    policy = SandboxPolicy(numeric_limits={"cpu": (0.0, 1.0)}, forbidden_keys={"exec"})
    with pytest.raises(SandboxViolationError, match="must be a dict"):
        policy.validate([])  # type: ignore[arg-type]
    with pytest.raises(SandboxViolationError, match="keys must be non-empty strings"):
        policy.validate({"": 1})


def test_permission_role_and_acl_cover_validation_paths():
    Permission(name="experiments:read")
    with pytest.raises(ValueError, match="permission name"):
        Permission(name="")
    with pytest.raises(ValueError, match="role name"):
        Role(name="", permissions=frozenset({"experiments:read"}))
    with pytest.raises(ValueError, match="role permission"):
        Role(name="viewer", permissions=frozenset({" "}))

    acl = AccessControlList()
    with pytest.raises(TypeError, match="Role instance"):
        acl.add_role("viewer")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="subject_roles"):
        acl.check("viewer", "experiments:read")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="required_permission"):
        acl.check(["viewer"], "")
    with pytest.raises(ValueError, match="subject role"):
        acl.check([""], "experiments:read")

    acl.add_role(Role(name="viewer", permissions=frozenset({"experiments:read"})))
    acl.require(["viewer"], "experiments:read")
