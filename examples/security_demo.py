"""Demonstrate real security primitives for future QES multi-tenant services."""
from __future__ import annotations

from qes.security import (
    AccessControlList,
    InvalidTokenError,
    PermissionDeniedError,
    QuotaExceededError,
    QuotaManager,
    Role,
    SandboxPolicy,
    SandboxViolationError,
    TokenIssuer,
)


def main() -> None:
    """Run the security demo."""
    issuer = TokenIssuer(secret_key=b"demo-secret-key-for-qes-security")
    token = issuer.issue(subject="tenant-demo", ttl_seconds=60.0, scopes=["experiments:read"])
    verified = issuer.verify(token)
    print(
        "TOKEN VERIFIED:",
        {
            "subject": verified.subject,
            "scopes": verified.scopes,
            "expires_at": round(verified.expires_at, 6),
        },
    )

    payload_hex, signature_hex = token.split(".")
    tampered_payload = ("0" if payload_hex[0] != "0" else "1") + payload_hex[1:]
    try:
        issuer.verify(f"{tampered_payload}.{signature_hex}")
    except InvalidTokenError as exc:
        print(f"TOKEN TAMPERING REJECTED: {exc}")

    quotas = QuotaManager({"tenant-demo": {"compute_seconds": 5.0}})
    quotas.charge("tenant-demo", "compute_seconds", 3.0)
    print("QUOTA REPORT AFTER VALID CHARGE:", quotas.usage_report("tenant-demo"))
    try:
        quotas.charge("tenant-demo", "compute_seconds", 3.0)
    except QuotaExceededError as exc:
        print(f"QUOTA EXCEEDED AS EXPECTED: {exc}")

    policy = SandboxPolicy(
        numeric_limits={"n_rooms": (1.0, 16.0), "array_size": (1.0, 512.0), "iterations": (1.0, 2000.0)},
        forbidden_keys={"exec", "subprocess"},
    )
    safe_config = policy.validate({"n_rooms": 4, "array_size": 128, "iterations": 1000})
    print("SANDBOX ACCEPTED CONFIG:", safe_config)
    try:
        policy.validate({"n_rooms": 100, "array_size": 128, "iterations": 1000})
    except SandboxViolationError as exc:
        print(f"SANDBOX VIOLATION CAUGHT: {exc}")

    acl = AccessControlList(
        [
            Role(name="viewer", permissions=frozenset({"experiments:read"})),
            Role(name="operator", permissions=frozenset({"experiments:read", "experiments:run"})),
        ]
    )
    print("RBAC CHECK (operator -> experiments:run):", acl.check(["operator"], "experiments:run"))
    try:
        acl.require(["viewer"], "experiments:run")
    except PermissionDeniedError as exc:
        print(f"RBAC DENIAL CAUGHT: {exc}")


if __name__ == "__main__":
    main()
