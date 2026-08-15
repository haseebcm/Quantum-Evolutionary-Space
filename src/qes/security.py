"""Security primitives for future multi-tenant QES services.

This module provides real, unit-tested logic for:

* HMAC-SHA256 bearer-token issuance and verification
* per-tenant resource-quota accounting
* sandbox-style configuration validation
* simple role-based access control (RBAC)

These are real building blocks suitable for a future multi-tenant deployment
of the classical NumPy-based QES framework, but they are not themselves a
network-facing authentication service, not a deployed multi-tenant system, and
not an integration with production key-management infrastructure such as a KMS
or HSM.

The RBAC ``Permission`` type defined here is unrelated to ``qes.permission``,
which models admissibility gates for room-state evolution. The shared word
"permission" is coincidental; the concepts are separate.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import TypedDict, TypeGuard


class SecurityError(Exception):
    """Base class for security-module errors."""


class InvalidTokenError(SecurityError):
    """Raised when a bearer token is malformed or fails signature validation."""


class TokenExpiredError(InvalidTokenError):
    """Raised when a bearer token is well-formed but already expired."""


class QuotaExceededError(SecurityError):
    """Raised when a resource charge would exceed the tenant's configured quota."""


class SandboxViolationError(SecurityError):
    """Raised when untrusted configuration violates sandbox policy constraints."""


class PermissionDeniedError(SecurityError):
    """Raised when an RBAC permission check fails."""


class _TokenClaims(TypedDict):
    sub: str
    iat: float
    exp: float
    scopes: list[str]
    nonce: str


@dataclass(frozen=True)
class AuthToken:
    """Verified bearer-token claims extracted from a signed token."""

    subject: str
    issued_at: float
    expires_at: float
    scopes: tuple[str, ...]
    nonce: str

    def is_expired(self, now: float | None = None) -> bool:
        """Return ``True`` when the token is expired at the supplied wall-clock time."""
        current_time = time.time() if now is None else now
        return current_time >= self.expires_at


class TokenIssuer:
    """Issue and verify real HMAC-SHA256-signed bearer tokens.

    The token format is intentionally simple rather than a JWT reimplementation:
    ``<hex-encoded-json-payload>.<hex-encoded-hmac-signature>``.
    """

    def __init__(self, secret_key: bytes | None = None):
        """Initialize the issuer with a real secret key.

        Args:
            secret_key: Secret HMAC key. If omitted, a new 32-byte random key is
                generated with ``secrets.token_bytes(32)``.
        """
        if secret_key is None:
            secret_key = secrets.token_bytes(32)
        if not isinstance(secret_key, bytes):
            raise TypeError("secret_key must be bytes")
        if not secret_key:
            raise ValueError("secret_key must not be empty")
        self._secret_key = secret_key

    def issue(self, subject: str, ttl_seconds: float, scopes: list[str]) -> str:
        """Issue a signed bearer token for a subject with scopes and a TTL."""
        if not isinstance(subject, str):
            raise TypeError("subject must be a string")
        if not subject.strip():
            raise ValueError("subject must not be empty")

        ttl = float(ttl_seconds)
        if ttl <= 0 or ttl != ttl or ttl == float("inf"):
            raise ValueError("ttl_seconds must be a finite positive number")

        normalized_scopes = self._normalize_scopes(scopes)
        issued_at = time.time()
        expires_at = issued_at + ttl
        payload = {
            "exp": expires_at,
            "iat": issued_at,
            "nonce": secrets.token_hex(16),
            "scopes": normalized_scopes,
            "sub": subject,
        }
        payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = self._sign(payload_bytes)
        return f"{payload_bytes.hex()}.{signature.hex()}"

    def verify(self, token: str) -> AuthToken:
        """Verify a bearer token and return the authenticated claims."""
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        if not token:
            raise InvalidTokenError("Token must not be empty")

        payload_hex, signature_hex = self._split_token(token)
        payload_bytes = self._decode_hex_component(payload_hex, "payload")
        provided_signature = self._decode_hex_component(signature_hex, "signature")
        expected_signature = self._sign(payload_bytes)

        if not hmac.compare_digest(provided_signature, expected_signature):
            raise InvalidTokenError("Token signature mismatch")

        claims = self._decode_payload(payload_bytes)
        token_data = AuthToken(
            subject=claims["sub"],
            issued_at=claims["iat"],
            expires_at=claims["exp"],
            scopes=tuple(claims["scopes"]),
            nonce=claims["nonce"],
        )
        if token_data.is_expired():
            raise TokenExpiredError(f"Token expired at {token_data.expires_at:.6f}")
        return token_data

    def _sign(self, payload_bytes: bytes) -> bytes:
        return hmac.new(self._secret_key, payload_bytes, hashlib.sha256).digest()

    @staticmethod
    def _split_token(token: str) -> tuple[str, str]:
        parts = token.split(".")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise InvalidTokenError("Token must contain exactly one payload/signature separator")
        return parts[0], parts[1]

    @staticmethod
    def _decode_hex_component(component: str, label: str) -> bytes:
        try:
            return bytes.fromhex(component)
        except ValueError as exc:
            raise InvalidTokenError(f"Token {label} is not valid hexadecimal") from exc

    @staticmethod
    def _normalize_scopes(scopes: list[str]) -> list[str]:
        if not isinstance(scopes, list):
            raise TypeError("scopes must be a list of strings")

        normalized: list[str] = []
        for scope in scopes:
            if not isinstance(scope, str):
                raise TypeError("each scope must be a string")
            if not scope.strip():
                raise ValueError("scope values must not be empty")
            normalized.append(scope)
        return normalized

    @staticmethod
    def _decode_payload(payload_bytes: bytes) -> _TokenClaims:
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidTokenError("Token payload is not valid JSON") from exc

        if not isinstance(payload, dict):
            raise InvalidTokenError("Token payload must decode to an object")

        subject = payload.get("sub")
        issued_at = payload.get("iat")
        expires_at = payload.get("exp")
        scopes = payload.get("scopes")
        nonce = payload.get("nonce")

        if not isinstance(subject, str) or not subject.strip():
            raise InvalidTokenError("Token subject is missing or invalid")
        if not _is_real_number(issued_at):
            raise InvalidTokenError("Token issued-at claim is missing or invalid")
        if not _is_real_number(expires_at):
            raise InvalidTokenError("Token expiry claim is missing or invalid")
        if float(expires_at) <= float(issued_at):
            raise InvalidTokenError("Token expiry must be later than issued-at time")
        if not isinstance(scopes, list) or any(
            not isinstance(scope, str) or not scope.strip() for scope in scopes
        ):
            raise InvalidTokenError("Token scopes claim is missing or invalid")
        if not isinstance(nonce, str) or not nonce:
            raise InvalidTokenError("Token nonce is missing or invalid")

        return {
            "sub": subject,
            "iat": float(issued_at),
            "exp": float(expires_at),
            "scopes": scopes,
            "nonce": nonce,
        }


@dataclass
class ResourceQuota:
    """Per-tenant quota state for named resources."""

    allotted: dict[str, float] = field(default_factory=dict)
    consumed: dict[str, float] = field(default_factory=dict)


class QuotaManager:
    """Thread-safe per-tenant resource accounting for future service deployment."""

    def __init__(self, tenant_quotas: dict[str, dict[str, float]] | None = None):
        """Initialize quota state from ``tenant_id -> resource -> limit`` mappings."""
        self._lock = threading.Lock()
        self._quotas: dict[str, ResourceQuota] = {}
        for tenant_id, resource_limits in (tenant_quotas or {}).items():
            self.configure_tenant(tenant_id, resource_limits)

    def configure_tenant(self, tenant_id: str, resource_limits: dict[str, float]) -> None:
        """Configure or replace the quota limits for one tenant."""
        _validate_identifier(tenant_id, "tenant_id")
        if not isinstance(resource_limits, dict):
            raise TypeError("resource_limits must be a dict[str, float]")

        normalized_limits: dict[str, float] = {}
        for resource, limit in resource_limits.items():
            _validate_identifier(resource, "resource")
            normalized_limit = _coerce_non_negative_number(limit, f"quota limit for {resource!r}")
            normalized_limits[resource] = normalized_limit

        with self._lock:
            self._quotas[tenant_id] = ResourceQuota(
                allotted=normalized_limits,
                consumed={resource: 0.0 for resource in normalized_limits},
            )

    def charge(self, tenant_id: str, resource: str, amount: float) -> None:
        """Charge resource usage to a tenant or raise ``QuotaExceededError``."""
        _validate_identifier(tenant_id, "tenant_id")
        _validate_identifier(resource, "resource")
        normalized_amount = _coerce_positive_number(amount, "amount")

        with self._lock:
            quota = self._quotas.get(tenant_id)
            if quota is None:
                raise QuotaExceededError(f"No quota configured for tenant {tenant_id!r}")

            limit = quota.allotted.get(resource)
            if limit is None:
                raise QuotaExceededError(
                    f"No quota configured for tenant {tenant_id!r} and resource {resource!r}"
                )

            current = quota.consumed.get(resource, 0.0)
            new_total = current + normalized_amount
            if new_total > limit:
                raise QuotaExceededError(
                    f"Quota exceeded for tenant {tenant_id!r}, resource {resource!r}: "
                    f"{new_total} > {limit}"
                )
            quota.consumed[resource] = new_total

    def usage_report(self, tenant_id: str) -> dict[str, object]:
        """Return the current allotted, consumed, and remaining quota for a tenant."""
        _validate_identifier(tenant_id, "tenant_id")

        with self._lock:
            quota = self._quotas.get(tenant_id)
            if quota is None:
                raise KeyError(f"Unknown tenant {tenant_id!r}")

            allotted = dict(quota.allotted)
            consumed = dict(quota.consumed)
            remaining = {
                resource: allotted[resource] - consumed.get(resource, 0.0) for resource in allotted
            }
        return {
            "tenant_id": tenant_id,
            "allotted": allotted,
            "consumed": consumed,
            "remaining": remaining,
        }


class SandboxPolicy:
    """Validate untrusted tenant configuration against enforceable safety limits."""

    def __init__(
        self,
        numeric_limits: dict[str, tuple[float, float]],
        forbidden_keys: set[str] | None = None,
    ):
        """Create a policy with inclusive numeric bounds and forbidden keys."""
        if not isinstance(numeric_limits, dict):
            raise TypeError("numeric_limits must be a dict[str, tuple[float, float]]")

        normalized_limits: dict[str, tuple[float, float]] = {}
        for key, bounds in numeric_limits.items():
            _validate_identifier(key, "numeric limit key")
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise TypeError(f"numeric limit for {key!r} must be a (min, max) tuple")
            min_value = _coerce_non_negative_number(bounds[0], f"minimum bound for {key!r}")
            max_value = _coerce_non_negative_number(bounds[1], f"maximum bound for {key!r}")
            if min_value > max_value:
                raise ValueError(f"numeric limit for {key!r} has min > max")
            normalized_limits[key] = (min_value, max_value)

        self.numeric_limits = normalized_limits
        self.forbidden_keys = {
            key for key in (forbidden_keys or set()) if isinstance(key, str) and key.strip()
        }

    def validate(self, config: dict[str, object]) -> dict[str, object]:
        """Validate and return a shallow copy of a tenant configuration."""
        if not isinstance(config, dict):
            raise SandboxViolationError("Sandboxed configuration must be a dict")

        validated: dict[str, object] = {}
        for key, value in config.items():
            if not isinstance(key, str) or not key.strip():
                raise SandboxViolationError("Sandboxed configuration keys must be non-empty strings")
            if key in self.forbidden_keys:
                raise SandboxViolationError(f"Configuration key {key!r} is forbidden by sandbox policy")
            if isinstance(value, (dict, list, tuple, set, bytes, bytearray)):
                raise SandboxViolationError(
                    f"Configuration key {key!r} must be a scalar JSON value, not {type(value).__name__}"
                )
            if key in self.numeric_limits:
                if not _is_real_number(value):
                    raise SandboxViolationError(
                        f"Configuration key {key!r} must be numeric, got {type(value).__name__}"
                    )
                number_value = float(value)
                min_value, max_value = self.numeric_limits[key]
                if number_value < min_value or number_value > max_value:
                    raise SandboxViolationError(
                        f"Configuration key {key!r}={number_value} violates bounds "
                        f"[{min_value}, {max_value}]"
                    )
            validated[key] = value
        return validated


@dataclass(frozen=True)
class Permission:
    """A named RBAC permission."""

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        """Validate permission metadata."""
        _validate_identifier(self.name, "permission name")


@dataclass(frozen=True)
class Role:
    """An RBAC role and the permissions it grants."""

    name: str
    permissions: frozenset[str]

    def __post_init__(self) -> None:
        """Validate role metadata and permission names."""
        _validate_identifier(self.name, "role name")
        for permission in self.permissions:
            _validate_identifier(permission, "role permission")


class AccessControlList:
    """Simple role-based access control evaluator."""

    def __init__(self, roles: list[Role] | None = None):
        """Initialize the ACL from a list of roles."""
        self._roles: dict[str, frozenset[str]] = {}
        for role in roles or []:
            self.add_role(role)

    def add_role(self, role: Role) -> None:
        """Register or replace one role definition."""
        if not isinstance(role, Role):
            raise TypeError("role must be a Role instance")
        self._roles[role.name] = role.permissions

    def check(self, subject_roles: list[str], required_permission: str) -> bool:
        """Return ``True`` iff any subject role grants the required permission."""
        if not isinstance(subject_roles, list):
            raise TypeError("subject_roles must be a list of strings")
        _validate_identifier(required_permission, "required_permission")

        for role_name in subject_roles:
            _validate_identifier(role_name, "subject role")
            if required_permission in self._roles.get(role_name, frozenset()):
                return True
        return False

    def require(self, subject_roles: list[str], required_permission: str) -> None:
        """Require an RBAC permission or raise ``PermissionDeniedError``."""
        if not self.check(subject_roles, required_permission):
            raise PermissionDeniedError(
                f"Roles {subject_roles!r} do not grant permission {required_permission!r}"
            )


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must not be empty")


def _coerce_positive_number(value: object, label: str) -> float:
    if not _is_real_number(value):
        raise TypeError(f"{label} must be a real number")
    number_value = float(value)
    if number_value <= 0:
        raise ValueError(f"{label} must be > 0")
    return number_value


def _coerce_non_negative_number(value: object, label: str) -> float:
    if not _is_real_number(value):
        raise TypeError(f"{label} must be a real number")
    number_value = float(value)
    if number_value < 0:
        raise ValueError(f"{label} must be >= 0")
    return number_value


def _is_real_number(value: object) -> TypeGuard[int | float]:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    number_value = float(value)
    return number_value == number_value and number_value not in (float("inf"), float("-inf"))
