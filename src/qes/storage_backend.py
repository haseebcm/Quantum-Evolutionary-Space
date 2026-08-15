"""Object-storage backends for reproducibility checkpoints and artifacts.

This module provides one fully working backend for local filesystem storage and
one optional S3 backend that uses real boto3 API calls when boto3 and valid AWS
configuration are available. In this session, only the local filesystem backend
was exercised end-to-end. The S3 backend code was mock-tested for API contract
coverage but was not executed against a real S3 bucket here because this
environment does not have boto3 installed or real AWS credentials configured.
"""
from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class StorageBackend(ABC):
    """Abstract object-storage interface used by higher-level QES persistence."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Persist raw bytes under ``key``."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Load raw bytes previously stored under ``key``."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove the object stored under ``key``."""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """Return sorted stored keys whose names start with ``prefix``."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether an object exists at ``key``."""


def _normalize_storage_path(value: str, *, allow_empty: bool) -> str:
    """Validate and normalize a storage key or prefix.

    Keys use forward-slash namespace separators regardless of platform. Empty
    prefixes are allowed for listing all objects; empty keys are rejected.
    """
    if not isinstance(value, str):
        raise TypeError("storage key must be a string")

    if value == "":
        if allow_empty:
            return ""
        raise ValueError("storage key must be a non-empty string")

    if "\x00" in value:
        raise ValueError("storage key must not contain NUL characters")

    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute() or windows_path.drive:
        raise ValueError("storage key must be relative and must not contain a drive prefix")
    if value.startswith(("/", "\\")):
        raise ValueError("storage key must be relative and must not start with a path separator")

    trailing_separator = value.endswith(("/", "\\"))
    normalized_input = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized_input)
    if posix_path.is_absolute():
        raise ValueError("storage key must be relative and must not be absolute")

    parts = [part for part in posix_path.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError("storage key must not contain '..' path traversal segments")
    if not parts:
        if allow_empty:
            return ""
        raise ValueError("storage key must include at least one non-empty path segment")

    normalized = "/".join(parts)
    if allow_empty and trailing_separator:
        return f"{normalized}/"
    return normalized


def _expected_digest_from_key(key: str) -> str:
    """Extract the expected SHA-256 digest encoded in a checkpoint key."""
    normalized = _normalize_storage_path(key, allow_empty=False)
    digest = normalized.rsplit("/", 1)[-1]
    if digest.endswith(".json"):
        digest = digest[:-5]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"checkpoint key {key!r} does not end with a 64-character SHA-256 digest")
    return digest


class LocalFilesystemStorageBackend(StorageBackend):
    """Real storage backend that persists objects under a configured directory."""

    def __init__(self, root_directory: str | Path) -> None:
        """Create a backend rooted at ``root_directory``."""
        self._root_directory = Path(root_directory).expanduser().resolve()
        self._root_directory.mkdir(parents=True, exist_ok=True)

    @property
    def root_directory(self) -> Path:
        """Return the filesystem directory used to store objects."""
        return self._root_directory

    def put(self, key: str, data: bytes) -> None:
        """Persist ``data`` to a real file under the configured root directory."""
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        """Read object bytes from the configured root directory."""
        path = self._path_for_key(key)
        if not path.is_file():
            raise FileNotFoundError(f"storage key {key!r} does not exist under {self._root_directory}")
        return path.read_bytes()

    def delete(self, key: str) -> None:
        """Delete the file backing ``key`` and prune empty parent directories."""
        path = self._path_for_key(key)
        if not path.is_file():
            raise FileNotFoundError(f"storage key {key!r} does not exist under {self._root_directory}")
        path.unlink()
        self._prune_empty_directories(path.parent)

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return sorted relative object keys stored under this backend."""
        normalized_prefix = _normalize_storage_path(prefix, allow_empty=True)
        keys = [
            path.relative_to(self._root_directory).as_posix()
            for path in self._root_directory.rglob("*")
            if path.is_file()
        ]
        return sorted(key for key in keys if key.startswith(normalized_prefix))

    def exists(self, key: str) -> bool:
        """Return whether ``key`` currently maps to a stored file."""
        return self._path_for_key(key).is_file()

    def _path_for_key(self, key: str) -> Path:
        normalized = _normalize_storage_path(key, allow_empty=False)
        candidate = self._root_directory.joinpath(*normalized.split("/")).resolve()
        try:
            candidate.relative_to(self._root_directory)
        except ValueError as error:
            raise ValueError(f"storage key {key!r} resolves outside the configured root directory") from error
        return candidate

    def _prune_empty_directories(self, directory: Path) -> None:
        current = directory
        while current != self._root_directory:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent


class S3StorageBackend(StorageBackend):
    """Optional S3 backend implemented with the real boto3 S3 client API.

    This class issues genuine boto3-style calls such as ``put_object``,
    ``get_object``, ``delete_object``, ``list_objects_v2``, and ``head_object``.
    In this session, the code was only reviewed and mock-tested against an
    injected fake client; it was not executed against a real S3 bucket because
    boto3 is not installed here and no AWS credentials are configured on this
    machine. Use dependency injection or mocking in tests, and run separate
    integration tests in an environment with a real bucket before claiming
    end-to-end S3 validation.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region_name: str | None = None,
        s3_client: Any | None = None,
    ) -> None:
        """Create an S3-backed storage backend for one bucket/prefix namespace."""
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("bucket must be a non-empty string")
        self._bucket = bucket.strip()
        self._prefix = _normalize_storage_path(prefix, allow_empty=True)
        if s3_client is None:
            try:
                import boto3
            except ImportError as error:
                raise ImportError(
                    "S3StorageBackend requires the optional 'boto3' package when no "
                    "client is injected. Install it with `pip install boto3`."
                ) from error
            self._client = boto3.client("s3", region_name=region_name)
        else:
            self._client = s3_client

    @property
    def bucket(self) -> str:
        """Return the bucket name used by this backend."""
        return self._bucket

    @property
    def prefix(self) -> str:
        """Return the optional S3 key prefix namespace for this backend."""
        return self._prefix

    def put(self, key: str, data: bytes) -> None:
        """Upload raw bytes to S3 using ``put_object``."""
        object_key = self._object_key(key)
        self._client.put_object(Bucket=self._bucket, Key=object_key, Body=data)

    def get(self, key: str) -> bytes:
        """Download raw bytes from S3 using ``get_object``."""
        object_key = self._object_key(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=object_key)
        except Exception as error:
            if self._is_missing_error(error):
                raise FileNotFoundError(
                    f"storage key {key!r} does not exist in S3 bucket {self._bucket!r}"
                ) from error
            raise
        body = response["Body"]
        return bytes(body.read())

    def delete(self, key: str) -> None:
        """Delete an object from S3 after verifying that it exists."""
        object_key = self._object_key(key)
        if not self.exists(key):
            raise FileNotFoundError(f"storage key {key!r} does not exist in S3 bucket {self._bucket!r}")
        self._client.delete_object(Bucket=self._bucket, Key=object_key)

    def list_keys(self, prefix: str = "") -> list[str]:
        """List stored keys by repeatedly calling ``list_objects_v2``."""
        normalized_prefix = _normalize_storage_path(prefix, allow_empty=True)
        query_prefix = self._object_prefix(normalized_prefix)
        keys: list[str] = []
        continuation_token: str | None = None

        while True:
            request: dict[str, Any] = {"Bucket": self._bucket, "Prefix": query_prefix}
            if continuation_token is not None:
                request["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**request)
            for item in response.get("Contents", []):
                full_key = str(item["Key"])
                logical_key = self._logical_key(full_key)
                if logical_key is not None:
                    keys.append(logical_key)
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            if continuation_token is None:
                break
        return sorted(keys)

    def exists(self, key: str) -> bool:
        """Return whether an object exists by calling ``head_object``."""
        object_key = self._object_key(key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=object_key)
        except Exception as error:
            if self._is_missing_error(error):
                return False
            raise
        return True

    def _object_key(self, key: str) -> str:
        normalized = _normalize_storage_path(key, allow_empty=False)
        if not self._prefix:
            return normalized
        return f"{self._prefix.rstrip('/')}/{normalized}"

    def _logical_key(self, object_key: str) -> str | None:
        if self._prefix:
            expected_prefix = f"{self._prefix.rstrip('/')}/"
            if object_key == self._prefix.rstrip("/"):
                return None
            if not object_key.startswith(expected_prefix):
                return None
            return object_key[len(expected_prefix) :]
        return object_key

    def _object_prefix(self, prefix: str) -> str:
        if not prefix:
            return self._prefix
        normalized = _normalize_storage_path(prefix, allow_empty=True)
        trailing_separator = normalized.endswith("/")
        normalized_key = normalized.rstrip("/")
        object_key = self._object_key(normalized_key)
        if trailing_separator:
            return f"{object_key}/"
        return object_key

    def _is_missing_error(self, error: Exception) -> bool:
        response = getattr(error, "response", None)
        if not isinstance(response, Mapping):
            return False
        error_payload = response.get("Error")
        if not isinstance(error_payload, Mapping):
            return False
        code = str(error_payload.get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound"}


class CheckpointStore:
    """Save and load JSON checkpoints with SHA-256-addressed storage keys."""

    def __init__(self, backend: StorageBackend, *, namespace: str = "checkpoints") -> None:
        """Create a checkpoint store on top of ``backend``."""
        self._backend = backend
        self._namespace = _normalize_storage_path(namespace, allow_empty=False)

    @property
    def namespace(self) -> str:
        """Return the namespace prefix used for checkpoint keys."""
        return self._namespace

    def save_checkpoint(self, checkpoint: Mapping[str, Any]) -> str:
        """Serialize and store one JSON checkpoint, returning its content-hash key."""
        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint must be a mapping")
        payload = json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        key = f"{self._namespace}/{digest}.json"
        self._backend.put(key, payload)
        return key

    def load_checkpoint(self, key: str) -> dict[str, Any]:
        """Load a checkpoint and verify that its bytes match the hash encoded in ``key``."""
        expected_digest = _expected_digest_from_key(key)
        payload = self._backend.get(key)
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"checkpoint integrity check failed for {key!r}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        checkpoint = json.loads(payload.decode("utf-8"))
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint stored at {key!r} must decode to a JSON object")
        return checkpoint

    def delete_checkpoint(self, key: str) -> None:
        """Delete a previously stored checkpoint by key."""
        self._backend.delete(key)

    def list_checkpoints(self) -> list[str]:
        """Return all checkpoint keys stored in this checkpoint namespace."""
        return self._backend.list_keys(f"{self._namespace}/")

    def exists(self, key: str) -> bool:
        """Return whether a checkpoint currently exists."""
        return self._backend.exists(key)
