from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

import qes.storage_backend as storage_module
from qes.storage_backend import (
    CheckpointStore,
    LocalFilesystemStorageBackend,
    S3StorageBackend,
)


def test_normalize_storage_path_and_digest_helpers_cover_edge_cases() -> None:
    with pytest.raises(TypeError, match="string"):
        storage_module._normalize_storage_path(1, allow_empty=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        storage_module._normalize_storage_path("", allow_empty=False)
    assert storage_module._normalize_storage_path("", allow_empty=True) == ""
    with pytest.raises(ValueError, match="NUL"):
        storage_module._normalize_storage_path("bad\x00key", allow_empty=False)
    with pytest.raises(ValueError, match="relative"):
        storage_module._normalize_storage_path("C:\\outside.txt", allow_empty=False)
    with pytest.raises(ValueError, match="path separator"):
        storage_module._normalize_storage_path("/outside.txt", allow_empty=False)
    with pytest.raises(ValueError, match="path traversal"):
        storage_module._normalize_storage_path("../escape", allow_empty=False)
    with pytest.raises(ValueError, match="at least one"):
        storage_module._normalize_storage_path("./", allow_empty=False)
    assert storage_module._normalize_storage_path("./", allow_empty=True) == ""
    assert storage_module._normalize_storage_path("folder\\sub\\", allow_empty=True) == "folder/sub/"

    digest = "a" * 64
    assert storage_module._expected_digest_from_key(f"checkpoints/{digest}.json") == digest
    assert storage_module._expected_digest_from_key(f"checkpoints/{digest}") == digest
    with pytest.raises(ValueError, match="64-character SHA-256"):
        storage_module._expected_digest_from_key("checkpoints/not-a-digest.json")


def test_local_filesystem_backend_put_get_exists_and_list_keys(tmp_path: Path) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "artifacts")

    backend.put("runs/alpha/result.bin", b"alpha")
    backend.put("runs/beta/result.bin", b"beta")

    assert backend.root_directory == (tmp_path / "artifacts").resolve()
    assert backend.exists("runs/alpha/result.bin") is True
    assert backend.get("runs/alpha/result.bin") == b"alpha"
    assert backend.list_keys() == ["runs/alpha/result.bin", "runs/beta/result.bin"]
    assert backend.list_keys("runs/alpha/") == ["runs/alpha/result.bin"]


def test_local_filesystem_backend_delete_removes_file_and_empty_directories(tmp_path: Path) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "storage")

    backend.put("nested/run/output.json", b"{}")
    backend.delete("nested/run/output.json")

    assert backend.exists("nested/run/output.json") is False
    assert not (tmp_path / "storage" / "nested").exists()


@pytest.mark.parametrize(
    "bad_key",
    ["../escape.txt", "..\\escape.txt", "/absolute.txt", "\\absolute.txt", "C:\\outside.txt"],
)
def test_local_filesystem_backend_rejects_path_traversal_and_absolute_keys(
    tmp_path: Path,
    bad_key: str,
) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "storage")

    with pytest.raises(ValueError, match="storage key"):
        backend.put(bad_key, b"blocked")


def test_local_filesystem_backend_raises_clear_errors_for_missing_keys(tmp_path: Path) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "storage")

    with pytest.raises(FileNotFoundError, match="missing.bin"):
        backend.get("missing.bin")
    with pytest.raises(FileNotFoundError, match="missing.bin"):
        backend.delete("missing.bin")


class FakeS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.objects: dict[str, bytes] = {}
        self.list_pages: list[dict[str, Any]] = []
        self.get_error: Exception | None = None
        self.head_error: Exception | None = None

    @staticmethod
    def missing_error(code: str = "404") -> RuntimeError:
        error = RuntimeError("missing")
        error.response = {"Error": {"Code": code}}  # type: ignore[attr-defined]
        return error

    @staticmethod
    def weird_error() -> RuntimeError:
        error = RuntimeError("weird")
        error.response = {"Error": {"Code": "AccessDenied"}}  # type: ignore[attr-defined]
        return error

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("put_object", kwargs))
        self.objects[str(kwargs["Key"])] = bytes(kwargs["Body"])
        return {}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_object", kwargs))
        if self.get_error is not None:
            raise self.get_error
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise self.missing_error()
        return {"Body": io.BytesIO(self.objects[key])}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("head_object", kwargs))
        if self.head_error is not None:
            raise self.head_error
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise self.missing_error()
        return {}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("delete_object", kwargs))
        del self.objects[str(kwargs["Key"])]
        return {}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_objects_v2", kwargs))
        if self.list_pages:
            return self.list_pages.pop(0)
        prefix = str(kwargs["Prefix"])
        return {
            "Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(prefix)],
            "IsTruncated": False,
        }


def test_local_filesystem_backend_stops_pruning_when_parent_not_empty(tmp_path: Path) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "storage")
    backend.put("nested/run/output.json", b"{}")
    backend.put("nested/keep.txt", b"keep")

    backend.delete("nested/run/output.json")

    assert (tmp_path / "storage" / "nested").exists()
    assert not (tmp_path / "storage" / "nested" / "run").exists()


def test_local_filesystem_backend_rejects_keys_resolving_outside_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "storage")
    monkeypatch.setattr(storage_module, "_normalize_storage_path", lambda *_args, **_kwargs: "../outside")
    with pytest.raises(ValueError, match="resolves outside"):
        backend._path_for_key("escape.txt")


def test_s3_storage_backend_uses_expected_boto3_calls() -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("demo-bucket", prefix="qes/artifacts", s3_client=client)

    backend.put("runs/demo.json", b'{"ok":true}')
    assert backend.get("runs/demo.json") == b'{"ok":true}'
    assert backend.exists("runs/demo.json") is True
    assert backend.list_keys("runs/") == ["runs/demo.json"]
    backend.delete("runs/demo.json")
    assert backend.exists("runs/demo.json") is False
    assert backend.bucket == "demo-bucket"
    assert backend.prefix == "qes/artifacts"

    assert client.calls == [
        (
            "put_object",
            {"Bucket": "demo-bucket", "Key": "qes/artifacts/runs/demo.json", "Body": b'{"ok":true}'},
        ),
        (
            "get_object",
            {"Bucket": "demo-bucket", "Key": "qes/artifacts/runs/demo.json"},
        ),
        (
            "head_object",
            {"Bucket": "demo-bucket", "Key": "qes/artifacts/runs/demo.json"},
        ),
        (
            "list_objects_v2",
            {"Bucket": "demo-bucket", "Prefix": "qes/artifacts/runs/"},
        ),
        (
            "head_object",
            {"Bucket": "demo-bucket", "Key": "qes/artifacts/runs/demo.json"},
        ),
        (
            "delete_object",
            {"Bucket": "demo-bucket", "Key": "qes/artifacts/runs/demo.json"},
        ),
        (
            "head_object",
            {"Bucket": "demo-bucket", "Key": "qes/artifacts/runs/demo.json"},
        ),
    ]


def test_s3_storage_backend_constructor_and_helpers_cover_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="bucket"):
        S3StorageBackend("", s3_client=FakeS3Client())

    real_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "boto3":
            raise ImportError("no boto3 here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="boto3"):
        S3StorageBackend("demo-bucket")

    fake_calls: list[tuple[str, str | None]] = []

    class FakeBoto3Module:
        @staticmethod
        def client(service: str, region_name: str | None = None) -> FakeS3Client:
            fake_calls.append((service, region_name))
            return FakeS3Client()

    monkeypatch.setattr("builtins.__import__", lambda name, *args, **kwargs: FakeBoto3Module if name == "boto3" else real_import(name, *args, **kwargs))
    imported_backend = S3StorageBackend("bucket-name", region_name="us-east-1")
    assert imported_backend.bucket == "bucket-name"
    assert fake_calls == [("s3", "us-east-1")]

    client = FakeS3Client()
    backend = S3StorageBackend(" demo-bucket ", prefix="", s3_client=client)
    assert backend.bucket == "demo-bucket"
    assert backend.prefix == ""
    assert backend._object_key("alpha/beta") == "alpha/beta"
    assert backend._object_prefix("") == ""
    assert backend._object_prefix("runs") == "runs"
    assert backend._object_prefix("runs/") == "runs/"
    assert backend._logical_key("alpha/beta") == "alpha/beta"
    assert backend._is_missing_error(Exception()) is False

    weird = RuntimeError("bad")
    weird.response = []  # type: ignore[attr-defined]
    assert backend._is_missing_error(weird) is False
    weird2 = RuntimeError("bad")
    weird2.response = {"Error": []}  # type: ignore[attr-defined]
    assert backend._is_missing_error(weird2) is False


def test_s3_storage_backend_missing_and_passthrough_errors() -> None:
    client = FakeS3Client()
    backend = S3StorageBackend("demo-bucket", prefix="root", s3_client=client)

    with pytest.raises(FileNotFoundError, match="missing.json"):
        backend.get("missing.json")
    with pytest.raises(FileNotFoundError, match="missing.json"):
        backend.delete("missing.json")
    assert backend.exists("missing.json") is False

    client.get_error = FakeS3Client.weird_error()
    with pytest.raises(RuntimeError, match="weird"):
        backend.get("runs/demo.json")

    client.head_error = FakeS3Client.weird_error()
    with pytest.raises(RuntimeError, match="weird"):
        backend.exists("runs/demo.json")


def test_s3_storage_backend_lists_paginated_keys_and_filters_prefix_namespace() -> None:
    client = FakeS3Client()
    client.list_pages = [
        {
            "Contents": [
                {"Key": "ns"},
                {"Key": "ns/runs/a.json"},
                {"Key": "other/runs/ignored.json"},
            ],
            "IsTruncated": True,
            "NextContinuationToken": "token-1",
        },
        {
            "Contents": [
                {"Key": "ns/runs/b.json"},
                {"Key": "ns/runs/c.json"},
            ],
            "IsTruncated": True,
        },
    ]
    backend = S3StorageBackend("demo-bucket", prefix="ns", s3_client=client)

    assert backend.list_keys("runs/") == ["runs/a.json", "runs/b.json", "runs/c.json"]
    assert client.calls == [
        ("list_objects_v2", {"Bucket": "demo-bucket", "Prefix": "ns/runs/"}),
        (
            "list_objects_v2",
            {
                "Bucket": "demo-bucket",
                "Prefix": "ns/runs/",
                "ContinuationToken": "token-1",
            },
        ),
    ]


def test_normalize_storage_path_reaches_posix_absolute_check(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWindowsPath:
        drive = ""

        def is_absolute(self) -> bool:
            return False

    class FakePosixPath:
        parts = ("fake",)

        def is_absolute(self) -> bool:
            return True

    monkeypatch.setattr(storage_module, "PureWindowsPath", lambda _value: FakeWindowsPath())
    monkeypatch.setattr(storage_module, "PurePosixPath", lambda _value: FakePosixPath())
    with pytest.raises(ValueError, match="must not be absolute"):
        storage_module._normalize_storage_path("posix-absolute-probe", allow_empty=False)


def test_checkpoint_store_round_trip_and_hash_key_behavior(tmp_path: Path) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "checkpoint-storage")
    store = CheckpointStore(backend)
    checkpoint = {"seed": 7, "scores": [1.25, 2.5], "metadata": {"label": "demo"}}

    key = store.save_checkpoint(checkpoint)

    payload = b'{"metadata":{"label":"demo"},"scores":[1.25,2.5],"seed":7}'
    expected_digest = hashlib.sha256(payload).hexdigest()
    assert key == f"checkpoints/{expected_digest}.json"
    assert store.list_checkpoints() == [key]
    assert store.exists(key) is True
    assert store.load_checkpoint(key) == checkpoint


def test_checkpoint_store_namespace_delete_and_validation_paths(tmp_path: Path) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "checkpoint-storage")
    store = CheckpointStore(backend, namespace="artifacts")
    assert store.namespace == "artifacts"

    with pytest.raises(TypeError, match="mapping"):
        store.save_checkpoint(["not", "a", "mapping"])  # type: ignore[arg-type]

    key = store.save_checkpoint({"value": 1})
    assert store.exists(key) is True
    store.delete_checkpoint(key)
    assert store.exists(key) is False

    payload = b"[1,2,3]"
    bad_key = f"artifacts/{hashlib.sha256(payload).hexdigest()}.json"
    backend.put(bad_key, payload)
    with pytest.raises(ValueError, match="JSON object"):
        store.load_checkpoint(bad_key)


def test_checkpoint_store_detects_integrity_mismatch(tmp_path: Path) -> None:
    backend = LocalFilesystemStorageBackend(tmp_path / "checkpoint-storage")
    store = CheckpointStore(backend)
    key = store.save_checkpoint({"value": 1})

    backend.put(key, b'{"value":2}')

    with pytest.raises(ValueError, match="integrity check failed"):
        store.load_checkpoint(key)
