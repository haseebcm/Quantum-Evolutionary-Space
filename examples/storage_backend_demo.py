"""Demonstrate real local object storage and honest optional S3 construction.

The local-filesystem portion of this example is executed for real on the current
machine. The S3 portion is only demonstrated as a construction example and does
not attempt any network access or bucket operations. In this session, boto3 was
not available, so the S3 section reports that it is skipped.
"""
from __future__ import annotations

import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qes.storage_backend import (  # noqa: E402
    CheckpointStore,
    LocalFilesystemStorageBackend,
    S3StorageBackend,
)


def _timed_call(label: str, action: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = action()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.6f}s")
    return result, elapsed


def main() -> None:
    """Run the storage backend demo with honest output about what is and is not exercised."""
    demo_root = PROJECT_ROOT / "examples" / "_storage_backend_demo_data"
    if demo_root.exists():
        shutil.rmtree(demo_root)

    print("QES storage backend demo")
    print("Local backend: real filesystem operations under examples\\_storage_backend_demo_data")

    backend = LocalFilesystemStorageBackend(demo_root)
    store = CheckpointStore(backend, namespace="demo-checkpoints")

    _, _ = _timed_call("put sample-1", lambda: backend.put("objects/sample-1.bin", b"alpha-bytes"))
    _, _ = _timed_call("put sample-2", lambda: backend.put("objects/sample-2.bin", b"beta-bytes"))
    listed, _ = _timed_call("list objects", lambda: backend.list_keys("objects/"))
    payload, _ = _timed_call("get sample-1", lambda: backend.get("objects/sample-1.bin"))
    checkpoint_key, _ = _timed_call(
        "save checkpoint",
        lambda: store.save_checkpoint({"seed": 7, "objective": 1.25, "status": "ok"}),
    )
    checkpoint, _ = _timed_call("load checkpoint", lambda: store.load_checkpoint(checkpoint_key))
    _, _ = _timed_call("delete sample-2", lambda: backend.delete("objects/sample-2.bin"))

    print(f"listed_keys={listed}")
    print(f"sample-1-bytes={payload!r}")
    print(f"checkpoint_key={checkpoint_key}")
    print(f"checkpoint_payload={checkpoint}")

    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError:
        print("S3 backend demo skipped honestly: boto3 is not installed in this environment.")
    else:
        print("S3 backend construction demo only: no network calls will be made.")
        s3_backend = S3StorageBackend("example-bucket", prefix="qes/demo", s3_client=boto3.client("s3"))
        print(
            "Constructed S3StorageBackend for bucket="
            f"{s3_backend.bucket!r} prefix={s3_backend.prefix!r}; no object operations executed."
        )

    shutil.rmtree(demo_root)
    print("Cleaned up demo data directory.")


if __name__ == "__main__":
    main()
