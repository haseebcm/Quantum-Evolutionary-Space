"""Lightweight serialization for QSEE-11L and related objects.

Provides JSON-friendly conversion helpers. These are intentionally simple and
should be replaced by a robust serialization layer for production use.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from collections.abc import Iterable
from typing import Any

import numpy as np

SERIALIZE_VERSION = 2
MAX_DECOMPRESSED_BYTES = 16 * 1024 * 1024


def _as_serializable(obj: Any) -> Any:
    if isinstance(obj, complex) or (isinstance(obj, np.number) and np.iscomplexobj(obj)):
        return {"__qes_complex__": [float(obj.real), float(obj.imag)]}
    if isinstance(obj, np.ndarray):
        return _as_serializable(obj.tolist())
    if isinstance(obj, np.generic):
        return _as_serializable(obj.item())
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "snapshot"):
        return _as_serializable(obj.snapshot())
    if isinstance(obj, dict):
        if not all(isinstance(key, str) for key in obj):
            raise TypeError("JSON mapping keys must be strings")
        data = {key: _as_serializable(value) for key, value in obj.items()}
        if set(data) in ({"__qes_complex__"}, {"__qes_mapping__"}):
            return {"__qes_mapping__": list(data.items())}
        return data
    if isinstance(obj, (list, tuple)):
        return [_as_serializable(v) for v in obj]
    raise TypeError(f"unsupported serialized type: {type(obj).__name__}")


def to_json(obj: Any) -> str:
    wrapper = {"_version": SERIALIZE_VERSION, "data": _as_serializable(obj)}
    return json.dumps(wrapper, indent=2, allow_nan=False)


def to_file(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf8") as fh:
        fh.write(to_json(obj))


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def from_json(json_string: str) -> Any:
    def decode(value: dict) -> Any:
        if set(value) == {"__qes_complex__"}:
            real, imag = value["__qes_complex__"]
            return complex(real, imag)
        if set(value) == {"__qes_mapping__"}:
            return dict(value["__qes_mapping__"])
        return value

    if len(json_string.encode("utf8")) > MAX_DECOMPRESSED_BYTES:
        raise ValueError("JSON payload exceeds size limit")
    parsed = json.loads(json_string, object_hook=decode, parse_constant=_reject_constant)
    if isinstance(parsed, dict) and "_version" in parsed:
        if parsed["_version"] not in (1, SERIALIZE_VERSION):
            raise ValueError("unsupported serialization version")
    if isinstance(parsed, dict) and "_version" in parsed:
        if "data" in parsed and len(parsed) == 2:
            return parsed["data"]
        else:
            parsed.pop("_version")
            return parsed
    return parsed


def from_file(path: str) -> Any:
    with open(path, encoding="utf8") as fh:
        return from_json(fh.read())


def to_bytes(obj: Any) -> bytes:
    """Encode data-only compressed JSON; legacy pickle payloads are rejected."""
    return zlib.compress(to_json(obj).encode("utf8"))


def _decompress(data: bytes, *, gzip_format: bool = False) -> bytes:
    decoder = zlib.decompressobj(31 if gzip_format else zlib.MAX_WBITS)
    decoded = decoder.decompress(data, MAX_DECOMPRESSED_BYTES + 1)
    if len(decoded) > MAX_DECOMPRESSED_BYTES or decoder.unconsumed_tail:
        raise ValueError("decompressed payload exceeds size limit")
    if not decoder.eof or decoder.unused_data:
        raise ValueError("invalid or trailing compressed data")
    return decoded


def from_bytes(data: bytes) -> Any:
    return from_json(_decompress(data).decode("utf8"))


def to_compressed(obj: Any) -> bytes:
    return gzip.compress(to_json(obj).encode("utf8"))


def from_compressed(data: bytes) -> Any:
    return from_json(_decompress(data, gzip_format=True).decode("utf8"))


def to_jsonl(objects: Iterable[Any], path: str) -> None:
    with open(path, "w", encoding="utf8") as fh:
        for obj in objects:
            fh.write(json.dumps(json.loads(to_json(obj)), allow_nan=False) + "\n")


def from_jsonl(path: str) -> list[Any]:
    with open(path, encoding="utf8") as fh:
        return [from_json(line) for line in fh if line.strip()]


def compute_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, expected: str) -> bool:
    return compute_checksum(data) == expected


def pretty(obj: Any) -> str:
    return json.dumps(_as_serializable(obj), indent=4, sort_keys=True)


def compute_diff(old: dict, new: dict) -> dict:
    """Return an operation envelope distinguishing deletion from JSON null."""
    return {
        "_patch_version": 2,
        "set": {key: value for key, value in new.items() if key not in old or old[key] != value},
        "delete": [key for key in old if key not in new],
    }


def apply_diff(base: dict, diff: dict) -> dict:
    if diff.get("_patch_version") != 2 or not isinstance(diff.get("set"), dict):
        raise ValueError("expected a version 2 patch envelope")
    if not isinstance(diff.get("delete"), list):
        raise ValueError("patch delete must be a list")
    result = dict(base)
    for key in diff["delete"]:
        result.pop(key, None)
    result.update(diff["set"])
    return result
