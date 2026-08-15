"""Lightweight serialization for QSEE-11L and related objects.

Provides JSON-friendly conversion helpers. These are intentionally simple and
should be replaced by a robust serialization layer for production use.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import zlib
from collections.abc import Iterable
from typing import Any

import numpy as np

SERIALIZE_VERSION = 1


def _as_serializable(obj: Any) -> Any:
    if isinstance(obj, complex) or (isinstance(obj, np.number) and np.iscomplexobj(obj)):
        return {"re": obj.real, "im": obj.imag}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "snapshot"):
        return _as_serializable(obj.snapshot())
    if isinstance(obj, dict):
        return {k: _as_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_serializable(v) for v in obj]
    return str(obj)


def to_json(obj: Any) -> str:
    data = _as_serializable(obj)
    if isinstance(data, dict):
        data_copy = dict(data)
        data_copy["_version"] = SERIALIZE_VERSION
        return json.dumps(data_copy, indent=2)
    
    wrapper = {"_version": SERIALIZE_VERSION, "data": data}
    return json.dumps(wrapper, indent=2)


def to_file(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf8") as fh:
        fh.write(to_json(obj))


def from_json(json_string: str) -> Any:
    parsed = json.loads(json_string)
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
    wrapper = {"_version": SERIALIZE_VERSION, "data": obj}
    return zlib.compress(pickle.dumps(wrapper))


def from_bytes(data: bytes) -> Any:
    parsed = pickle.loads(zlib.decompress(data))
    if isinstance(parsed, dict) and "_version" in parsed and "data" in parsed:
        return parsed["data"]
    return parsed


def to_compressed(obj: Any) -> bytes:
    return gzip.compress(to_json(obj).encode("utf8"))


def from_compressed(data: bytes) -> Any:
    json_string = gzip.decompress(data).decode("utf8")
    return from_json(json_string)


def to_jsonl(objects: Iterable[Any], path: str) -> None:
    with open(path, "w", encoding="utf8") as fh:
        for obj in objects:
            data = _as_serializable(obj)
            if isinstance(data, dict):
                data_copy = dict(data)
                data_copy["_version"] = SERIALIZE_VERSION
                fh.write(json.dumps(data_copy) + "\n")
            else:
                wrapper = {"_version": SERIALIZE_VERSION, "data": data}
                fh.write(json.dumps(wrapper) + "\n")


def from_jsonl(path: str) -> list[Any]:
    results = []
    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict) and "_version" in parsed:
                if "data" in parsed and len(parsed) == 2:
                    results.append(parsed["data"])
                else:
                    parsed.pop("_version")
                    results.append(parsed)
            else:
                results.append(parsed)
    return results


def compute_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, expected: str) -> bool:
    return compute_checksum(data) == expected


def pretty(obj: Any) -> str:
    return json.dumps(_as_serializable(obj), indent=4, sort_keys=True)


def compute_diff(old: dict, new: dict) -> dict:
    diff = {}
    for k, v in new.items():
        if k not in old or old[k] != v:
            diff[k] = v
    for k in old:
        if k not in new:
            diff[k] = None
    return diff


def apply_diff(base: dict, diff: dict) -> dict:
    result = dict(base)
    for k, v in diff.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v
    return result
