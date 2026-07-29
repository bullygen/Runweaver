"""Stable canonical fingerprints for plans, cache keys and lineage."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(mode="json"))
    if is_dataclass(value):
        return canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonicalize(item) for item in value), key=repr)
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {
        "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "__repr__": repr(value),
    }


def stable_json(value: Any) -> str:
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def callable_fingerprint(function: object, *, explicit_version: str | None = None) -> str:
    """Hash source when available and retain a stable distribution fallback."""

    if explicit_version:
        return fingerprint({"explicit_version": explicit_version})
    module = getattr(function, "__module__", "")
    qualname = getattr(function, "__qualname__", repr(function))
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = None
    distribution_version = None
    if module:
        root = module.split(".", 1)[0]
        with suppress(importlib.metadata.PackageNotFoundError):
            distribution_version = importlib.metadata.version(root)
    closure = None
    cells = getattr(function, "__closure__", None)
    if cells:
        closure = [canonicalize(cell.cell_contents) for cell in cells]
    return fingerprint({
        "module": module,
        "qualname": qualname,
        "source": source,
        "closure": closure,
        "distribution_version": distribution_version,
    })


def cache_key(
    *,
    semantic_id: str,
    code_fingerprint: str,
    parameters: Mapping[str, object],
    input_hashes: list[str],
    model_hash: str | None,
    serializer_versions: Mapping[str, str],
    environment: Mapping[str, str],
    seed: int,
    external_dependencies: Mapping[str, str],
) -> str:
    return fingerprint({
        "semantic_id": semantic_id,
        "code_fingerprint": code_fingerprint,
        "parameters": parameters,
        "input_hashes": input_hashes,
        "model_hash": model_hash,
        "serializer_versions": serializer_versions,
        "environment": environment,
        "seed": seed,
        "external_dependencies": external_dependencies,
    })
