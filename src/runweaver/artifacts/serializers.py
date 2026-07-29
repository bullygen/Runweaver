"""Safe serializers used by durable pipeline mode."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from pydantic import BaseModel

from runweaver.artifacts.hashing import canonicalize
from runweaver.exceptions import ArtifactError


class JsonSerializer:
    id = "json"
    version = "1"
    media_type = "application/json"

    def dumps(self, value: object) -> bytes:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return json.dumps(
            canonicalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def loads(self, payload: bytes) -> object:
        return json.loads(payload.decode("utf-8"))


class BytesSerializer:
    id = "bytes"
    version = "1"
    media_type = "application/octet-stream"

    def dumps(self, value: object) -> bytes:
        if not isinstance(value, bytes):
            raise ArtifactError("bytes serializer accepts bytes only")
        return value

    def loads(self, payload: bytes) -> object:
        return payload


class TextSerializer:
    id = "text"
    version = "1"
    media_type = "text/plain; charset=utf-8"

    def dumps(self, value: object) -> bytes:
        if not isinstance(value, str):
            raise ArtifactError("text serializer accepts str only")
        return value.encode("utf-8")

    def loads(self, payload: bytes) -> object:
        return payload.decode("utf-8")


class NumpySerializer:
    """Non-pickle NumPy ``.npy`` serializer."""

    id = "numpy-npy"
    version = "1"
    media_type = "application/x-npy"

    def dumps(self, value: object) -> bytes:
        if not isinstance(value, np.ndarray):
            raise ArtifactError("numpy serializer accepts ndarray only")
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return buffer.getvalue()

    def loads(self, payload: bytes) -> object:
        return np.load(io.BytesIO(payload), allow_pickle=False)


class SerializerRegistry:
    def __init__(self) -> None:
        self._items = {
            serializer.id: serializer
            for serializer in (JsonSerializer(), BytesSerializer(), TextSerializer(), NumpySerializer())
        }

    def register(self, serializer: object) -> None:
        serializer_id = getattr(serializer, "id", None)
        if not serializer_id:
            raise ArtifactError("serializer must declare a non-empty id")
        self._items[str(serializer_id)] = serializer

    def get(self, serializer_id: str) -> Any:
        try:
            return self._items[serializer_id]
        except KeyError as exc:
            raise ArtifactError(f"serializer {serializer_id!r} is not registered") from exc
