"""Content-addressed fsspec artifact store with a two-phase commit marker."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import suppress
from pathlib import PurePosixPath
from uuid import uuid4

import fsspec

from runweaver.domain.models import ArtifactRef
from runweaver.exceptions import ArtifactCorruptionError, ArtifactError


class FsspecArtifactStore:
    """Store immutable artifacts on any fsspec filesystem.

    Local files use atomic rename. Remote files become visible only after their
    manifest has been committed, which is the durable completion marker.
    """

    def __init__(self, root_uri: str) -> None:
        self.root_uri = root_uri.rstrip("/")
        self.fs, self.root_path = fsspec.core.url_to_fs(self.root_uri)
        self.fs.makedirs(self._join("objects"), exist_ok=True)
        self.fs.makedirs(self._join(".tmp"), exist_ok=True)

    def _join(self, *parts: str) -> str:
        base = PurePosixPath(self.root_path)
        path = base.joinpath(*(part.strip("/") for part in parts))
        return str(path)

    def _uri(self, path: str) -> str:
        protocol = self.fs.protocol
        if isinstance(protocol, (tuple, list)):
            protocol = protocol[0]
        return path if protocol in ("file", "local") else f"{protocol}://{path}"

    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str,
        serializer_id: str,
        metadata: Mapping[str, object] | None = None,
        lineage: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        digest = hashlib.sha256(payload).hexdigest()
        final_path = self._join("objects", "sha256", digest[:2], digest)
        manifest_path = f"{final_path}.manifest.json"
        temporary = self._join(".tmp", str(uuid4()))
        self.fs.makedirs(str(PurePosixPath(final_path).parent), exist_ok=True)
        try:
            with self.fs.open(temporary, "wb") as handle:
                handle.write(payload)
                if hasattr(handle, "flush"):
                    handle.flush()
                with suppress(AttributeError, OSError):
                    os.fsync(handle.fileno())
            if not self.fs.exists(final_path):
                self.fs.mv(temporary, final_path)
            else:
                self.fs.rm(temporary)
            manifest = {
                "format": "runweaver-artifact-manifest-v1",
                "content_hash": digest,
                "size_bytes": len(payload),
                "media_type": media_type,
                "serializer_id": serializer_id,
                "serializer_version": "1",
                "metadata": dict(metadata or {}),
                "lineage": dict(lineage or {}),
            }
            manifest_tmp = self._join(".tmp", f"{uuid4()}.manifest.json")
            with self.fs.open(manifest_tmp, "wb") as handle:
                handle.write(json.dumps(manifest, sort_keys=True).encode("utf-8"))
            if self.fs.exists(manifest_path):
                self.fs.rm(manifest_tmp)
            else:
                self.fs.mv(manifest_tmp, manifest_path)
        except Exception as exc:
            if self.fs.exists(temporary):
                self.fs.rm(temporary)
            raise ArtifactError(f"failed to commit artifact {digest}: {exc}") from exc
        return ArtifactRef(
            uri=self._uri(final_path),
            content_hash=digest,
            size_bytes=len(payload),
            media_type=media_type,
            serializer_id=serializer_id,
            manifest_uri=self._uri(manifest_path),
            metadata=dict(metadata or {}),
            lineage=dict(lineage or {}),
        )

    def read_bytes(self, artifact: ArtifactRef) -> bytes:
        fs, path = fsspec.core.url_to_fs(artifact.uri)
        if not fs.exists(path):
            raise ArtifactCorruptionError(f"artifact is missing: {artifact.uri}")
        with fs.open(path, "rb") as handle:
            payload = handle.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != artifact.content_hash:
            raise ArtifactCorruptionError(
                f"artifact hash mismatch for {artifact.uri}: expected "
                f"{artifact.content_hash}, got {actual}"
            )
        return payload

    def verify(self, artifact: ArtifactRef) -> bool:
        self.read_bytes(artifact)
        if artifact.manifest_uri:
            manifest_fs, manifest_path = fsspec.core.url_to_fs(artifact.manifest_uri)
            if not manifest_fs.exists(manifest_path):
                raise ArtifactCorruptionError(f"artifact manifest is missing: {artifact.manifest_uri}")
            with manifest_fs.open(manifest_path, "rb") as handle:
                manifest = json.loads(handle.read().decode("utf-8"))
            if manifest.get("content_hash") != artifact.content_hash:
                raise ArtifactCorruptionError(f"manifest hash mismatch: {artifact.manifest_uri}")
        return True

    def lineage(self, artifact: ArtifactRef) -> Mapping[str, object]:
        self.verify(artifact)
        return artifact.lineage
