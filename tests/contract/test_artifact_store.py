from __future__ import annotations

from pathlib import Path

import pytest
from runweaver import FsspecArtifactStore
from runweaver.exceptions import ArtifactCorruptionError


@pytest.mark.parametrize("root", ["memory://runweaver-contract", None])
def test_artifact_store_contract(root: str | None, tmp_path: Path) -> None:
    store = FsspecArtifactStore(root or str(tmp_path / "artifacts"))
    artifact = store.put_bytes(
        b"contract",
        media_type="application/octet-stream",
        serializer_id="bytes",
        metadata={"case": "contract"},
        lineage={"upstream": ["abc"]},
    )
    assert store.read_bytes(artifact) == b"contract"
    assert store.verify(artifact)
    assert artifact.content_hash in artifact.uri
    assert store.lineage(artifact)["upstream"] == ["abc"]


def test_hash_mismatch_is_detected(tmp_path: Path) -> None:
    store = FsspecArtifactStore(str(tmp_path / "artifacts"))
    artifact = store.put_bytes(
        b"good",
        media_type="application/octet-stream",
        serializer_id="bytes",
    )
    Path(artifact.uri).write_bytes(b"bad")
    with pytest.raises(ArtifactCorruptionError, match="hash mismatch"):
        store.verify(artifact)
