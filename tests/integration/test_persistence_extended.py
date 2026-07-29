from __future__ import annotations

from pathlib import Path

from runweaver import FsspecArtifactStore, MetricRecord
from runweaver.domain.models import (
    CheckpointPayload,
    RunKind,
)
from runweaver.persistence import DurableCheckpointManager, SqlAlchemyStateStore


def test_metrics_artifacts_events_export_and_checkpoint(tmp_path: Path) -> None:
    state = SqlAlchemyStateStore(f"sqlite:///{tmp_path / 'state.db'}")
    state.initialize()
    run = state.create_run(logical_key="block", kind=RunKind.BLOCK)
    state.add_event(run.id, "custom", {"x": 1})
    state.report_metric(run.id, MetricRecord(name="loss", value=0.5))
    artifacts = FsspecArtifactStore(str(tmp_path / "artifacts"))
    state_artifact = artifacts.put_bytes(
        b"state",
        media_type="application/octet-stream",
        serializer_id="bytes",
    )
    state.register_artifact(run.id, state_artifact, cache_key="cache")
    assert state.find_cache("cache").content_hash == state_artifact.content_hash
    assert state.metrics(run.id)[0].value == 0.5
    assert state.get_artifact(str(state_artifact.id)).uri == state_artifact.uri
    assert state.artifacts_under(run.id)[0].uri == state_artifact.uri
    assert any(event["event_type"] == "custom" for event in state.events(run.id))

    manager = DurableCheckpointManager(state, artifacts, run.id)
    checkpoint = CheckpointPayload(
        block_fingerprint="block",
        parameters_fingerprint="params",
        input_lineage_fingerprint="inputs",
        state_artifact=state_artifact,
        serializer_id="json",
    )
    saved = manager.save(checkpoint)
    assert artifacts.verify(saved)
    assert manager.latest().block_fingerprint == "block"
    assert state.invalidate_cache("cache") == 1
    assert state.find_cache("cache") is None
