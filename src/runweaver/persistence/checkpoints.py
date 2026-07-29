"""Checkpoint manager backed by the artifact and domain stores."""

from __future__ import annotations

from runweaver.artifacts.serializers import JsonSerializer
from runweaver.contracts import ArtifactStore
from runweaver.domain.models import ArtifactRef, CheckpointPayload
from runweaver.persistence.store import SqlAlchemyStateStore


class DurableCheckpointManager:
    def __init__(
        self,
        state_store: SqlAlchemyStateStore,
        artifact_store: ArtifactStore,
        block_run_id: str,
    ) -> None:
        self.state_store = state_store
        self.artifact_store = artifact_store
        self.block_run_id = block_run_id
        self.serializer = JsonSerializer()

    def save(self, payload: CheckpointPayload) -> ArtifactRef:
        encoded = self.serializer.dumps(payload)
        artifact = self.artifact_store.put_bytes(
            encoded,
            media_type=self.serializer.media_type,
            serializer_id=self.serializer.id,
            metadata={"kind": "checkpoint", "block_run_id": self.block_run_id},
            lineage={
                "block_fingerprint": payload.block_fingerprint,
                "parameters_fingerprint": payload.parameters_fingerprint,
                "input_lineage_fingerprint": payload.input_lineage_fingerprint,
            },
        )
        self.state_store.register_artifact(self.block_run_id, artifact)
        self.state_store.save_checkpoint(self.block_run_id, payload.model_dump_json())
        return artifact

    def latest(self, block_run_id: str | None = None) -> CheckpointPayload | None:
        encoded = self.state_store.latest_checkpoint(block_run_id or self.block_run_id)
        return CheckpointPayload.model_validate_json(encoded) if encoded else None
