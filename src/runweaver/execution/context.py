"""Stable runtime services exposed to user blocks."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

import numpy as np

from runweaver.contracts import ArtifactStore, CheckpointManager
from runweaver.domain.models import MetricRecord, ResourceRequirements
from runweaver.exceptions import CancellationRequested


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = "cancellation requested"

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "cancellation requested") -> None:
        self._reason = reason
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancellationRequested(self._reason)


class EnvironmentSecrets:
    """Resolve secrets by name without making them serializable run state."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def get(self, name: str) -> str:
        key = f"{self.prefix}{name}"
        try:
            return os.environ[key]
        except KeyError as exc:
            raise KeyError(f"secret {name!r} is not configured") from exc


class NullCheckpointManager:
    def save(self, payload: object) -> object:
        raise RuntimeError("checkpoint persistence is not configured")

    def latest(self, block_run_id: str) -> None:
        return None


@dataclass
class RunContext:
    experiment_id: UUID
    round_id: UUID | None
    trial_id: UUID
    pipeline_run_id: str
    block_run_id: str
    partition_id: str | None
    seed: int
    work_dir: Path
    artifact_store: ArtifactStore
    checkpoint_manager: CheckpointManager = field(default_factory=NullCheckpointManager)
    metric_sink: Callable[[MetricRecord], None] = field(default=lambda metric: None)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("runweaver"))
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    secrets: EnvironmentSecrets = field(default_factory=EnvironmentSecrets)
    clock: Clock = field(default_factory=SystemClock)
    parameters: Mapping[str, object] = field(default_factory=dict)

    def child_rng(self, stream: int | str = 0) -> np.random.Generator:
        """Create a reproducible child RNG without mutable global seed state."""

        stream_id = stream if isinstance(stream, int) else sum(stream.encode("utf-8"))
        return np.random.default_rng(np.random.SeedSequence([self.seed, int(stream_id)]))

    def report_metric(self, metric: MetricRecord) -> None:
        self.metric_sink(metric)

    def for_partition(self, partition_id: str, *, block_run_id: str | None = None) -> RunContext:
        return RunContext(
            experiment_id=self.experiment_id,
            round_id=self.round_id,
            trial_id=self.trial_id,
            pipeline_run_id=self.pipeline_run_id,
            block_run_id=block_run_id or self.block_run_id,
            partition_id=partition_id,
            seed=int(np.random.SeedSequence([self.seed, sum(partition_id.encode())]).generate_state(1)[0]),
            work_dir=self.work_dir / "partitions" / partition_id,
            artifact_store=self.artifact_store,
            checkpoint_manager=self.checkpoint_manager,
            metric_sink=self.metric_sink,
            logger=self.logger,
            cancellation=self.cancellation,
            resources=self.resources,
            secrets=self.secrets,
            clock=self.clock,
            parameters=self.parameters,
        )
