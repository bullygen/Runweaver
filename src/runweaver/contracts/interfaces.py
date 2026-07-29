"""Backend-independent structural contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from runweaver.domain.models import (
    ArtifactRef,
    CheckpointPayload,
    DecisionRecord,
    ExperimentHistory,
    MetricRecord,
    ParameterSpace,
    TrialPlan,
)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Serializer(Protocol):
    id: str
    version: str
    media_type: str

    def dumps(self, value: object) -> bytes: ...
    def loads(self, payload: bytes) -> object: ...


class ArtifactStore(Protocol):
    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str,
        serializer_id: str,
        metadata: Mapping[str, object] | None = None,
        lineage: Mapping[str, object] | None = None,
    ) -> ArtifactRef: ...

    def read_bytes(self, artifact: ArtifactRef) -> bytes: ...
    def verify(self, artifact: ArtifactRef) -> bool: ...


class StateStore(Protocol):
    def initialize(self) -> None: ...
    def create_run(self, **values: object) -> object: ...
    def transition(self, run_id: str, target: object, **values: object) -> object: ...
    def report_metric(self, run_id: str, metric: MetricRecord) -> None: ...
    def register_artifact(self, run_id: str, artifact: ArtifactRef, cache_key: str | None = None) -> None: ...


class Executor(Protocol):
    def run(self, pipeline: object, initial_input: BaseModel, **kwargs: object) -> object: ...


class Tracker(Protocol):
    def start_trial(self, plan: TrialPlan) -> None: ...
    def log_metric(self, plan: TrialPlan, metric: MetricRecord) -> None: ...
    def log_artifact(self, plan: TrialPlan, artifact: ArtifactRef) -> None: ...
    def end_trial(self, plan: TrialPlan, status: str) -> None: ...


class ExperimentPlanner(Protocol):
    id: str
    version: str

    def propose(
        self,
        space: ParameterSpace,
        history: ExperimentHistory,
        request: object | None = None,
    ) -> list[TrialPlan]: ...


class DecisionPolicy(Protocol):
    id: str
    version: str

    def decide(self, history: ExperimentHistory, context: object | None = None) -> DecisionRecord: ...


class RefinementStrategy(Protocol):
    id: str
    version: str

    def refine(
        self,
        space: ParameterSpace,
        history: ExperimentHistory,
        selection: Iterable[TrialPlan],
        request: object | None = None,
    ) -> ParameterSpace: ...


class CheckpointManager(Protocol):
    def save(self, payload: CheckpointPayload) -> ArtifactRef: ...
    def latest(self, block_run_id: str) -> CheckpointPayload | None: ...


class SecretProvider(Protocol):
    def get(self, name: str) -> str: ...


class WorkDirectoryProvider(Protocol):
    def for_run(self, run_id: str) -> Path: ...
