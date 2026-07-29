"""Local synchronous/thread/process execution with durable partition resume."""

from __future__ import annotations

import logging
import pickle
import platform
import random
import time
from collections.abc import Mapping
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field

from runweaver.artifacts import FsspecArtifactStore, SerializerRegistry, cache_key, fingerprint
from runweaver.domain.models import (
    CacheMode,
    Experiment,
    MetricRecord,
    RunKind,
    RunState,
    TrialPlan,
)
from runweaver.exceptions import (
    BlockExecutionError,
    CancellationRequested,
    RetryableBlockError,
)
from runweaver.execution.cancellation import SignalController
from runweaver.execution.context import CancellationToken, NullCheckpointManager, RunContext
from runweaver.persistence import DurableCheckpointManager, SqlAlchemyStateStore
from runweaver.pipeline import Pipeline, PipelineNode


class LocalBackend(StrEnum):
    SYNCHRONOUS = "synchronous"
    THREADS = "threads"
    PROCESSES = "processes"


class MaterializationMode(StrEnum):
    EPHEMERAL = "ephemeral"
    DURABLE = "durable"


class LocalExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    backend: LocalBackend = LocalBackend.SYNCHRONOUS
    materialization: MaterializationMode = MaterializationMode.EPHEMERAL
    max_workers: int | None = Field(default=None, ge=1)
    work_dir: Path = Path(".runweaver/work")
    artifact_root: str = ".runweaver/artifacts"
    state_database_url: str = "sqlite:///.runweaver/state.db"
    install_signal_handlers: bool = True
    worker_lease_seconds: float = Field(default=120.0, gt=0)
    retry_jitter_seed: int = 0


@dataclass(frozen=True)
class PipelineResult:
    state: RunState
    outputs: Mapping[str, BaseModel]
    metrics: tuple[MetricRecord, ...]
    pipeline_run_id: str
    block_run_ids: Mapping[str, str]
    cache_hits: tuple[str, ...] = ()
    resumed_partitions: tuple[str, ...] = ()

    @property
    def final_output(self) -> BaseModel:
        return list(self.outputs.values())[-1]


def _environment_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
    }


def _worker_execute(
    block: object,
    payload: Mapping[str, object],
    context_values: Mapping[str, object],
) -> dict[str, object]:
    """Spawn-safe worker entrypoint; never returns backend-specific handles."""


    typed_block = block
    artifact_store = FsspecArtifactStore(str(context_values["artifact_root"]))
    context = RunContext(
        experiment_id=UUID(str(context_values["experiment_id"])),
        round_id=UUID(str(context_values["round_id"])) if context_values.get("round_id") else None,
        trial_id=UUID(str(context_values["trial_id"])),
        pipeline_run_id=str(context_values["pipeline_run_id"]),
        block_run_id=str(context_values["block_run_id"]),
        partition_id=str(context_values["partition_id"]),
        seed=int(context_values["seed"]),
        work_dir=Path(str(context_values["work_dir"])),
        artifact_store=artifact_store,
        resources=context_values["resources"],
        parameters=context_values["parameters"],
    )
    inputs = typed_block.input_type.model_validate(payload)
    output = typed_block.execute(inputs, context)
    return output.model_dump(mode="python")


class LocalExecutor:
    """Execute a typed pipeline without requiring an orchestration server."""

    def __init__(
        self,
        config: LocalExecutionConfig | None = None,
        *,
        artifact_store: FsspecArtifactStore | None = None,
        state_store: SqlAlchemyStateStore | None = None,
    ) -> None:
        self.config = config or LocalExecutionConfig()
        self.artifact_store = artifact_store or FsspecArtifactStore(self.config.artifact_root)
        self.state_store = state_store
        if self.config.materialization == MaterializationMode.DURABLE:
            self.state_store = self.state_store or SqlAlchemyStateStore(self.config.state_database_url)
            self.state_store.initialize()
        self.serializers = SerializerRegistry()
        self.logger = logging.getLogger("runweaver.execution")

    def run(
        self,
        pipeline: Pipeline,
        initial_input: BaseModel,
        *,
        experiment: Experiment | None = None,
        trial_plan: TrialPlan | None = None,
        resume: bool = True,
        cancellation: CancellationToken | None = None,
    ) -> PipelineResult:
        pipeline.validate()
        if experiment is None:
            experiment = Experiment(
                id=uuid5(NAMESPACE_URL, f"runweaver:direct:{pipeline.name}"),
                name=pipeline.name,
            )
        if trial_plan is None:
            direct_fingerprint = fingerprint({
                "pipeline": pipeline.describe(),
                "input": initial_input,
                "seed": experiment.seed,
            })
            trial_plan = TrialPlan(
                id=uuid5(NAMESPACE_URL, f"runweaver:trial:{direct_fingerprint}"),
                parameters={},
                seed=experiment.seed,
                pipeline_version=pipeline.version,
                planner_id="direct",
                planner_version="1",
                fingerprint=direct_fingerprint,
            )
        token = cancellation or CancellationToken()
        metrics: list[MetricRecord] = []
        output_hashes: dict[str, str] = {}
        outputs: dict[str, BaseModel] = {}
        block_run_ids: dict[str, str] = {}
        cache_hits: list[str] = []
        resumed_partitions: list[str] = []

        experiment_run_id, trial_run_id, pipeline_run_id = self._prepare_hierarchy(
            experiment, trial_plan, pipeline
        )
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        controller = (
            SignalController(token, on_first=lambda: self._request_pause(pipeline_run_id))
            if self.config.install_signal_handlers
            else _NullContext()
        )
        try:
            self._ensure_running(experiment_run_id)
            self._ensure_running(trial_run_id)
            self._ensure_running(pipeline_run_id)
            with controller:
                for node in pipeline.topological_order():
                    token.raise_if_cancelled()
                    source_input = self._build_input(node, initial_input, outputs)
                    node_output, run_id, artifact_hash, hit, resumed = self._execute_node(
                        node=node,
                        source_input=source_input,
                        experiment=experiment,
                        trial_plan=trial_plan,
                        pipeline_run_id=pipeline_run_id,
                        upstream_hashes=[
                            output_hashes[dependency] for dependency in node.dependencies
                        ] or [fingerprint(initial_input)],
                        resume=resume,
                        token=token,
                        metric_sink=metrics.append,
                    )
                    outputs[node.id] = node_output
                    block_run_ids[node.id] = run_id
                    output_hashes[node.id] = artifact_hash
                    if hit:
                        cache_hits.append(node.id)
                    resumed_partitions.extend(resumed)
            self._finish_run(pipeline_run_id, RunState.COMPLETED)
            self._finish_run(trial_run_id, RunState.COMPLETED)
            self._finish_run(experiment_run_id, RunState.COMPLETED)
            return PipelineResult(
                state=RunState.COMPLETED,
                outputs=outputs,
                metrics=tuple(metrics),
                pipeline_run_id=pipeline_run_id,
                block_run_ids=block_run_ids,
                cache_hits=tuple(cache_hits),
                resumed_partitions=tuple(resumed_partitions),
            )
        except CancellationRequested:
            self._pause_running(pipeline_run_id)
            return PipelineResult(
                state=RunState.PAUSED,
                outputs=outputs,
                metrics=tuple(metrics),
                pipeline_run_id=pipeline_run_id,
                block_run_ids=block_run_ids,
                cache_hits=tuple(cache_hits),
                resumed_partitions=tuple(resumed_partitions),
            )
        except Exception as exc:
            self._fail_running(pipeline_run_id, exc)
            self._fail_running(trial_run_id, exc)
            self._fail_running(experiment_run_id, exc)
            raise

    def _prepare_hierarchy(
        self,
        experiment: Experiment,
        plan: TrialPlan,
        pipeline: Pipeline,
    ) -> tuple[str, str, str]:
        if not self.state_store:
            return str(experiment.id), str(plan.id), str(uuid4())
        experiment_run = self.state_store.create_run(
            logical_key=f"experiment:{experiment.id}",
            kind=RunKind.EXPERIMENT,
            metadata=experiment.model_dump(mode="json"),
        )
        trial_run = self.state_store.create_run(
            logical_key=f"experiment:{experiment.id}:trial:{plan.fingerprint}",
            kind=RunKind.TRIAL,
            parent_id=experiment_run.id,
            metadata=plan.model_dump(mode="json"),
        )
        pipeline_run = self.state_store.create_run(
            logical_key=(
                f"{trial_run.logical_key}:pipeline:{pipeline.name}:{pipeline.version}:"
                f"{fingerprint(pipeline.describe())}"
            ),
            kind=RunKind.PIPELINE,
            parent_id=trial_run.id,
            metadata=pipeline.describe(),
        )
        return experiment_run.id, trial_run.id, pipeline_run.id

    def _ensure_running(self, run_id: str) -> None:
        if not self.state_store:
            return
        record = self.state_store.get_run(run_id)
        if record is None or record.state in (RunState.COMPLETED, RunState.CACHED):
            return
        if record.state in (RunState.PAUSED, RunState.FAILED, RunState.STALE, RunState.ORPHANED) or record.state == RunState.PLANNED:
            record = self.state_store.transition(run_id, RunState.QUEUED)
        if record.state == RunState.QUEUED:
            self.state_store.transition(run_id, RunState.RUNNING)

    def _build_input(
        self,
        node: PipelineNode,
        initial_input: BaseModel,
        outputs: Mapping[str, BaseModel],
    ) -> BaseModel:
        if not node.dependencies:
            source = initial_input.model_dump(mode="python")
        elif len(node.dependencies) == 1:
            source = outputs[node.dependencies[0]].model_dump(mode="python")
        else:
            source = {}
            for dependency in node.dependencies:
                overlap = source.keys() & outputs[dependency].model_fields.keys()
                if overlap:
                    raise BlockExecutionError(
                        f"fan-in for {node.id} has colliding fields: {sorted(overlap)}"
                    )
                source.update(outputs[dependency].model_dump(mode="python"))
        return node.block.input_type.model_validate(source)

    def _execute_node(
        self,
        *,
        node: PipelineNode,
        source_input: BaseModel,
        experiment: Experiment,
        trial_plan: TrialPlan,
        pipeline_run_id: str,
        upstream_hashes: list[str],
        resume: bool,
        token: CancellationToken,
        metric_sink: Any,
    ) -> tuple[BaseModel, str, str, bool, list[str]]:
        logical_key = f"pipeline:{pipeline_run_id}:block:{node.id}:{node.block.spec.code_fingerprint}"
        if self.state_store:
            block_record = self.state_store.create_run(
                logical_key=logical_key,
                kind=RunKind.BLOCK,
                parent_id=pipeline_run_id,
                metadata={"node": node.id, "spec": node.block.spec.model_dump(mode="json")},
            )
            block_run_id = block_record.id
        else:
            block_record = None
            block_run_id = str(uuid4())
        serializer = self.serializers.get(node.block.spec.serializer_id)
        effective_key = cache_key(
            semantic_id=node.block.spec.semantic_id,
            code_fingerprint=node.block.spec.code_fingerprint,
            parameters=trial_plan.parameters,
            input_hashes=upstream_hashes,
            model_hash=None,
            serializer_versions={serializer.id: serializer.version},
            environment=_environment_fingerprint(),
            seed=trial_plan.seed,
            external_dependencies=node.block.spec.external_dependencies,
        )

        if (
            resume
            and self.state_store
            and block_record
            and block_record.state in (RunState.COMPLETED, RunState.CACHED)
        ):
            artifacts = self.state_store.artifacts_for_run(block_run_id)
            if artifacts:
                artifact = artifacts[-1]
                if self.artifact_store.verify(artifact):
                    value = serializer.loads(self.artifact_store.read_bytes(artifact))
                    return (
                        node.block.output_type.model_validate(value),
                        block_run_id,
                        artifact.content_hash,
                        True,
                        [],
                    )

        if (
            self.state_store
            and node.block.spec.cache_policy in (CacheMode.READ_WRITE, CacheMode.READ_ONLY)
        ):
            cached = self.state_store.find_cache(effective_key)
            if cached and self.artifact_store.verify(cached):
                self._ensure_running(block_run_id)
                value = node.block.output_type.model_validate(
                    serializer.loads(self.artifact_store.read_bytes(cached))
                )
                self.state_store.register_artifact(block_run_id, cached, effective_key)
                self._finish_run(block_run_id, RunState.CACHED)
                return value, block_run_id, cached.content_hash, True, []

        self._ensure_running(block_run_id)
        context = RunContext(
            experiment_id=experiment.id,
            round_id=trial_plan.round_id,
            trial_id=trial_plan.id,
            pipeline_run_id=pipeline_run_id,
            block_run_id=block_run_id,
            partition_id=None,
            seed=trial_plan.seed,
            work_dir=self.config.work_dir / pipeline_run_id / node.id,
            artifact_store=self.artifact_store,
            checkpoint_manager=(
                DurableCheckpointManager(self.state_store, self.artifact_store, block_run_id)
                if self.state_store
                else NullCheckpointManager()
            ),
            metric_sink=lambda metric: self._record_metric(block_run_id, metric, metric_sink),
            logger=self.logger,
            cancellation=token,
            resources=node.resources or node.block.spec.resources,
            parameters=trial_plan.parameters,
        )
        context.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            if node.map_over:
                output, resumed = self._execute_map(
                    node, source_input, context, logical_key, resume
                )
            else:
                output = self._execute_with_retry(node, source_input, context)
                resumed = []
            payload = serializer.dumps(output)
            if self.config.materialization == MaterializationMode.EPHEMERAL:
                return output, block_run_id, fingerprint(output), False, resumed
            artifact = self.artifact_store.put_bytes(
                payload,
                media_type=serializer.media_type,
                serializer_id=serializer.id,
                metadata={
                    "pipeline_run_id": pipeline_run_id,
                    "block_run_id": block_run_id,
                    "node_id": node.id,
                    "cache_key": effective_key,
                },
                lineage={
                    "producing_block": node.block.spec.semantic_id,
                    "block_fingerprint": node.block.spec.code_fingerprint,
                    "parameters": dict(trial_plan.parameters),
                    "upstream_hashes": upstream_hashes,
                    "trial_id": str(trial_plan.id),
                    "environment": _environment_fingerprint(),
                    "serializer": f"{serializer.id}:{serializer.version}",
                },
            )
            if self.state_store:
                self.state_store.register_artifact(block_run_id, artifact, effective_key)
                self._finish_run(block_run_id, RunState.COMPLETED)
            return output, block_run_id, artifact.content_hash, False, resumed
        except CancellationRequested:
            self._pause_running(block_run_id)
            raise
        except Exception as exc:
            self._fail_running(block_run_id, exc)
            if isinstance(exc, BlockExecutionError):
                raise
            raise BlockExecutionError(f"block {node.id!r} failed: {exc}") from exc

    def _execute_with_retry(
        self,
        node: PipelineNode,
        inputs: BaseModel,
        context: RunContext,
    ) -> BaseModel:
        policy = node.block.spec.retry_policy
        rng = random.Random(self.config.retry_jitter_seed + context.seed)
        for attempt in range(1, policy.max_attempts + 1):
            context.cancellation.raise_if_cancelled()
            try:
                return node.block.output_type.model_validate(node.block.execute(inputs, context))
            except CancellationRequested:
                raise
            except Exception as exc:
                retryable = isinstance(exc, RetryableBlockError) or exc.__class__.__name__ in policy.retryable_exceptions
                if not retryable or attempt >= policy.max_attempts:
                    raise
                delay = min(
                    policy.maximum_delay_s,
                    policy.initial_delay_s * policy.exponential_base ** (attempt - 1),
                ) + rng.uniform(0, policy.jitter_s)
                if self.state_store:
                    self.state_store.add_event(
                        context.block_run_id,
                        "retry_scheduled",
                        {"attempt": attempt, "delay_s": delay, "error": repr(exc)},
                    )
                context.cancellation.raise_if_cancelled()
                if delay:
                    time.sleep(delay)
        raise AssertionError("retry loop exhausted unexpectedly")

    def _execute_map(
        self,
        node: PipelineNode,
        source_input: BaseModel,
        context: RunContext,
        block_logical_key: str,
        resume: bool,
    ) -> tuple[BaseModel, list[str]]:
        items = list(getattr(source_input, node.map_over))
        results: list[BaseModel | None] = [None] * len(items)
        pending: list[tuple[int, BaseModel, RunContext, str]] = []
        resumed: list[str] = []
        serializer = self.serializers.get(node.block.spec.serializer_id)
        for index, item in enumerate(items):
            partition_id = f"{index:08d}"
            child_context = context.for_partition(partition_id)
            payload = source_input.model_dump(mode="python")
            payload[node.map_over] = [item]
            partition_input = node.block.input_type.model_validate(payload)
            partition_run_id = str(uuid4())
            if self.state_store:
                record = self.state_store.create_run(
                    logical_key=f"{block_logical_key}:partition:{partition_id}",
                    kind=RunKind.PARTITION,
                    parent_id=context.block_run_id,
                    metadata={"partition_id": partition_id, "index": index},
                )
                partition_run_id = record.id
                child_context.block_run_id = partition_run_id
                if resume and record.state in (RunState.COMPLETED, RunState.CACHED):
                    artifacts = self.state_store.artifacts_for_run(record.id)
                    if artifacts and self.artifact_store.verify(artifacts[-1]):
                        value = serializer.loads(self.artifact_store.read_bytes(artifacts[-1]))
                        results[index] = node.block.output_type.model_validate(value)
                        resumed.append(partition_id)
                        continue
                self._ensure_running(record.id)
            pending.append((index, partition_input, child_context, partition_run_id))

        if pending:
            if self.config.backend == LocalBackend.SYNCHRONOUS or node.parallelism == 1:
                for index, partition_input, child_context, run_id in pending:
                    results[index] = self._run_partition(
                        node, partition_input, child_context, run_id, serializer
                    )
            else:
                self._run_parallel_partitions(node, pending, results, serializer)
        return self._collect_outputs(node, [result for result in results if result is not None]), resumed

    def _run_partition(
        self,
        node: PipelineNode,
        inputs: BaseModel,
        context: RunContext,
        run_id: str,
        serializer: object,
    ) -> BaseModel:
        try:
            output = self._execute_with_retry(node, inputs, context)
            if self.state_store:
                payload = serializer.dumps(output)
                artifact = self.artifact_store.put_bytes(
                    payload,
                    media_type=serializer.media_type,
                    serializer_id=serializer.id,
                    metadata={"kind": "partition", "partition_id": context.partition_id},
                    lineage={
                        "producing_block": node.block.spec.semantic_id,
                        "block_fingerprint": node.block.spec.code_fingerprint,
                        "partition_id": context.partition_id,
                    },
                )
                self.state_store.register_artifact(run_id, artifact)
                self._finish_run(run_id, RunState.COMPLETED)
            return output
        except Exception as exc:
            self._fail_running(run_id, exc)
            raise

    def _run_parallel_partitions(
        self,
        node: PipelineNode,
        pending: list[tuple[int, BaseModel, RunContext, str]],
        results: list[BaseModel | None],
        serializer: object,
    ) -> None:
        workers = min(node.parallelism, self.config.max_workers or node.parallelism)
        executor_type: type[ThreadPoolExecutor | ProcessPoolExecutor]
        executor_type = (
            ProcessPoolExecutor if self.config.backend == LocalBackend.PROCESSES else ThreadPoolExecutor
        )
        if self.config.backend == LocalBackend.PROCESSES:
            try:
                pickle.dumps(node.block)
            except Exception as exc:
                raise BlockExecutionError(
                    f"block {node.id!r} is not process-serializable; use a top-level "
                    "function/class or the threads backend"
                ) from exc
        with executor_type(max_workers=workers) as executor:
            futures: dict[Future[object], tuple[int, RunContext, str]] = {}
            for index, partition_input, context, run_id in pending:
                if self.config.backend == LocalBackend.PROCESSES:
                    context_values = {
                        "artifact_root": self.config.artifact_root,
                        "experiment_id": str(context.experiment_id),
                        "round_id": str(context.round_id) if context.round_id else None,
                        "trial_id": str(context.trial_id),
                        "pipeline_run_id": context.pipeline_run_id,
                        "block_run_id": run_id,
                        "partition_id": context.partition_id,
                        "seed": context.seed,
                        "work_dir": str(context.work_dir),
                        "resources": context.resources,
                        "parameters": dict(context.parameters),
                    }
                    future = executor.submit(
                        _worker_execute,
                        node.block,
                        partition_input.model_dump(mode="python"),
                        context_values,
                    )
                else:
                    future = executor.submit(
                        self._execute_with_retry, node, partition_input, context
                    )
                futures[future] = (index, context, run_id)
            for future in as_completed(futures):
                index, context, run_id = futures[future]
                try:
                    raw = future.result()
                    output = node.block.output_type.model_validate(raw)
                    results[index] = output
                    if self.state_store:
                        payload = serializer.dumps(output)
                        artifact = self.artifact_store.put_bytes(
                            payload,
                            media_type=serializer.media_type,
                            serializer_id=serializer.id,
                            metadata={"kind": "partition", "partition_id": context.partition_id},
                            lineage={
                                "producing_block": node.block.spec.semantic_id,
                                "block_fingerprint": node.block.spec.code_fingerprint,
                                "partition_id": context.partition_id,
                            },
                        )
                        self.state_store.register_artifact(run_id, artifact)
                        self._finish_run(run_id, RunState.COMPLETED)
                except Exception as exc:
                    self._fail_running(run_id, exc)
                    for other in futures:
                        other.cancel()
                    raise

    def _collect_outputs(self, node: PipelineNode, outputs: list[BaseModel]) -> BaseModel:
        if not outputs:
            return node.block.output_type.model_validate({})
        collected: dict[str, object] = {}
        for name in node.block.output_type.model_fields:
            values = [getattr(output, name) for output in outputs]
            if all(isinstance(value, (list, tuple)) for value in values):
                collected[name] = [item for value in values for item in value]
            else:
                collected[name] = values
        return node.block.output_type.model_validate(collected)

    def _record_metric(self, run_id: str, metric: MetricRecord, sink: Any) -> None:
        sink(metric)
        if self.state_store:
            self.state_store.report_metric(run_id, metric)

    def _finish_run(self, run_id: str, target: RunState) -> None:
        if not self.state_store:
            return
        record = self.state_store.get_run(run_id)
        if record and record.state not in (target, RunState.COMPLETED, RunState.CACHED):
            self.state_store.transition(run_id, target)

    def _fail_running(self, run_id: str, exc: Exception) -> None:
        if not self.state_store:
            return
        record = self.state_store.get_run(run_id)
        if record and record.state in (RunState.RUNNING, RunState.PAUSING):
            self.state_store.transition(run_id, RunState.FAILED, error=repr(exc))

    def _request_pause(self, run_id: str) -> None:
        if not self.state_store:
            return
        record = self.state_store.get_run(run_id)
        if record and record.state == RunState.RUNNING:
            self.state_store.transition(run_id, RunState.PAUSING)

    def _pause_running(self, run_id: str) -> None:
        if not self.state_store:
            return
        record = self.state_store.get_run(run_id)
        if record and record.state == RunState.RUNNING:
            record = self.state_store.transition(run_id, RunState.PAUSING)
        if record and record.state == RunState.PAUSING:
            self.state_store.transition(run_id, RunState.PAUSED)


class _NullContext:
    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None
