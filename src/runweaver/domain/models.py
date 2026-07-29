"""Immutable domain schemas shared by all execution backends."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(UTC)


class FrozenModel(BaseModel):
    """Base model for versioned immutable domain values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DataSpec(FrozenModel):
    type_name: str
    schema_version: str
    semantic_role: str
    media_type: str | None = None


class ModelSpec(FrozenModel):
    type_name: str
    schema_version: str
    framework: str | None = None
    semantic_role: str = "processing_model"


class DataRef(FrozenModel):
    uri: str
    content_hash: str
    spec: DataSpec
    size_bytes: int | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)


class ModelRef(FrozenModel):
    uri: str
    content_hash: str
    spec: ModelSpec
    metadata: Mapping[str, object] = Field(default_factory=dict)


class ArtifactRef(FrozenModel):
    id: UUID = Field(default_factory=uuid4)
    uri: str
    content_hash: str
    size_bytes: int
    media_type: str
    serializer_id: str
    serializer_version: str = "1"
    manifest_uri: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Mapping[str, object] = Field(default_factory=dict)
    lineage: Mapping[str, object] = Field(default_factory=dict)


class Cardinality(StrEnum):
    ONE = "one"
    OPTIONAL = "optional"
    MANY = "many"
    STREAM = "stream"


class VersionCompatibility(StrEnum):
    EXACT = "exact"
    SAME_MAJOR = "same_major"
    BACKWARD = "backward"


class PortSpec(FrozenModel):
    name: str
    schema_ref: str
    schema_version: str = "1"
    required: bool = True
    cardinality: Cardinality = Cardinality.ONE
    semantic_role: str = "data"
    media_types: tuple[str, ...] = ()
    version_compatibility: VersionCompatibility = VersionCompatibility.EXACT


class BlockRole(StrEnum):
    GENERATION = "generation"
    PREPROCESSING = "preprocessing"
    PROCESSING = "processing"
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    AGGREGATION = "aggregation"
    CUSTOM = "custom"


class ExecutionPattern(StrEnum):
    SINGLE = "single"
    MAP = "map"
    BATCH = "batch"
    REDUCE = "reduce"
    MAP_REDUCE = "map_reduce"
    STREAM = "stream"
    SERVICE_CALL = "service_call"
    SUBPROCESS = "subprocess"


class CacheMode(StrEnum):
    DISABLED = "disabled"
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    REFRESH = "refresh"


class CheckpointCapability(StrEnum):
    NONE = "none"
    BLOCK = "block"
    PARTITION = "partition"
    INTERNAL = "internal"


class CodeFingerprintPolicy(StrEnum):
    EXPLICIT = "explicit"
    SOURCE = "source"
    GIT = "git"
    DISTRIBUTION = "distribution"
    CUSTOM = "custom"


class ResourceRequirements(FrozenModel):
    cpu_cores: float = Field(default=1.0, gt=0)
    gpu_count: float = Field(default=0.0, ge=0)
    gpu_memory_mb: int | None = Field(default=None, ge=0)
    ram_mb: int | None = Field(default=None, ge=0)
    scratch_mb: int | None = Field(default=None, ge=0)
    custom: Mapping[str, float] = Field(default_factory=dict)
    concurrency_group: str | None = None
    affinity: tuple[str, ...] = ()
    anti_affinity: tuple[str, ...] = ()
    exclusive: bool = False
    estimated_duration_s: float | None = Field(default=None, ge=0)


class RetryPolicy(FrozenModel):
    max_attempts: int = Field(default=1, ge=1)
    initial_delay_s: float = Field(default=0.0, ge=0)
    exponential_base: float = Field(default=2.0, ge=1)
    maximum_delay_s: float = Field(default=60.0, ge=0)
    jitter_s: float = Field(default=0.0, ge=0)
    retryable_exceptions: tuple[str, ...] = ("RetryableBlockError",)


class SideEffect(FrozenModel):
    name: str
    description: str
    recoverable: bool = False


class BlockSpec(FrozenModel):
    name: str
    version: str
    role: BlockRole = BlockRole.CUSTOM
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]
    parameter_schema: Mapping[str, object] = Field(default_factory=dict)
    input_ports: tuple[PortSpec, ...] = ()
    output_ports: tuple[PortSpec, ...] = ()
    deterministic: bool = True
    idempotent: bool = True
    cache_policy: CacheMode = CacheMode.READ_WRITE
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_s: float | None = Field(default=None, gt=0)
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    execution_pattern: ExecutionPattern = ExecutionPattern.SINGLE
    checkpoint_capability: CheckpointCapability = CheckpointCapability.NONE
    code_fingerprint_policy: CodeFingerprintPolicy = CodeFingerprintPolicy.SOURCE
    code_fingerprint: str
    side_effects: tuple[SideEffect, ...] = ()
    serializer_id: str = "json"
    tags: tuple[str, ...] = ()
    description: str = ""
    external_dependencies: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_retry_semantics(self) -> BlockSpec:
        if (
            self.retry_policy.max_attempts > 1
            and not self.idempotent
            and self.side_effects
            and not all(effect.recoverable for effect in self.side_effects)
        ):
            raise ValueError("non-idempotent block with irreversible side effects cannot retry")
        if self.cache_policy != CacheMode.DISABLED and not self.deterministic:
            raise ValueError("non-deterministic blocks must disable deterministic caching")
        return self

    @property
    def semantic_id(self) -> str:
        return f"{self.name}:{self.version}"


class RunState(StrEnum):
    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PRUNED = "pruned"
    SKIPPED = "skipped"
    CACHED = "cached"
    STALE = "stale"
    ORPHANED = "orphaned"
    RETRY_WAIT = "retry_wait"


class RunKind(StrEnum):
    EXPERIMENT = "experiment"
    ROUND = "round"
    TRIAL = "trial"
    PIPELINE = "pipeline"
    BLOCK = "block"
    PARTITION = "partition"
    CHECKPOINT = "checkpoint"


class MetricDirection(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    TARGET = "target"


class MetricRecord(FrozenModel):
    name: str
    value: float | int | list[float] | ArtifactRef
    direction: MetricDirection | None = None
    step: int | None = None
    split: str | None = None
    aggregation: str | None = None
    uncertainty: Mapping[str, float] = Field(default_factory=dict)
    unit: str | None = None
    tags: Mapping[str, str] = Field(default_factory=dict)
    provenance: Mapping[str, object] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=utc_now)


class ActivationCondition(FrozenModel):
    parameter: str
    operator: Literal["eq", "ne", "in", "not_in", "gt", "ge", "lt", "le"] = "eq"
    value: object

    def matches(self, values: Mapping[str, object]) -> bool:
        actual = values.get(self.parameter)
        match self.operator:
            case "eq":
                return actual == self.value
            case "ne":
                return actual != self.value
            case "in":
                return actual in self.value  # type: ignore[operator]
            case "not_in":
                return actual not in self.value  # type: ignore[operator]
            case "gt":
                return bool(actual > self.value)  # type: ignore[operator]
            case "ge":
                return bool(actual >= self.value)  # type: ignore[operator]
            case "lt":
                return bool(actual < self.value)  # type: ignore[operator]
            case "le":
                return bool(actual <= self.value)  # type: ignore[operator]
        return False


class ParameterBase(FrozenModel):
    name: str
    path: str | None = None
    default: object | None = None
    unit: str | None = None
    description: str = ""
    activation: ActivationCondition | None = None
    quantization: float | int | None = None
    participates_in_hash: bool = True

    @property
    def resolved_path(self) -> str:
        return self.path or self.name


class FloatParameter(ParameterBase):
    kind: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False


class IntegerParameter(ParameterBase):
    kind: Literal["integer"] = "integer"
    low: int
    high: int
    log: bool = False


class CategoricalParameter(ParameterBase):
    kind: Literal["categorical"] = "categorical"
    values: tuple[str | int | float | bool, ...]


class BooleanParameter(ParameterBase):
    kind: Literal["boolean"] = "boolean"
    default: bool = False


class OrdinalParameter(ParameterBase):
    kind: Literal["ordinal"] = "ordinal"
    values: tuple[str | int | float, ...]


class FixedParameter(ParameterBase):
    kind: Literal["fixed"] = "fixed"
    value: object


class DerivedParameter(ParameterBase):
    kind: Literal["derived"] = "derived"
    source_paths: tuple[str, ...]
    expression: str


Parameter = Annotated[
    FloatParameter | IntegerParameter | CategoricalParameter | BooleanParameter | OrdinalParameter | FixedParameter | DerivedParameter,
    Field(discriminator="kind"),
]


class ParameterConstraint(FrozenModel):
    expression: str
    message: str = "parameter constraint failed"


class ParameterSpace(FrozenModel):
    version: str = "1"
    parameters: tuple[Parameter, ...]
    constraints: tuple[ParameterConstraint, ...] = ()
    parent_version: str | None = None
    metadata: Mapping[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_names(self) -> ParameterSpace:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        return self


class ResourcePlan(FrozenModel):
    per_block: Mapping[str, ResourceRequirements] = Field(default_factory=dict)


class TrialPlan(FrozenModel):
    id: UUID = Field(default_factory=uuid4)
    parameters: Mapping[str, object]
    seed: int
    replicate_index: int = 0
    pipeline_version: str
    dataset_version: str | None = None
    resource_plan: ResourcePlan = Field(default_factory=ResourcePlan)
    stop_policy: Mapping[str, object] = Field(default_factory=dict)
    expected_outputs: tuple[str, ...] = ()
    round_id: UUID | None = None
    planner_id: str
    planner_version: str
    fingerprint: str
    metadata: Mapping[str, object] = Field(default_factory=dict)


class Objective(FrozenModel):
    metric: str
    direction: MetricDirection
    target: float | None = None


class Experiment(FrozenModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    version: str = "1"
    description: str = ""
    tags: Mapping[str, str] = Field(default_factory=dict)
    seed: int = 0
    objectives: tuple[Objective, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)


class DecisionKind(StrEnum):
    SELECT = "select"
    PARETO = "pareto"
    REPLICATE = "replicate"
    REFINE = "refine"
    STOP = "stop"
    CONTINUE = "continue"
    PROMOTE = "promote"
    REJECT = "reject"


class DecisionRecord(FrozenModel):
    id: UUID = Field(default_factory=uuid4)
    kind: DecisionKind
    selected_trial_ids: tuple[UUID, ...] = ()
    policy_id: str
    policy_version: str
    inputs_fingerprint: str
    explanation: str
    details: Mapping[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TrialResult(FrozenModel):
    trial_plan: TrialPlan
    state: RunState
    metrics: tuple[MetricRecord, ...] = ()
    outputs: Mapping[str, object] = Field(default_factory=dict)
    error: str | None = None


class ExperimentHistory(FrozenModel):
    trials: tuple[TrialResult, ...] = ()
    decisions: tuple[DecisionRecord, ...] = ()


class CheckpointPayload(FrozenModel):
    version: str = "1"
    block_fingerprint: str
    parameters_fingerprint: str
    input_lineage_fingerprint: str
    progress_cursor: Mapping[str, object] = Field(default_factory=dict)
    rng_states: Mapping[str, object] = Field(default_factory=dict)
    state_artifact: ArtifactRef
    serializer_id: str
    serializer_version: str = "1"
    created_at: datetime = Field(default_factory=utc_now)
    compatibility: Mapping[str, object] = Field(default_factory=dict)
