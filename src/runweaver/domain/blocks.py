"""Single block contract, convenience base class and function adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from pydantic import BaseModel

from runweaver.artifacts.hashing import callable_fingerprint
from runweaver.domain.models import (
    BlockRole,
    BlockSpec,
    CacheMode,
    CheckpointCapability,
    CodeFingerprintPolicy,
    ExecutionPattern,
    ResourceRequirements,
    RetryPolicy,
    SideEffect,
)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel, covariant=True)

if TYPE_CHECKING:
    from runweaver.execution.context import RunContext


class Block(Protocol[InputT, OutputT]):
    @property
    def spec(self) -> BlockSpec: ...
    @property
    def input_type(self) -> type[InputT]: ...
    @property
    def output_type(self) -> type[OutputT]: ...
    def execute(self, inputs: InputT, context: RunContext) -> OutputT: ...


class BaseBlock(ABC, Generic[InputT, OutputT]):
    """Base class for blocks that need implementation state or helper methods."""

    spec: BlockSpec
    input_type: type[InputT]
    output_type: type[OutputT]

    @abstractmethod
    def execute(self, inputs: InputT, context: RunContext) -> OutputT:
        """Execute a typed unit of work."""


class FunctionBlock(BaseBlock[InputT, OutputT]):
    def __init__(
        self,
        function: Callable[[InputT, RunContext], OutputT],
        *,
        inputs: type[InputT],
        outputs: type[OutputT],
        spec: BlockSpec,
    ) -> None:
        self.function = function
        self.input_type = inputs
        self.output_type = outputs
        self.spec = spec

    def execute(self, inputs: InputT, context: RunContext) -> OutputT:
        result = self.function(inputs, context)
        return self.output_type.model_validate(result)


def function_block(
    function: Callable[[InputT, RunContext], OutputT],
    *,
    inputs: type[InputT],
    outputs: type[OutputT],
    name: str | None = None,
    version: str = "1",
    role: BlockRole = BlockRole.CUSTOM,
    deterministic: bool = True,
    idempotent: bool = True,
    cache_policy: CacheMode | None = None,
    retry_policy: RetryPolicy | None = None,
    timeout_s: float | None = None,
    resources: ResourceRequirements | None = None,
    execution_pattern: ExecutionPattern = ExecutionPattern.SINGLE,
    checkpoint_capability: CheckpointCapability = CheckpointCapability.NONE,
    serializer_id: str = "json",
    side_effects: tuple[SideEffect, ...] = (),
    tags: tuple[str, ...] = (),
    description: str = "",
) -> FunctionBlock[InputT, OutputT]:
    """Wrap a typed Python callable without binding user code to a backend."""

    semantic_name = name or getattr(function, "__name__", function.__class__.__name__)
    effective_cache = cache_policy or (CacheMode.READ_WRITE if deterministic else CacheMode.DISABLED)
    spec = BlockSpec(
        name=semantic_name,
        version=version,
        role=role,
        input_schema=inputs.model_json_schema(),
        output_schema=outputs.model_json_schema(),
        deterministic=deterministic,
        idempotent=idempotent,
        cache_policy=effective_cache,
        retry_policy=retry_policy or RetryPolicy(),
        timeout_s=timeout_s,
        resources=resources or ResourceRequirements(),
        execution_pattern=execution_pattern,
        checkpoint_capability=checkpoint_capability,
        code_fingerprint_policy=CodeFingerprintPolicy.SOURCE,
        code_fingerprint=callable_fingerprint(function),
        side_effects=side_effects,
        serializer_id=serializer_id,
        tags=tags,
        description=description or (getattr(function, "__doc__", "") or "").strip(),
    )
    return FunctionBlock(function, inputs=inputs, outputs=outputs, spec=spec)
