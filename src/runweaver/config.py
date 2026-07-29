"""Versioned declarative experiment configuration."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from runweaver.domain.models import Experiment, ParameterSpace, ResourceRequirements
from runweaver.exceptions import ConfigurationError
from runweaver.execution import LocalBackend, LocalExecutionConfig, MaterializationMode
from runweaver.pipeline import Pipeline
from runweaver.plugins import PluginRegistry

_ENV_PATTERN = re.compile(r"^\$\{ENV:([A-Z][A-Z0-9_]*)\}$")
_SECRET_KEYS = re.compile(r"(password|passwd|token|secret|credential|api[_-]?key)", re.IGNORECASE)


class BlockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    implementation: str
    depends_on: tuple[str, ...] = ()
    parameters: Mapping[str, object] = Field(default_factory=dict)
    map_over: str | None = None
    parallelism: int = Field(default=1, ge=1)
    resources: ResourceRequirements | None = None


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str = "1"
    blocks: tuple[BlockConfig, ...]


class PlannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal[
        "grid", "random", "full_factorial", "latin_hypercube", "sobol", "halton", "optuna"
    ] = "random"
    n_trials: int = Field(default=1, ge=1)
    seed: int = 0
    options: Mapping[str, object] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_root: str = ".runweaver/artifacts"
    state_database_url: str = "sqlite:///.runweaver/state.db"

    @field_validator("artifact_root", "state_database_url", mode="before")
    @classmethod
    def interpolate_operational_env(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        match = _ENV_PATTERN.match(value)
        if not match:
            return value
        key = match.group(1)
        if key not in os.environ:
            raise ValueError(f"operational environment variable {key} is not set")
        return os.environ[key]


class ExecutorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: LocalBackend = LocalBackend.SYNCHRONOUS
    materialization: MaterializationMode = MaterializationMode.DURABLE
    max_workers: int | None = Field(default=None, ge=1)
    work_dir: Path = Path(".runweaver/work")


class ExperimentConfig(BaseModel):
    """JSON-serializable, versioned configuration without embedded credentials."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"] = "1"
    experiment: Experiment
    pipeline: PipelineConfig
    parameter_space: ParameterSpace | None = None
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    initial_input: Mapping[str, object] = Field(default_factory=dict)
    tracking: Mapping[str, object] = Field(default_factory=dict)
    stop_conditions: Mapping[str, object] = Field(default_factory=dict)
    observability: Mapping[str, object] = Field(default_factory=dict)
    secret_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reject_embedded_secrets(self) -> ExperimentConfig:
        def walk(value: object, path: str = "") -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    if str(key) == "secret_names":
                        continue
                    if (
                        _SECRET_KEYS.search(str(key))
                        and child is not None
                        and not (isinstance(child, str) and child.startswith("secret://"))
                    ):
                        raise ValueError(
                            f"secret value at {child_path} must be a secret://name reference"
                        )
                    walk(child, child_path)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    walk(child, f"{path}[{index}]")

        walk(self.model_dump(mode="python"))
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        try:
            return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigurationError(f"invalid experiment config {path}: {exc}") from exc

    def to_json_schema(self) -> dict[str, object]:
        return self.model_json_schema()

    def build_pipeline(self, registry: PluginRegistry | None = None) -> Pipeline:
        registry = registry or PluginRegistry()
        pipeline = Pipeline(self.pipeline.name, version=self.pipeline.version)
        for block_config in self.pipeline.blocks:
            block = registry.resolve_block(
                block_config.implementation,
                block_config.parameters,
            )
            pipeline.add(
                block,
                id=block_config.id,
                depends_on=block_config.depends_on,
                map_over=block_config.map_over,
                parallelism=block_config.parallelism,
                resources=block_config.resources,
            )
        return pipeline.validate()

    def local_execution_config(self) -> LocalExecutionConfig:
        return LocalExecutionConfig(
            backend=self.executor.backend,
            materialization=self.executor.materialization,
            max_workers=self.executor.max_workers,
            work_dir=self.executor.work_dir,
            artifact_root=self.storage.artifact_root,
            state_database_url=self.storage.state_database_url,
        )
