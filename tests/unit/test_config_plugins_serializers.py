from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from runweaver import ConfigurationError, ExperimentConfig
from runweaver.artifacts.serializers import (
    BytesSerializer,
    JsonSerializer,
    NumpySerializer,
    SerializerRegistry,
    TextSerializer,
)
from runweaver.builtins import scale_factory
from runweaver.plugins import PluginRegistry


def test_declarative_example_builds_and_schema_is_versioned() -> None:
    config = ExperimentConfig.from_json(Path("examples/experiment.json"))
    pipeline = config.build_pipeline()
    assert [node.id for node in pipeline.topological_order()] == ["generate", "scale", "evaluate"]
    assert config.to_json_schema()["title"] == "ExperimentConfig"
    assert config.local_execution_config().materialization.value == "durable"


def test_secret_and_unknown_plugin_are_rejected(monkeypatch) -> None:
    payload = json.loads(Path("examples/experiment.json").read_text())
    payload["tracking"] = {"api_token": "live-token"}
    with pytest.raises(ValidationError, match="secret://"):
        ExperimentConfig.model_validate(payload)
    registry = PluginRegistry(discover=False)
    with pytest.raises(ConfigurationError, match="unknown block"):
        registry.resolve_block("missing")
    registry.register_block("x", scale_factory)
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register_block("x", scale_factory)


def test_operational_environment_interpolation(monkeypatch) -> None:
    payload = json.loads(Path("examples/experiment.json").read_text())
    monkeypatch.setenv("RUNWEAVER_TEST_ARTIFACTS", "memory://configured")
    payload["storage"]["artifact_root"] = "${ENV:RUNWEAVER_TEST_ARTIFACTS}"
    config = ExperimentConfig.model_validate(payload)
    assert config.storage.artifact_root == "memory://configured"


def test_safe_serializer_contracts() -> None:
    json_serializer = JsonSerializer()
    assert json_serializer.loads(json_serializer.dumps({"b": 2, "a": 1})) == {"a": 1, "b": 2}
    assert BytesSerializer().loads(BytesSerializer().dumps(b"x")) == b"x"
    assert TextSerializer().loads(TextSerializer().dumps("текст")) == "текст"
    array = np.arange(5)
    restored = NumpySerializer().loads(NumpySerializer().dumps(array))
    assert np.array_equal(restored, array)
    registry = SerializerRegistry()
    assert registry.get("json").id == "json"
    with pytest.raises(Exception, match="not registered"):
        registry.get("unknown")
