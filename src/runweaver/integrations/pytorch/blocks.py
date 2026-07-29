"""Optional PyTorch helpers for user-authored training and inference blocks."""

from __future__ import annotations

import random
from abc import ABC
from collections.abc import Mapping
from typing import Any

import numpy as np

from runweaver.domain.blocks import BaseBlock
from runweaver.domain.models import ModelRef, ModelSpec
from runweaver.exceptions import BackendUnavailableError, CheckpointCompatibilityError


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "PyTorch integration requires `pip install runweaver[pytorch]`"
        ) from exc
    return torch


def deterministic_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


def resolve_device(requested: str = "auto") -> object:
    torch = _torch()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def snapshot_training_state(
    model: object,
    *,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    cursor: Mapping[str, object] | None = None,
) -> dict[str, object]:
    torch = _torch()
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cursor": dict(cursor or {}),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    for name, value in (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        if value is not None:
            payload[name] = value.state_dict()
    return payload


def restore_training_state(
    payload: Mapping[str, object],
    model: object,
    *,
    optimizer: object | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
) -> Mapping[str, object]:
    torch = _torch()
    if "model" not in payload:
        raise CheckpointCompatibilityError("checkpoint lacks model state")
    model.load_state_dict(payload["model"])
    for name, value in (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        if value is not None and name in payload:
            value.load_state_dict(payload[name])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_states" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])
    if "python_rng_state" in payload:
        random.setstate(payload["python_rng_state"])
    return payload.get("cursor", {})


class TorchTrainingBlock(BaseBlock[Any, Any], ABC):
    """Convenience base; users retain ownership of their training loop."""


class TorchInferenceBlock(BaseBlock[Any, Any], ABC):
    """Convenience base for typed inference blocks."""


class TorchMetricsAdapter:
    def __init__(self, metric: object) -> None:
        self.metric = metric

    def update(self, *args: object, **kwargs: object) -> None:
        self.metric.update(*args, **kwargs)

    def compute(self) -> object:
        value = self.metric.compute()
        return value.detach().cpu().tolist() if hasattr(value, "detach") else value

    def reset(self) -> None:
        self.metric.reset()


def torch_model_ref(uri: str, content_hash: str, metadata: Mapping[str, object] | None = None) -> ModelRef:
    return ModelRef(
        uri=uri,
        content_hash=content_hash,
        spec=ModelSpec(
            type_name="torch_state_dict",
            schema_version="1",
            framework="pytorch",
        ),
        metadata=dict(metadata or {}),
    )
