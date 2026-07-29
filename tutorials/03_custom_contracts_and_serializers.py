# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""Tutorial 3: custom schemas, serializer registration and early port errors."""

# %%
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel
from runweaver import (
    LocalExecutionConfig,
    LocalExecutor,
    MaterializationMode,
    Pipeline,
    PortCompatibilityError,
    function_block,
)
from runweaver.execution import RunContext


class PointCloud(BaseModel):
    schema_version: str = "1"
    points: list[tuple[float, float]]


class BoundingBox(BaseModel):
    schema_version: str = "1"
    minimum: tuple[float, float]
    maximum: tuple[float, float]


class PointCloudSerializer:
    id = "point-cloud-json"
    version = "1"
    media_type = "application/vnd.runweaver.point-cloud+json"

    def dumps(self, value: object) -> bytes:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        return json.dumps({"format": self.id, "payload": value}).encode()

    def loads(self, payload: bytes) -> object:
        return json.loads(payload)["payload"]


def translate(inputs: PointCloud, context: RunContext) -> PointCloud:
    return PointCloud(points=[(x + 1, y - 1) for x, y in inputs.points])


def incompatible(inputs: BoundingBox, context: RunContext) -> BoundingBox:
    return inputs


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="runweaver-t03-"))
    block = function_block(
        translate,
        inputs=PointCloud,
        outputs=PointCloud,
        serializer_id="point-cloud-json",
    )
    executor = LocalExecutor(LocalExecutionConfig(
        materialization=MaterializationMode.DURABLE,
        artifact_root=str(root / "artifacts"),
        state_database_url=f"sqlite:///{root / 'state.db'}",
        work_dir=root / "work",
        install_signal_handlers=False,
    ))
    executor.serializers.register(PointCloudSerializer())
    result = executor.run(Pipeline("point-cloud").then(block), PointCloud(points=[(0, 0), (2, 4)]))
    print("translated:", result.final_output.points)

    try:
        Pipeline("invalid").then(block).then(
            function_block(incompatible, inputs=BoundingBox, outputs=BoundingBox)
        ).validate()
    except PortCompatibilityError as exc:
        print("expected preflight error:", exc)
    else:
        raise AssertionError("incompatible ports were not rejected")


if __name__ == "__main__":
    main()
