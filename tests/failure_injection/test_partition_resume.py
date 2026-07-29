from __future__ import annotations

import pytest
from pydantic import BaseModel
from runweaver import LocalExecutionConfig, LocalExecutor, Pipeline, function_block
from runweaver.exceptions import BlockExecutionError
from runweaver.execution import RunContext


class Items(BaseModel):
    items: list[int]


def fail_once(inputs: Items, context: RunContext) -> Items:
    value = inputs.items[0]
    marker = context.work_dir / "failed-once"
    context.work_dir.mkdir(parents=True, exist_ok=True)
    if value == 2 and not marker.exists():
        marker.write_text("injected", encoding="utf-8")
        raise RuntimeError("injected partition failure")
    return Items(items=[value * 10])


def test_resume_reuses_only_completed_partitions(durable_config: LocalExecutionConfig) -> None:
    pipeline = Pipeline("resume").then(
        function_block(fail_once, inputs=Items, outputs=Items),
        map_over="items",
        parallelism=1,
    )
    executor = LocalExecutor(durable_config)
    with pytest.raises(BlockExecutionError, match="injected"):
        executor.run(pipeline, Items(items=[0, 1, 2, 3]))
    result = executor.run(pipeline, Items(items=[0, 1, 2, 3]), resume=True)
    assert result.final_output == Items(items=[0, 10, 20, 30])
    assert result.resumed_partitions == ("00000000", "00000001")
