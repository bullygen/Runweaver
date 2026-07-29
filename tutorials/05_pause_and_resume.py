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
"""Tutorial 5: cooperative pause and partition-level resume."""

# %%
from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel
from runweaver import (
    CancellationRequested,
    LocalExecutionConfig,
    LocalExecutor,
    MaterializationMode,
    Pipeline,
    function_block,
)
from runweaver.execution import RunContext


class Work(BaseModel):
    items: list[int]


def long_partition(inputs: Work, context: RunContext) -> Work:
    value = inputs.items[0]
    marker = context.work_dir / "pause-once"
    context.work_dir.mkdir(parents=True, exist_ok=True)
    if value == 3 and not marker.exists():
        marker.write_text("safe point reached", encoding="utf-8")
        raise CancellationRequested("tutorial pause at a safe point")
    context.cancellation.raise_if_cancelled()
    return Work(items=[value * value])


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="runweaver-t05-"))
    pipeline = Pipeline("pause-resume").then(
        function_block(long_partition, inputs=Work, outputs=Work),
        map_over="items",
        parallelism=1,
    )
    executor = LocalExecutor(LocalExecutionConfig(
        materialization=MaterializationMode.DURABLE,
        artifact_root=str(root / "artifacts"),
        state_database_url=f"sqlite:///{root / 'state.db'}",
        work_dir=root / "work",
        install_signal_handlers=False,
    ))
    paused = executor.run(pipeline, Work(items=list(range(6))))
    assert paused.state.value == "paused"
    resumed = executor.run(pipeline, Work(items=list(range(6))), resume=True)
    print("state after first process:", paused.state.value)
    print("resumed partitions:", resumed.resumed_partitions)
    print("result:", resumed.final_output.items)
    print(f"durable state={root}")


if __name__ == "__main__":
    main()
