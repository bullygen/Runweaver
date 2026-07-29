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
"""Tutorial 2: process-parallel partitions, ordered fan-in and partial resume."""

# %%
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pydantic import BaseModel
from runweaver import (
    LocalBackend,
    LocalExecutionConfig,
    LocalExecutor,
    MaterializationMode,
    Pipeline,
    function_block,
)
from runweaver.execution import RunContext


class ArrayBatch(BaseModel):
    arrays: list[list[float]]


def centered_filter(inputs: ArrayBatch, context: RunContext) -> ArrayBatch:
    values = inputs.arrays[0]
    marker = context.work_dir / "injected-once"
    context.work_dir.mkdir(parents=True, exist_ok=True)
    if values[0] == 99 and not marker.exists():
        marker.write_text("failed", encoding="utf-8")
        raise RuntimeError("intentional one-time partition failure")
    mean = sum(values) / len(values)
    return ArrayBatch(arrays=[[value - mean for value in values]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["threads", "processes"], default="threads")
    parser.add_argument("--out")
    args, _ = parser.parse_known_args()
    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="runweaver-t02-"))
    pipeline = Pipeline("partition-filter").then(
        function_block(centered_filter, inputs=ArrayBatch, outputs=ArrayBatch),
        map_over="arrays",
        parallelism=2,
    )
    executor = LocalExecutor(LocalExecutionConfig(
        backend=LocalBackend(args.backend),
        materialization=MaterializationMode.DURABLE,
        work_dir=root / "work",
        artifact_root=str(root / "artifacts"),
        state_database_url=f"sqlite:///{root / 'state.db'}",
        install_signal_handlers=False,
    ))
    source = ArrayBatch(arrays=[[1, 2, 3], [10, 14], [99, 101], [-2, 0, 2]])
    try:
        executor.run(pipeline, source)
    except Exception as exc:
        print("expected partial failure:", exc)
    second = executor.run(pipeline, source, resume=True)
    print("ordered fan-in:", second.final_output.arrays)
    print("partitions reused after failure:", second.resumed_partitions)
    print(f"outputs={root}")


if __name__ == "__main__":
    main()
