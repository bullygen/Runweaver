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
"""Tutorial 11: unchanged blocks across local, Prefect, Ray and MLflow adapters."""

# %%
from __future__ import annotations

import argparse
import importlib.util

from pydantic import BaseModel
from runweaver import (
    LocalExecutor,
    Pipeline,
    ResourceRequirements,
    function_block,
)
from runweaver.execution import RunContext


class Value(BaseModel):
    value: float


def square(inputs: Value, context: RunContext) -> Value:
    return Value(value=inputs.value**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["local", "prefect", "ray"],
        default="local",
        help="Ray additionally requires runweaver[ray].",
    )
    args, _ = parser.parse_known_args()
    pipeline = Pipeline("portable").then(
        function_block(
            square,
            inputs=Value,
            outputs=Value,
            resources=ResourceRequirements(cpu_cores=1, custom={"accelerator": 0}),
        )
    )
    if args.backend == "local":
        result = LocalExecutor().run(pipeline, Value(value=3))
    elif args.backend == "prefect":
        from runweaver.integrations.prefect import PrefectExecutor

        result = PrefectExecutor().run(pipeline, Value(value=3))
    else:
        if importlib.util.find_spec("ray") is None:
            print("SKIP: install `runweaver[ray]` for the Ray execution branch")
            return
        from runweaver.integrations.ray import RayExecutor

        result = RayExecutor().run(pipeline, Value(value=3))
    print("backend:", args.backend, "value:", result.final_output.value)
    print("the block imports no Prefect, Ray or MLflow types")
    if importlib.util.find_spec("mlflow") is None:
        print("MLflow sink available after `pip install runweaver[mlflow]`")


if __name__ == "__main__":
    main()
