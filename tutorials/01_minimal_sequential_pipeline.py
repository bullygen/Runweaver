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
"""Tutorial 1: a complete synthetic-regression pipeline in one script."""

# %%
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from runweaver import (
    BlockRole,
    LocalExecutionConfig,
    LocalExecutor,
    MaterializationMode,
    MetricDirection,
    MetricRecord,
    Pipeline,
    function_block,
)
from runweaver.execution import RunContext


class RegressionState(BaseModel):
    x: list[float] = []
    y: list[float] = []
    x_mean: float = 0.0
    x_scale: float = 1.0
    slope: float = 0.0
    intercept: float = 0.0
    predictions: list[float] = []
    mse: float | None = None


def generate(inputs: RegressionState, context: RunContext) -> RegressionState:
    rng = context.child_rng("generation")
    x = np.linspace(-2, 2, 48)
    y = 1.75 * x - 0.4 + rng.normal(0, 0.08, len(x))
    return RegressionState(x=x.tolist(), y=y.tolist())


def preprocess(inputs: RegressionState, context: RunContext) -> RegressionState:
    x = np.asarray(inputs.x)
    mean, scale = float(x.mean()), float(x.std())
    return inputs.model_copy(update={
        "x": ((x - mean) / scale).tolist(),
        "x_mean": mean,
        "x_scale": scale,
    })


def fit(inputs: RegressionState, context: RunContext) -> RegressionState:
    slope, intercept = np.polyfit(inputs.x, inputs.y, deg=1)
    return inputs.model_copy(update={"slope": float(slope), "intercept": float(intercept)})


def predict(inputs: RegressionState, context: RunContext) -> RegressionState:
    predictions = inputs.slope * np.asarray(inputs.x) + inputs.intercept
    return inputs.model_copy(update={"predictions": predictions.tolist()})


def evaluate(inputs: RegressionState, context: RunContext) -> RegressionState:
    mse = float(np.mean((np.asarray(inputs.predictions) - np.asarray(inputs.y)) ** 2))
    context.report_metric(MetricRecord(
        name="mse",
        value=mse,
        direction=MetricDirection.MINIMIZE,
        split="train",
        unit="squared target",
    ))
    return inputs.model_copy(update={"mse": mse})


def build_pipeline() -> Pipeline:
    return (
        Pipeline("synthetic-regression")
        .then(function_block(generate, inputs=RegressionState, outputs=RegressionState, role=BlockRole.GENERATION))
        .then(function_block(preprocess, inputs=RegressionState, outputs=RegressionState, role=BlockRole.PREPROCESSING))
        .then(function_block(fit, inputs=RegressionState, outputs=RegressionState, role=BlockRole.TRAINING))
        .then(function_block(predict, inputs=RegressionState, outputs=RegressionState, role=BlockRole.INFERENCE))
        .then(function_block(evaluate, inputs=RegressionState, outputs=RegressionState, role=BlockRole.EVALUATION))
    )


# %%
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ephemeral", "durable", "both"], default="both")
    parser.add_argument("--out")
    args, _ = parser.parse_known_args()
    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="runweaver-t01-"))
    modes = (
        [MaterializationMode.EPHEMERAL, MaterializationMode.DURABLE]
        if args.mode == "both"
        else [MaterializationMode(args.mode)]
    )
    for mode in modes:
        executor = LocalExecutor(LocalExecutionConfig(
            materialization=mode,
            work_dir=root / mode / "work",
            artifact_root=str(root / mode / "artifacts"),
            state_database_url=f"sqlite:///{root / mode / 'state.db'}",
            install_signal_handlers=False,
        ))
        result = executor.run(build_pipeline(), RegressionState())
        print(mode.value, f"mse={result.final_output.mse:.6f}", f"metrics={len(result.metrics)}")
    print(f"outputs={root}")


if __name__ == "__main__":
    main()
