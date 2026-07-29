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
"""Tutorial 4: a user-owned PyTorch loop inside typed Runweaver blocks."""

# %%
from __future__ import annotations

import importlib.util

from pydantic import BaseModel
from runweaver import MetricDirection, MetricRecord, Pipeline, function_block
from runweaver.execution import LocalExecutor, RunContext


class TrainingState(BaseModel):
    x: list[list[float]]
    y: list[int]
    weights: list[list[float]] = []
    predictions: list[int] = []
    accuracy: float = 0.0


def train(inputs: TrainingState, context: RunContext) -> TrainingState:
    import torch
    from runweaver.integrations.pytorch import deterministic_seed, snapshot_training_state

    deterministic_seed(context.seed)
    x = torch.tensor(inputs.x, dtype=torch.float32)
    y = torch.tensor(inputs.y, dtype=torch.long)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    for epoch in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            context.report_metric(MetricRecord(
                name="train_loss",
                value=float(loss.detach()),
                direction=MetricDirection.MINIMIZE,
                step=epoch,
            ))
    snapshot = snapshot_training_state(model, optimizer=optimizer, cursor={"epoch": 30})
    assert snapshot["cursor"] == {"epoch": 30}
    weights = model.weight.detach().cpu().tolist()
    predictions = model(x).argmax(dim=1).tolist()
    accuracy = sum(int(a == b) for a, b in zip(predictions, inputs.y, strict=True)) / len(inputs.y)
    return inputs.model_copy(update={
        "weights": weights,
        "predictions": predictions,
        "accuracy": accuracy,
    })


def main() -> None:
    if importlib.util.find_spec("torch") is None:
        print("SKIP: install `runweaver[pytorch]` to execute the training block")
        return
    data = TrainingState(
        x=[[-1, -1], [-1, 0], [1, 0], [1, 1]],
        y=[0, 0, 1, 1],
    )
    pipeline = Pipeline("torch-classification").then(
        function_block(train, inputs=TrainingState, outputs=TrainingState)
    )
    result = LocalExecutor().run(pipeline, data)
    print("accuracy:", result.final_output.accuracy)
    print("reported learning-curve points:", len(result.metrics))


if __name__ == "__main__":
    main()
