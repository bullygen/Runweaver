from __future__ import annotations

import pytest
from pydantic import BaseModel
from runweaver import (
    Experiment,
    FloatParameter,
    LocalExecutor,
    MetricDirection,
    MetricRecord,
    ParameterSpace,
    Pipeline,
    RandomPlanner,
    Study,
    TopKPolicy,
    function_block,
)
from runweaver.exceptions import CancellationRequested
from runweaver.execution import CancellationToken, RunContext
from runweaver.execution.context import EnvironmentSecrets


class Value(BaseModel):
    value: float


def objective(inputs: Value, context: RunContext) -> Value:
    x = float(context.parameters["x"])
    score = -(x - 0.25) ** 2
    context.report_metric(MetricRecord(
        name="score",
        value=score,
        direction=MetricDirection.MAXIMIZE,
    ))
    return Value(value=score)


def test_study_plans_executes_and_decides() -> None:
    study = Study(
        experiment=Experiment(name="study"),
        pipeline=Pipeline("objective").then(
            function_block(objective, inputs=Value, outputs=Value)
        ),
        planner=RandomPlanner(4, seed=3),
        parameter_space=ParameterSpace(parameters=(
            FloatParameter(name="x", low=-1, high=1),
        )),
        initial_input=Value(value=0),
        executor=LocalExecutor(),
        decision_policy=TopKPolicy("score"),
    )
    result = study.run()
    assert len(result.history.trials) == 4
    assert len(result.decision.selected_trial_ids) == 1


def test_context_rng_cancellation_and_named_secrets(monkeypatch) -> None:
    token = CancellationToken()
    assert token.cancelled is False
    token.cancel("stop")
    with pytest.raises(CancellationRequested, match="stop"):
        token.raise_if_cancelled()
    monkeypatch.setenv("TEST_SECRET", "value")
    assert EnvironmentSecrets().get("TEST_SECRET") == "value"
    with pytest.raises(KeyError, match="not configured"):
        EnvironmentSecrets().get("MISSING_SECRET")
