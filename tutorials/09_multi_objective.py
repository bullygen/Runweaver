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
"""Tutorial 9: quality/latency objectives, a constraint and a Pareto policy."""

# %%
from __future__ import annotations

from runweaver import (
    ExperimentHistory,
    MetricDirection,
    MetricRecord,
    ParetoFrontPolicy,
    RunState,
    TrialPlan,
    TrialResult,
)
from runweaver.artifacts import fingerprint


def make_trial(index: int, quality: float, latency: float, memory: float) -> TrialResult:
    plan = TrialPlan(
        parameters={"width": 8 * (index + 1)},
        seed=index,
        pipeline_version="multi-v1",
        planner_id="manual",
        planner_version="1",
        fingerprint=fingerprint({"index": index}),
    )
    return TrialResult(
        trial_plan=plan,
        state=RunState.COMPLETED,
        metrics=(
            MetricRecord(name="quality", value=quality, direction=MetricDirection.MAXIMIZE),
            MetricRecord(name="latency_ms", value=latency, direction=MetricDirection.MINIMIZE),
            MetricRecord(name="memory_mb", value=memory),
        ),
    )


def main() -> None:
    history = ExperimentHistory(trials=(
        make_trial(0, 0.78, 4.0, 50),
        make_trial(1, 0.86, 7.5, 90),
        make_trial(2, 0.91, 15.0, 140),
        make_trial(3, 0.84, 12.0, 220),  # rejected by memory constraint
    ))
    feasible = history.model_copy(update={
        "trials": tuple(
            trial
            for trial in history.trials
            if next(metric.value for metric in trial.metrics if metric.name == "memory_mb") <= 160
        )
    })
    decision = ParetoFrontPolicy({
        "quality": MetricDirection.MAXIMIZE,
        "latency_ms": MetricDirection.MINIMIZE,
    }).decide(feasible)
    print("Pareto trial ids:", [str(item) for item in decision.selected_trial_ids])
    print("explanation:", decision.explanation)


if __name__ == "__main__":
    main()
