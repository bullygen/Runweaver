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
"""Tutorial 10: custom planner, decision, refinement and plugin registration."""

# %%
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from runweaver import (
    DecisionRecord,
    ExperimentHistory,
    FloatParameter,
    ParameterSpace,
    Pipeline,
    PlanningRequest,
    RunState,
    TrialPlan,
    TrialResult,
    function_block,
)
from runweaver.artifacts import fingerprint
from runweaver.domain.models import DecisionKind
from runweaver.execution import RunContext
from runweaver.plugins import PluginRegistry


class Value(BaseModel):
    value: float


def offset_factory(parameters: dict[str, object]):
    offset = float(parameters.get("offset", 0))

    def offset_value(inputs: Value, context: RunContext) -> Value:
        return Value(value=inputs.value + offset)

    return function_block(offset_value, inputs=Value, outputs=Value, name="custom-offset")


@dataclass(frozen=True)
class EndpointPlanner:
    id: str = "endpoint"
    version: str = "1"

    def propose(self, space, history, request=None):
        request = request or PlanningRequest(n_trials=2)
        parameter = space.parameters[0]
        rows = [{"x": parameter.low}, {"x": parameter.high}]
        return [
            TrialPlan(
                parameters=row,
                seed=request.seed + index,
                pipeline_version=request.pipeline_version,
                planner_id=self.id,
                planner_version=self.version,
                fingerprint=fingerprint(row),
            )
            for index, row in enumerate(rows)
        ]


@dataclass(frozen=True)
class PositivePolicy:
    id: str = "positive"
    version: str = "1"

    def decide(self, history, context=None):
        selected = tuple(
            trial.trial_plan.id
            for trial in history.trials
            if float(trial.trial_plan.parameters["x"]) > 0
        )
        return DecisionRecord(
            kind=DecisionKind.SELECT,
            selected_trial_ids=selected,
            policy_id=self.id,
            policy_version=self.version,
            inputs_fingerprint=fingerprint(history),
            explanation="selected positive endpoint plans",
        )


@dataclass(frozen=True)
class HalfIntervalRefinement:
    id: str = "half_interval"
    version: str = "1"

    def refine(self, space, history, selection, request=None):
        parameter = space.parameters[0]
        center = float(next(iter(selection)).parameters["x"])
        radius = (parameter.high - parameter.low) / 4
        child_parameter = parameter.model_copy(update={
            "low": max(parameter.low, center - radius),
            "high": min(parameter.high, center + radius),
        })
        return space.model_copy(update={
            "version": f"{space.version}.half",
            "parent_version": space.version,
            "parameters": (child_parameter,),
        })


def main() -> None:
    registry = PluginRegistry(discover=False)
    registry.register_block("example.offset", offset_factory)
    block = registry.resolve_block("example.offset", {"offset": 2.5})
    result = __import__("runweaver").LocalExecutor().run(
        Pipeline("plugin").then(block),
        Value(value=1.0),
    )
    space = ParameterSpace(parameters=(FloatParameter(name="x", low=-2, high=2),))
    plans = EndpointPlanner().propose(space, ExperimentHistory())
    history = ExperimentHistory(trials=tuple(
        TrialResult(trial_plan=plan, state=RunState.COMPLETED) for plan in plans
    ))
    decision = PositivePolicy().decide(history)
    selected = [plan for plan in plans if plan.id in decision.selected_trial_ids]
    child = HalfIntervalRefinement().refine(space, history, selected)
    # A real package publishes the factory under [project.entry-points."runweaver.blocks"].
    print("plugin output:", result.final_output.value)
    print("custom plans:", [dict(plan.parameters) for plan in plans])
    print("custom decision:", decision.explanation)
    print("custom refined bounds:", child.parameters[0].low, child.parameters[0].high)
    print("entry-point group: runweaver.blocks")


if __name__ == "__main__":
    main()
