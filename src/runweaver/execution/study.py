"""High-level study orchestration over immutable plans and typed pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from runweaver.domain.models import (
    DecisionRecord,
    Experiment,
    ExperimentHistory,
    ParameterSpace,
    RunState,
    TrialResult,
)
from runweaver.execution.engine import LocalExecutor, PipelineResult
from runweaver.pipeline import Pipeline
from runweaver.planning.planners import PlanningRequest


@dataclass(frozen=True)
class StudyResult:
    experiment: Experiment
    history: ExperimentHistory
    decision: DecisionRecord | None
    pipeline_results: tuple[PipelineResult, ...]
    refined_space: ParameterSpace | None = None


class Study:
    """Plan and execute an experiment without exposing backend runtime types."""

    def __init__(
        self,
        *,
        experiment: Experiment,
        pipeline: Pipeline,
        planner: object,
        parameter_space: ParameterSpace,
        initial_input: BaseModel | Callable[[object], BaseModel],
        executor: LocalExecutor | None = None,
        decision_policy: object | None = None,
        refinement: object | None = None,
    ) -> None:
        self.experiment = experiment
        self.pipeline = pipeline
        self.planner = planner
        self.parameter_space = parameter_space
        self.initial_input = initial_input
        self.executor = executor or LocalExecutor()
        self.decision_policy = decision_policy
        self.refinement = refinement

    def run(
        self,
        *,
        resume: bool = True,
        request: PlanningRequest | None = None,
    ) -> StudyResult:
        plans = self.planner.propose(
            self.parameter_space,
            ExperimentHistory(),
            request,
        )
        trials: list[TrialResult] = []
        pipeline_results: list[PipelineResult] = []
        for plan in plans:
            initial = self.initial_input(plan) if callable(self.initial_input) else self.initial_input
            try:
                result = self.executor.run(
                    self.pipeline,
                    initial,
                    experiment=self.experiment,
                    trial_plan=plan,
                    resume=resume,
                )
                pipeline_results.append(result)
                trials.append(TrialResult(
                    trial_plan=plan,
                    state=result.state,
                    metrics=result.metrics,
                    outputs={
                        node_id: output.model_dump(mode="json")
                        for node_id, output in result.outputs.items()
                    },
                ))
            except Exception as exc:
                trials.append(TrialResult(
                    trial_plan=plan,
                    state=RunState.FAILED,
                    error=repr(exc),
                ))
        history = ExperimentHistory(trials=tuple(trials))
        decision = self.decision_policy.decide(history) if self.decision_policy else None
        refined = None
        if decision and self.refinement and decision.selected_trial_ids:
            selected = [
                trial.trial_plan
                for trial in trials
                if trial.trial_plan.id in decision.selected_trial_ids
            ]
            refined = self.refinement.refine(
                self.parameter_space,
                history,
                selected,
            )
        if decision:
            history = history.model_copy(update={"decisions": (decision,)})
        return StudyResult(
            experiment=self.experiment,
            history=history,
            decision=decision,
            pipeline_results=tuple(pipeline_results),
            refined_space=refined,
        )
