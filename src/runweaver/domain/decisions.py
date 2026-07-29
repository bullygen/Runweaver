"""Pure selection policies separated from metric computation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from uuid import UUID

from runweaver.artifacts.hashing import fingerprint
from runweaver.domain.models import (
    DecisionKind,
    DecisionRecord,
    ExperimentHistory,
    MetricDirection,
    RunState,
    TrialResult,
)


def scalar_metric(trial: TrialResult, name: str) -> float | None:
    for metric in reversed(trial.metrics):
        if metric.name == name and isinstance(metric.value, (int, float)):
            return float(metric.value)
    return None


@dataclass(frozen=True)
class TopKPolicy:
    objective: str
    direction: MetricDirection = MetricDirection.MAXIMIZE
    k: int = 1
    id: str = "top_k"
    version: str = "1"

    def decide(self, history: ExperimentHistory, context: object | None = None) -> DecisionRecord:
        completed = [trial for trial in history.trials if trial.state == RunState.COMPLETED]
        scored = [
            (value, trial)
            for trial in completed
            if (value := scalar_metric(trial, self.objective)) is not None
        ]
        scored.sort(key=lambda item: item[0], reverse=self.direction != MetricDirection.MINIMIZE)
        selected = tuple(trial.trial_plan.id for _, trial in scored[: self.k])
        return DecisionRecord(
            kind=DecisionKind.SELECT,
            selected_trial_ids=selected,
            policy_id=self.id,
            policy_version=self.version,
            inputs_fingerprint=fingerprint(history),
            explanation=f"selected top {len(selected)} trials by {self.objective}",
        )


@dataclass(frozen=True)
class BestFeasiblePolicy(TopKPolicy):
    constraints: Mapping[str, tuple[str, float]] | None = None
    id: str = "best_feasible"

    def decide(self, history: ExperimentHistory, context: object | None = None) -> DecisionRecord:
        constraints = self.constraints or {}
        feasible = []
        for trial in history.trials:
            if trial.state != RunState.COMPLETED:
                continue
            accepted = True
            for metric_name, (operator, threshold) in constraints.items():
                value = scalar_metric(trial, metric_name)
                if value is None:
                    accepted = False
                    break
                accepted &= value >= threshold if operator in ("ge", ">=") else value <= threshold
            if accepted:
                feasible.append(trial)
        decision = super().decide(
            ExperimentHistory(trials=tuple(feasible), decisions=history.decisions),
            context,
        )
        return decision.model_copy(update={
            "policy_id": self.id,
            "explanation": f"selected best feasible trial by {self.objective}; constraints={constraints}",
        })


@dataclass(frozen=True)
class ThresholdPolicy:
    metric: str
    threshold: float
    operator: str = "ge"
    id: str = "threshold"
    version: str = "1"

    def decide(self, history: ExperimentHistory, context: object | None = None) -> DecisionRecord:
        selected = []
        for trial in history.trials:
            value = scalar_metric(trial, self.metric)
            if value is None:
                continue
            accepted = value >= self.threshold if self.operator in ("ge", ">=") else value <= self.threshold
            if accepted:
                selected.append(trial.trial_plan.id)
        return DecisionRecord(
            kind=DecisionKind.SELECT,
            selected_trial_ids=tuple(selected),
            policy_id=self.id,
            policy_version=self.version,
            inputs_fingerprint=fingerprint(history),
            explanation=f"{len(selected)} trials passed {self.metric} {self.operator} {self.threshold}",
        )


@dataclass(frozen=True)
class ParetoFrontPolicy:
    objectives: Mapping[str, MetricDirection]
    id: str = "pareto_front"
    version: str = "1"

    def decide(self, history: ExperimentHistory, context: object | None = None) -> DecisionRecord:
        candidates: list[tuple[TrialResult, tuple[float, ...]]] = []
        for trial in history.trials:
            values = tuple(scalar_metric(trial, name) for name in self.objectives)
            if all(value is not None for value in values):
                numeric_values = cast(tuple[float, ...], values)
                normalized = tuple(
                    float(value) if direction == MetricDirection.MINIMIZE else -float(value)
                    for value, direction in zip(
                        numeric_values, self.objectives.values(), strict=True
                    )
                )
                candidates.append((trial, normalized))
        selected: list[UUID] = []
        for index, (trial, values) in enumerate(candidates):
            dominated = any(
                other_index != index
                and all(
                    other <= value
                    for other, value in zip(other_values, values, strict=True)
                )
                and any(
                    other < value
                    for other, value in zip(other_values, values, strict=True)
                )
                for other_index, (_, other_values) in enumerate(candidates)
            )
            if not dominated:
                selected.append(trial.trial_plan.id)
        return DecisionRecord(
            kind=DecisionKind.PARETO,
            selected_trial_ids=tuple(selected),
            policy_id=self.id,
            policy_version=self.version,
            inputs_fingerprint=fingerprint(history),
            explanation=f"computed Pareto front for {tuple(self.objectives)}",
        )
