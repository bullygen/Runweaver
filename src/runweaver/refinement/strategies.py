"""Version-preserving local search-space refinement strategies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np

from runweaver.artifacts.hashing import fingerprint
from runweaver.domain.models import (
    CategoricalParameter,
    ExperimentHistory,
    FloatParameter,
    IntegerParameter,
    Parameter,
    ParameterSpace,
    TrialPlan,
)
from runweaver.exceptions import PlanningError


def _selected_plans(history: ExperimentHistory, selection: Iterable[TrialPlan]) -> list[TrialPlan]:
    plans = list(selection)
    if not plans:
        raise PlanningError("refinement requires at least one selected trial")
    return plans


@dataclass(frozen=True)
class EliteZoomStrategy:
    top_fraction: float = 0.2
    top_k: int | None = None
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    expansion_factor: float = 1.25
    minimum_fraction: float = 0.05
    global_exploration_fraction: float = 0.15
    id: str = "elite_zoom"
    version: str = "1"

    def refine(
        self,
        space: ParameterSpace,
        history: ExperimentHistory,
        selection: Iterable[TrialPlan],
        request: object | None = None,
    ) -> ParameterSpace:
        plans = _selected_plans(history, selection)
        count = self.top_k or max(1, int(np.ceil(len(plans) * self.top_fraction)))
        elite = plans[:count]
        refined: list[Parameter] = []
        for parameter in space.parameters:
            samples = [
                plan.parameters[parameter.resolved_path]
                for plan in elite
                if parameter.resolved_path in plan.parameters
            ]
            if isinstance(parameter, FloatParameter) and samples:
                global_width = parameter.high - parameter.low
                low_q, high_q = np.quantile(np.asarray(samples, dtype=float), [self.lower_quantile, self.upper_quantile])
                center = (float(low_q) + float(high_q)) / 2.0
                width = max(
                    (float(high_q) - float(low_q)) * self.expansion_factor,
                    global_width * self.minimum_fraction,
                )
                low = max(parameter.low, center - width / 2.0)
                high = min(parameter.high, center + width / 2.0)
                refined.append(parameter.model_copy(update={"low": low, "high": high}))
            elif isinstance(parameter, IntegerParameter) and samples:
                global_width = parameter.high - parameter.low
                low_q, high_q = np.quantile(np.asarray(samples, dtype=float), [self.lower_quantile, self.upper_quantile])
                width = max(
                    int(np.ceil((high_q - low_q) * self.expansion_factor)),
                    max(1, int(np.ceil(global_width * self.minimum_fraction))),
                )
                center = round((low_q + high_q) / 2)
                low = max(parameter.low, center - width // 2)
                high = min(parameter.high, max(low + 1, center + int(np.ceil(width / 2))))
                refined.append(parameter.model_copy(update={"low": low, "high": high}))
            elif isinstance(parameter, CategoricalParameter) and samples:
                retained = tuple(value for value in parameter.values if value in set(samples))
                refined.append(parameter.model_copy(update={"values": retained or parameter.values}))
            else:
                refined.append(parameter)
        metadata = dict(space.metadata) | {
            "refinement": self.id,
            "source_space_version": space.version,
            "elite_trial_ids": [str(plan.id) for plan in elite],
            "global_exploration_fraction": self.global_exploration_fraction,
        }
        new_version = f"{space.version}.zoom-{fingerprint(metadata)[:8]}"
        return ParameterSpace(
            version=new_version,
            parent_version=space.version,
            parameters=tuple(refined),
            constraints=space.constraints,
            metadata=metadata,
        )


@dataclass(frozen=True)
class NeighborhoodGridStrategy:
    radius_fraction: float = 0.1
    id: str = "neighborhood_grid"
    version: str = "1"

    def refine(
        self,
        space: ParameterSpace,
        history: ExperimentHistory,
        selection: Iterable[TrialPlan],
        request: object | None = None,
    ) -> ParameterSpace:
        center = _selected_plans(history, selection)[0]
        parameters: list[Parameter] = []
        for parameter in space.parameters:
            value = center.parameters.get(parameter.resolved_path)
            if isinstance(parameter, FloatParameter) and value is not None:
                numeric_value = float(cast(float | int, value))
                radius = (parameter.high - parameter.low) * self.radius_fraction
                parameters.append(parameter.model_copy(update={
                    "low": max(parameter.low, numeric_value - radius),
                    "high": min(parameter.high, numeric_value + radius),
                }))
            elif isinstance(parameter, IntegerParameter) and value is not None:
                numeric_value = int(cast(float | int, value))
                radius = max(1, int(np.ceil((parameter.high - parameter.low) * self.radius_fraction)))
                parameters.append(parameter.model_copy(update={
                    "low": max(parameter.low, numeric_value - radius),
                    "high": min(parameter.high, numeric_value + radius),
                }))
            else:
                parameters.append(parameter)
        return ParameterSpace(
            version=f"{space.version}.grid-{fingerprint(center.parameters)[:8]}",
            parent_version=space.version,
            parameters=tuple(parameters),
            constraints=space.constraints,
            metadata=dict(space.metadata) | {"refinement": self.id, "center_trial_id": str(center.id)},
        )


@dataclass(frozen=True)
class TrustRegionStrategy(NeighborhoodGridStrategy):
    initial_radius_fraction: float = 0.2
    expansion: float = 1.5
    contraction: float = 0.5
    improved: bool = True
    id: str = "trust_region"

    def refine(
        self,
        space: ParameterSpace,
        history: ExperimentHistory,
        selection: Iterable[TrialPlan],
        request: object | None = None,
    ) -> ParameterSpace:
        radius = self.initial_radius_fraction * (self.expansion if self.improved else self.contraction)
        strategy = NeighborhoodGridStrategy(radius_fraction=min(1.0, max(0.001, radius)))
        result = strategy.refine(space, history, selection, request)
        return result.model_copy(update={
            "version": result.version.replace(".grid-", ".trust-"),
            "metadata": dict(result.metadata) | {
                "refinement": self.id,
                "trust_radius_fraction": radius,
                "improved": self.improved,
            },
        })


@dataclass(frozen=True)
class ReplicationStrategy:
    replicates: int = 3
    id: str = "replication"
    version: str = "1"

    def refine(
        self,
        space: ParameterSpace,
        history: ExperimentHistory,
        selection: Iterable[TrialPlan],
        request: object | None = None,
    ) -> ParameterSpace:
        plans = _selected_plans(history, selection)
        return space.model_copy(update={
            "version": f"{space.version}.rep-{self.replicates}",
            "parent_version": space.version,
            "metadata": dict(space.metadata) | {
                "refinement": self.id,
                "replicates": self.replicates,
                "candidate_trial_ids": [str(plan.id) for plan in plans],
            },
        })


@dataclass(frozen=True)
class RobustnessStrategy(NeighborhoodGridStrategy):
    perturbation_fraction: float = 0.03
    id: str = "robustness"

    def refine(
        self,
        space: ParameterSpace,
        history: ExperimentHistory,
        selection: Iterable[TrialPlan],
        request: object | None = None,
    ) -> ParameterSpace:
        result = NeighborhoodGridStrategy(self.perturbation_fraction).refine(
            space, history, selection, request
        )
        return result.model_copy(update={
            "version": result.version.replace(".grid-", ".robust-"),
            "metadata": dict(result.metadata) | {
                "refinement": self.id,
                "perturbation_fraction": self.perturbation_fraction,
            },
        })
