"""Deterministic grid, random and SciPy QMC experiment planners."""

from __future__ import annotations

import csv
import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from scipy.stats import qmc

from runweaver.artifacts.hashing import fingerprint
from runweaver.domain.models import (
    BooleanParameter,
    CategoricalParameter,
    DerivedParameter,
    ExperimentHistory,
    FixedParameter,
    OrdinalParameter,
    Parameter,
    ParameterSpace,
    TrialPlan,
)
from runweaver.exceptions import PlanningError
from runweaver.planning.space import resolve_row, validate_constraints


@dataclass(frozen=True)
class PlanningRequest:
    n_trials: int
    seed: int = 0
    pipeline_version: str = "1"
    dataset_version: str | None = None
    replicate_count: int = 1


def _sampled(space: ParameterSpace) -> list[Parameter]:
    return [
        parameter
        for parameter in space.parameters
        if not isinstance(parameter, (FixedParameter, DerivedParameter))
    ]


def _materialize(
    planner_id: str,
    planner_version: str,
    space: ParameterSpace,
    unit_design: np.ndarray,
    request: PlanningRequest,
) -> list[TrialPlan]:
    parameters = _sampled(space)
    plans: list[TrialPlan] = []
    for row_index, row in enumerate(unit_design):
        unit_values = {parameter.name: float(row[index]) for index, parameter in enumerate(parameters)}
        try:
            values = resolve_row(space, unit_values)
        except PlanningError:
            continue
        for replicate in range(request.replicate_count):
            seed = int(np.random.SeedSequence([request.seed, row_index, replicate]).generate_state(1)[0])
            payload = {
                "parameters": values,
                "seed": seed,
                "replicate": replicate,
                "pipeline_version": request.pipeline_version,
                "dataset_version": request.dataset_version,
                "planner_id": planner_id,
                "planner_version": planner_version,
                "space_version": space.version,
            }
            plan_fingerprint = fingerprint(payload)
            plans.append(TrialPlan(
                id=uuid5(NAMESPACE_URL, f"runweaver:trial:{plan_fingerprint}"),
                parameters=values,
                seed=seed,
                replicate_index=replicate,
                pipeline_version=request.pipeline_version,
                dataset_version=request.dataset_version,
                planner_id=planner_id,
                planner_version=planner_version,
                fingerprint=plan_fingerprint,
            ))
        if len(plans) >= request.n_trials * request.replicate_count:
            break
    if len(plans) < request.n_trials * request.replicate_count:
        raise PlanningError(
            f"{planner_id} produced {len(plans)} valid plans; "
            f"{request.n_trials * request.replicate_count} requested"
        )
    return plans


class RandomPlanner:
    id = "random"
    version = "1"

    def __init__(self, n_trials: int, *, seed: int = 0) -> None:
        self.n_trials = n_trials
        self.seed = seed

    def propose(
        self,
        space: ParameterSpace,
        history: ExperimentHistory | None = None,
        request: PlanningRequest | None = None,
    ) -> list[TrialPlan]:
        req = request or PlanningRequest(n_trials=self.n_trials, seed=self.seed)
        rng = np.random.default_rng(req.seed)
        design = rng.random((max(req.n_trials * 10, req.n_trials), len(_sampled(space))))
        return _materialize(self.id, self.version, space, design, req)


class GridPlanner:
    id = "grid"
    version = "1"

    def __init__(self, points_per_dimension: int | Mapping[str, int] = 3, *, seed: int = 0) -> None:
        self.points_per_dimension = points_per_dimension
        self.seed = seed

    def propose(
        self,
        space: ParameterSpace,
        history: ExperimentHistory | None = None,
        request: PlanningRequest | None = None,
    ) -> list[TrialPlan]:
        parameters = _sampled(space)
        axes: list[np.ndarray] = []
        for parameter in parameters:
            if isinstance(parameter, (CategoricalParameter, OrdinalParameter)):
                count = len(parameter.values)
            elif isinstance(parameter, BooleanParameter):
                count = 2
            else:
                count = (
                    int(self.points_per_dimension.get(parameter.name, 3))
                    if isinstance(self.points_per_dimension, Mapping)
                    else int(self.points_per_dimension)
                )
            axes.append((np.arange(count, dtype=float) + 0.5) / count)
        design = np.asarray(list(itertools.product(*axes)), dtype=float)
        req = request or PlanningRequest(n_trials=len(design), seed=self.seed)
        if req.n_trials > len(design):
            raise PlanningError(f"grid contains only {len(design)} points")
        return _materialize(self.id, self.version, space, design, req)


class FullFactorialPlanner(GridPlanner):
    id = "full_factorial"


class _QmcPlanner:
    engine_name = ""
    version = "1"

    def __init__(self, n_trials: int, *, seed: int = 0, scramble: bool = True) -> None:
        self.n_trials = n_trials
        self.seed = seed
        self.scramble = scramble

    def _engine(self, dimensions: int, seed: int) -> qmc.QMCEngine:
        raise NotImplementedError

    @property
    def id(self) -> str:
        return self.engine_name

    def propose(
        self,
        space: ParameterSpace,
        history: ExperimentHistory | None = None,
        request: PlanningRequest | None = None,
    ) -> list[TrialPlan]:
        req = request or PlanningRequest(n_trials=self.n_trials, seed=self.seed)
        dimensions = len(_sampled(space))
        if dimensions == 0:
            design = np.zeros((req.n_trials, 0))
        else:
            engine = self._engine(dimensions, req.seed)
            design = engine.random(max(req.n_trials * 4, req.n_trials))
        return _materialize(self.id, self.version, space, design, req)


class LatinHypercubePlanner(_QmcPlanner):
    engine_name = "latin_hypercube"

    def _engine(self, dimensions: int, seed: int) -> qmc.QMCEngine:
        return qmc.LatinHypercube(d=dimensions, scramble=self.scramble, seed=seed)


class SobolPlanner(_QmcPlanner):
    engine_name = "sobol"

    def _engine(self, dimensions: int, seed: int) -> qmc.QMCEngine:
        return qmc.Sobol(d=dimensions, scramble=self.scramble, seed=seed)


class HaltonPlanner(_QmcPlanner):
    engine_name = "halton"

    def _engine(self, dimensions: int, seed: int) -> qmc.QMCEngine:
        return qmc.Halton(d=dimensions, scramble=self.scramble, seed=seed)


class DesignMatrixPlanner:
    id = "design_matrix"
    version = "1"

    def __init__(self, rows: Iterable[Mapping[str, object]], *, seed: int = 0) -> None:
        self.rows = tuple(dict(row) for row in rows)
        self.seed = seed

    @classmethod
    def from_csv(cls, path: str | Path, *, seed: int = 0) -> DesignMatrixPlanner:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return cls(csv.DictReader(handle), seed=seed)

    def to_csv(self, path: str | Path) -> None:
        if not self.rows:
            raise PlanningError("cannot export an empty design matrix")
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def propose(
        self,
        space: ParameterSpace,
        history: ExperimentHistory | None = None,
        request: PlanningRequest | None = None,
    ) -> list[TrialPlan]:
        req = request or PlanningRequest(n_trials=len(self.rows), seed=self.seed)
        if req.n_trials > len(self.rows):
            raise PlanningError("design matrix has fewer rows than requested")
        plans = []
        for row_index, source in enumerate(self.rows[: req.n_trials]):
            values = dict(source)
            validate_constraints(space, values)
            seed = int(np.random.SeedSequence([req.seed, row_index]).generate_state(1)[0])
            payload = {
                "parameters": values,
                "seed": seed,
                "pipeline_version": req.pipeline_version,
                "dataset_version": req.dataset_version,
                "planner_id": self.id,
                "planner_version": self.version,
                "space_version": space.version,
            }
            plan_fingerprint = fingerprint(payload)
            plans.append(TrialPlan(
                id=uuid5(NAMESPACE_URL, f"runweaver:trial:{plan_fingerprint}"),
                parameters=values,
                seed=seed,
                pipeline_version=req.pipeline_version,
                dataset_version=req.dataset_version,
                planner_id=self.id,
                planner_version=self.version,
                fingerprint=plan_fingerprint,
            ))
        return plans
