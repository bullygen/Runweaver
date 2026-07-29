"""Optuna-backed adaptive planning without exposing Optuna trials as domain plans."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from runweaver.artifacts.hashing import fingerprint
from runweaver.domain.models import (
    BooleanParameter,
    CategoricalParameter,
    DerivedParameter,
    ExperimentHistory,
    FixedParameter,
    FloatParameter,
    IntegerParameter,
    OrdinalParameter,
    ParameterSpace,
    TrialPlan,
)
from runweaver.exceptions import BackendUnavailableError, PlanningError
from runweaver.planning.planners import PlanningRequest
from runweaver.planning.space import _safe_expression, validate_constraints


def _optuna():
    try:
        import optuna
    except ImportError as exc:
        raise BackendUnavailableError(
            "Optuna integration requires `pip install runweaver[optuna]`"
        ) from exc
    return optuna


class OptunaPlanner:
    """Materialize immutable plans from TPE, CMA-ES, GP or NSGA-II suggestions."""

    id = "optuna"
    version = "1"

    def __init__(
        self,
        n_trials: int,
        *,
        study_name: str,
        storage: str,
        sampler: str = "tpe",
        directions: tuple[str, ...] = ("maximize",),
        seed: int = 0,
        pruner: str = "median",
    ) -> None:
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.sampler_name = sampler
        self.directions = directions
        self.seed = seed
        self.pruner_name = pruner
        self._active: dict[str, object] = {}

    def _study(self):
        optuna = _optuna()
        samplers = {
            "tpe": lambda: optuna.samplers.TPESampler(seed=self.seed),
            "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=self.seed),
            "gp": lambda: optuna.samplers.GPSampler(seed=self.seed),
            "nsga2": lambda: optuna.samplers.NSGAIISampler(seed=self.seed),
        }
        if self.sampler_name not in samplers:
            raise PlanningError(f"unknown Optuna sampler: {self.sampler_name}")
        pruner = (
            optuna.pruners.MedianPruner()
            if self.pruner_name == "median"
            else optuna.pruners.NopPruner()
        )
        return optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=samplers[self.sampler_name](),
            pruner=pruner,
            directions=list(self.directions),
            load_if_exists=True,
        )

    def propose(
        self,
        space: ParameterSpace,
        history: ExperimentHistory | None = None,
        request: PlanningRequest | None = None,
    ) -> list[TrialPlan]:
        request = request or PlanningRequest(n_trials=self.n_trials, seed=self.seed)
        study = self._study()
        plans: list[TrialPlan] = []
        attempts = 0
        while len(plans) < request.n_trials and attempts < request.n_trials * 10:
            attempts += 1
            trial = study.ask()
            values: dict[str, object] = {}
            try:
                for parameter in space.parameters:
                    if parameter.activation and not parameter.activation.matches(values):
                        continue
                    path = parameter.resolved_path
                    if isinstance(parameter, FloatParameter):
                        values[path] = trial.suggest_float(
                            parameter.name,
                            parameter.low,
                            parameter.high,
                            log=parameter.log,
                            step=float(parameter.quantization) if parameter.quantization else None,
                        )
                    elif isinstance(parameter, IntegerParameter):
                        values[path] = trial.suggest_int(
                            parameter.name,
                            parameter.low,
                            parameter.high,
                            log=parameter.log,
                            step=int(parameter.quantization or 1),
                        )
                    elif isinstance(parameter, CategoricalParameter | OrdinalParameter):
                        values[path] = trial.suggest_categorical(parameter.name, list(parameter.values))
                    elif isinstance(parameter, BooleanParameter):
                        values[path] = trial.suggest_categorical(parameter.name, [False, True])
                    elif isinstance(parameter, FixedParameter):
                        values[path] = parameter.value
                    elif isinstance(parameter, DerivedParameter):
                        names = {name: values[name] for name in parameter.source_paths}
                        values[path] = _safe_expression(parameter.expression, names)
                validate_constraints(space, values)
            except Exception:
                study.tell(trial, state=_optuna().trial.TrialState.FAIL)
                continue
            plan_seed = request.seed + int(trial.number)
            payload = {
                "parameters": values,
                "seed": plan_seed,
                "pipeline_version": request.pipeline_version,
                "dataset_version": request.dataset_version,
                "study_name": self.study_name,
                "optuna_trial_number": trial.number,
                "sampler": self.sampler_name,
            }
            plan_fingerprint = fingerprint(payload)
            plan = TrialPlan(
                id=uuid5(NAMESPACE_URL, f"runweaver:trial:{plan_fingerprint}"),
                parameters=values,
                seed=plan_seed,
                pipeline_version=request.pipeline_version,
                dataset_version=request.dataset_version,
                planner_id=self.id,
                planner_version=self.version,
                fingerprint=plan_fingerprint,
                metadata={
                    "study_name": self.study_name,
                    "optuna_trial_number": trial.number,
                    "sampler": self.sampler_name,
                    "pruner": self.pruner_name,
                },
            )
            self._active[str(plan.id)] = trial
            plans.append(plan)
        if len(plans) != request.n_trials:
            raise PlanningError("Optuna could not produce enough valid constrained trials")
        return plans

    def report_intermediate(self, plan: TrialPlan, value: float, step: int) -> bool:
        trial = self._active.get(str(plan.id))
        if trial is None:
            raise PlanningError("intermediate reporting requires a plan materialized in this process")
        if len(self.directions) != 1:
            return False
        trial.report(value, step)
        return bool(trial.should_prune())

    def tell(
        self,
        plan: TrialPlan,
        values: float | tuple[float, ...] | None,
        *,
        state: str = "complete",
    ) -> None:
        optuna = _optuna()
        study = self._study()
        number = int(plan.metadata["optuna_trial_number"])
        states = {
            "complete": optuna.trial.TrialState.COMPLETE,
            "failed": optuna.trial.TrialState.FAIL,
            "pruned": optuna.trial.TrialState.PRUNED,
        }
        study.tell(number, values=values, state=states[state])
        self._active.pop(str(plan.id), None)
