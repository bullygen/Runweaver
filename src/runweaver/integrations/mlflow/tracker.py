"""Optional MLflow tracking sink; never used as the resume source of truth."""

from __future__ import annotations

from collections.abc import Mapping

from runweaver.domain.models import ArtifactRef, MetricRecord, TrialPlan
from runweaver.exceptions import BackendUnavailableError


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise BackendUnavailableError(
            "MLflow integration requires `pip install runweaver[mlflow]`"
        ) from exc
    return mlflow


def _flatten(values: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, name))
        else:
            flattened[name] = value
    return flattened


class MLflowTracker:
    def __init__(self, *, experiment_name: str, tracking_uri: str | None = None) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._runs: dict[str, object] = {}

    def start_trial(self, plan: TrialPlan) -> None:
        mlflow = _mlflow()
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=str(plan.id))
        self._runs[str(plan.id)] = run
        mlflow.set_tags({
            "runweaver.trial_id": str(plan.id),
            "runweaver.trial_fingerprint": plan.fingerprint,
            "runweaver.planner": f"{plan.planner_id}:{plan.planner_version}",
        })
        flattened = _flatten(plan.parameters)
        mlflow.log_params({key: str(value)[:500] for key, value in flattened.items()})
        mlflow.log_text(plan.model_dump_json(indent=2), "runweaver/trial_plan.json")

    def log_metric(self, plan: TrialPlan, metric: MetricRecord) -> None:
        mlflow = _mlflow()
        if isinstance(metric.value, (int, float)):
            mlflow.log_metric(metric.name, float(metric.value), step=metric.step)
        else:
            mlflow.log_text(metric.model_dump_json(indent=2), f"runweaver/metrics/{metric.name}.json")

    def log_artifact(self, plan: TrialPlan, artifact: ArtifactRef) -> None:
        mlflow = _mlflow()
        mlflow.set_tag(f"runweaver.artifact.{artifact.id}", artifact.uri)
        mlflow.log_text(artifact.model_dump_json(indent=2), f"runweaver/artifacts/{artifact.id}.json")

    def end_trial(self, plan: TrialPlan, status: str) -> None:
        mlflow = _mlflow()
        mlflow.end_run(status=status.upper())
        self._runs.pop(str(plan.id), None)
