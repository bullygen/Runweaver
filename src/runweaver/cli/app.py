"""Typer command line interface for validation, planning, execution and recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from runweaver.config import ExperimentConfig
from runweaver.domain.models import ExperimentHistory, ParameterSpace, RunState
from runweaver.execution import LocalExecutor
from runweaver.persistence import SqlAlchemyStateStore
from runweaver.planning import (
    FullFactorialPlanner,
    GridPlanner,
    HaltonPlanner,
    LatinHypercubePlanner,
    PlanningRequest,
    RandomPlanner,
    SobolPlanner,
)

app = typer.Typer(no_args_is_help=True, help="Typed, durable computational experiment pipelines.")
artifacts_app = typer.Typer(no_args_is_help=True)
inspect_app = typer.Typer(no_args_is_help=True)
app.add_typer(artifacts_app, name="artifacts")
app.add_typer(inspect_app, name="inspect")


def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _planner(config: ExperimentConfig):
    values = config.planner
    if values.kind == "grid":
        return GridPlanner(values.options.get("points_per_dimension", 3), seed=values.seed)
    if values.kind == "full_factorial":
        return FullFactorialPlanner(values.options.get("points_per_dimension", 3), seed=values.seed)
    if values.kind == "random":
        return RandomPlanner(values.n_trials, seed=values.seed)
    if values.kind == "latin_hypercube":
        return LatinHypercubePlanner(values.n_trials, seed=values.seed)
    if values.kind == "sobol":
        return SobolPlanner(values.n_trials, seed=values.seed)
    if values.kind == "halton":
        return HaltonPlanner(values.n_trials, seed=values.seed)
    if values.kind == "optuna":
        from runweaver.integrations.optuna import OptunaPlanner

        return OptunaPlanner(
            values.n_trials,
            study_name=str(values.options.get("study_name", config.experiment.name)),
            storage=str(values.options.get("storage", config.storage.state_database_url)),
            sampler=str(values.options.get("sampler", "tpe")),
            seed=values.seed,
        )
    raise typer.BadParameter(f"unknown planner: {values.kind}")


@app.command("validate")
def validate(config_path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    config = ExperimentConfig.from_json(config_path)
    pipeline = config.build_pipeline()
    _emit({
        "valid": True,
        "schema_version": config.schema_version,
        "pipeline": pipeline.describe(),
    })


@app.command("schema")
def schema(output: Annotated[Path | None, typer.Option("--output", "-o")] = None) -> None:
    payload = ExperimentConfig.model_json_schema()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        output.write_text(encoded + "\n", encoding="utf-8")
    else:
        typer.echo(encoded)


@app.command("plan")
def plan(
    config_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("plan.json"),
) -> None:
    config = ExperimentConfig.from_json(config_path)
    space = config.parameter_space or ParameterSpace(parameters=())
    plans = _planner(config).propose(
        space,
        ExperimentHistory(),
        PlanningRequest(
            n_trials=config.planner.n_trials,
            seed=config.planner.seed,
            pipeline_version=config.pipeline.version,
        ),
    )
    output.write_text(
        json.dumps(
            {"schema_version": "1", "trials": [item.model_dump(mode="json") for item in plans]},
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    _emit({"trial_count": len(plans), "output": str(output)})


@app.command("run")
def run(
    config_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
) -> None:
    config = ExperimentConfig.from_json(config_path)
    pipeline = config.build_pipeline()
    roots = [node for node in pipeline.topological_order() if not node.dependencies]
    if len(roots) != 1:
        raise typer.BadParameter("CLI run currently requires exactly one root block")
    initial = roots[0].block.input_type.model_validate(config.initial_input)
    executor = LocalExecutor(config.local_execution_config())
    if config.parameter_space:
        plans = _planner(config).propose(
            config.parameter_space,
            ExperimentHistory(),
            PlanningRequest(
                n_trials=config.planner.n_trials,
                seed=config.planner.seed,
                pipeline_version=config.pipeline.version,
            ),
        )
    else:
        plans = [None]
    results = [
        executor.run(
            pipeline,
            initial,
            experiment=config.experiment,
            trial_plan=trial_plan,
            resume=resume,
        )
        for trial_plan in plans
    ]
    _emit({
        "experiment_id": str(config.experiment.id),
        "runs": [
            {
                "pipeline_run_id": result.pipeline_run_id,
                "state": result.state,
                "final_output": result.final_output.model_dump(mode="json"),
                "cache_hits": result.cache_hits,
                "resumed_partitions": result.resumed_partitions,
            }
            for result in results
        ],
    })


def _store(database_url: str) -> SqlAlchemyStateStore:
    store = SqlAlchemyStateStore(database_url)
    store.initialize()
    return store


@app.command("status")
def status(
    run_id: str,
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
) -> None:
    record = _store(database_url).get_run(run_id)
    if not record:
        raise typer.BadParameter(f"unknown run id: {run_id}")
    _emit(record)


@inspect_app.command("trial")
def inspect_trial(
    trial_id: str,
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
) -> None:
    store = _store(database_url)
    record = store.get_run(trial_id)
    if not record:
        raise typer.BadParameter(f"unknown trial id: {trial_id}")
    _emit({
        "run": record,
        "children": store.list_runs(parent_id=trial_id),
        "metrics": store.metrics(trial_id),
        "events": store.events(trial_id),
    })


@app.command("pause")
def pause(
    run_id: str,
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
) -> None:
    store = _store(database_url)
    record = store.get_run(run_id)
    if not record:
        raise typer.BadParameter(f"unknown run id: {run_id}")
    if record.state == RunState.RUNNING:
        record = store.transition(run_id, RunState.PAUSING)
    _emit({"run_id": run_id, "state": record.state, "note": "workers pause at their next safe point"})


@app.command("cancel")
def cancel(
    run_id: str,
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
) -> None:
    store = _store(database_url)
    record = store.get_run(run_id)
    if not record:
        raise typer.BadParameter(f"unknown run id: {run_id}")
    record = store.transition(run_id, RunState.CANCELLED)
    _emit({"run_id": run_id, "state": record.state})


@app.command("resume")
def resume_command(
    config_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Resume from durable block/partition state after validating the config."""

    run(config_path, resume=True)


@app.command("retry")
def retry(
    run_id: str,
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
) -> None:
    """Return failed, paused or orphaned work to the durable queue."""

    store = _store(database_url)
    record = store.get_run(run_id)
    if not record:
        raise typer.BadParameter(f"unknown run id: {run_id}")
    retryable = {
        RunState.FAILED,
        RunState.PAUSED,
        RunState.RETRY_WAIT,
        RunState.STALE,
        RunState.ORPHANED,
    }
    if record.state not in retryable:
        raise typer.BadParameter(f"run {run_id} in state {record.state.value} is not retryable")
    record = store.transition(
        run_id,
        RunState.QUEUED,
        metadata_update={"manual_retry": True},
    )
    _emit({"run_id": run_id, "state": record.state})


@app.command("reconcile")
def reconcile(
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
    requeue: Annotated[bool, typer.Option("--requeue/--mark-only")] = False,
) -> None:
    """Mark expired worker leases stale and optionally return them to the queue."""

    store = _store(database_url)
    stale_ids = store.reconcile_stale()
    requeued: list[str] = []
    if requeue:
        for run_id in stale_ids:
            store.transition(
                run_id,
                RunState.QUEUED,
                metadata_update={"reconciled_retry": True},
            )
            requeued.append(run_id)
    _emit({"stale": stale_ids, "requeued": requeued})


@artifacts_app.command("verify")
def verify_artifacts(
    run_id: str,
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
    artifact_root: Annotated[str, typer.Option("--artifact-root")] = ".runweaver/artifacts",
) -> None:
    from runweaver.artifacts import FsspecArtifactStore

    store = _store(database_url)
    artifact_store = FsspecArtifactStore(artifact_root)
    artifacts = store.artifacts_under(run_id)
    failures = []
    for artifact in artifacts:
        try:
            artifact_store.verify(artifact)
        except Exception as exc:
            failures.append({"artifact_id": str(artifact.id), "error": repr(exc)})
    _emit({"run_id": run_id, "verified": len(artifacts) - len(failures), "failures": failures})
    if failures:
        raise typer.Exit(code=2)


@app.command("lineage")
def lineage(
    artifact_id: str,
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
) -> None:
    artifact = _store(database_url).get_artifact(artifact_id)
    if not artifact:
        raise typer.BadParameter(f"unknown artifact id: {artifact_id}")
    _emit({"artifact": artifact, "lineage": artifact.lineage})


@app.command("export")
def export_run(
    run_id: str,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("experiment-export.json"),
    database_url: Annotated[str, typer.Option("--database-url")] = "sqlite:///.runweaver/state.db",
) -> None:
    store = _store(database_url)
    records = [store.get_run(item) for item in store.descendant_ids(run_id)]
    payload = {
        "schema_version": "1",
        "runs": [record.model_dump(mode="json") for record in records if record],
        "artifacts": [artifact.model_dump(mode="json") for artifact in store.artifacts_under(run_id)],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    _emit({"output": str(output), "run_count": len(payload["runs"])})


if __name__ == "__main__":
    app()
