"""Prefect 3 orchestration adapter preserving Runweaver domain semantics."""

from __future__ import annotations

from pydantic import BaseModel

from runweaver.domain.models import Experiment, TrialPlan
from runweaver.exceptions import BackendUnavailableError
from runweaver.execution import LocalExecutor
from runweaver.pipeline import Pipeline


def _prefect():
    try:
        import prefect
    except ImportError as exc:
        raise BackendUnavailableError(
            "Prefect integration requires `pip install runweaver[prefect]`"
        ) from exc
    return prefect


class PrefectExecutor:
    """Run a domain pipeline in a Prefect flow.

    Domain state, artifact commits, cache and resume remain owned by the wrapped
    local executor. Prefect provides orchestration lifecycle and UI visibility.
    """

    def __init__(self, local_executor: LocalExecutor | None = None) -> None:
        self.local_executor = local_executor or LocalExecutor()

    def compile(self, pipeline: Pipeline):
        prefect = _prefect()
        run_task = prefect.task(
            self.local_executor.run,
            name=f"runweaver:{pipeline.name}",
            persist_result=False,
        )

        @prefect.flow(name=f"runweaver:{pipeline.name}:{pipeline.version}")
        def compiled(
            initial_input: BaseModel,
            experiment: Experiment | None = None,
            trial_plan: TrialPlan | None = None,
            resume: bool = True,
        ):
            return run_task(
                pipeline,
                initial_input,
                experiment=experiment,
                trial_plan=trial_plan,
                resume=resume,
            )

        return compiled

    def run(self, pipeline: Pipeline, initial_input: BaseModel, **kwargs: object) -> object:
        return self.compile(pipeline)(initial_input, **kwargs)

    @staticmethod
    def reconcile(prefect_state: object, domain_state: str) -> dict[str, str]:
        return {
            "prefect_state": getattr(prefect_state, "name", type(prefect_state).__name__),
            "domain_state": domain_state,
        }
