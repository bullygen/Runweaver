"""Optional Ray Core executor and resource mapping."""

from __future__ import annotations

from pydantic import BaseModel

from runweaver.domain.models import ResourceRequirements
from runweaver.exceptions import BackendUnavailableError
from runweaver.execution import LocalExecutor
from runweaver.pipeline import Pipeline


def _ray():
    try:
        import ray
    except ImportError as exc:
        raise BackendUnavailableError(
            "Ray integration requires `pip install runweaver[ray]`"
        ) from exc
    return ray


def ray_resource_options(resources: ResourceRequirements) -> dict[str, object]:
    options: dict[str, object] = {
        "num_cpus": resources.cpu_cores,
        "num_gpus": resources.gpu_count,
    }
    if resources.custom:
        options["resources"] = dict(resources.custom)
    if resources.ram_mb:
        options["memory"] = resources.ram_mb * 1024 * 1024
    return options


def _remote_run(executor: LocalExecutor, pipeline: Pipeline, initial_input: BaseModel, kwargs: dict[str, object]):
    return executor.run(pipeline, initial_input, **kwargs)


class RayExecutor:
    """Execute without leaking ``ObjectRef`` into the public return value."""

    def __init__(
        self,
        local_executor: LocalExecutor | None = None,
        *,
        address: str | None = None,
        init_options: dict[str, object] | None = None,
    ) -> None:
        self.local_executor = local_executor or LocalExecutor()
        self.address = address
        self.init_options = init_options or {}

    def run(
        self,
        pipeline: Pipeline,
        initial_input: BaseModel,
        *,
        resources: ResourceRequirements | None = None,
        **kwargs: object,
    ) -> object:
        ray = _ray()
        if not ray.is_initialized():
            ray.init(address=self.address, **self.init_options)
        remote = ray.remote(_remote_run).options(
            **ray_resource_options(resources or ResourceRequirements())
        )
        return ray.get(remote.remote(self.local_executor, pipeline, initial_input, dict(kwargs)))


class PrefectRayExecutor:
    """Prefect flow wrapper using the official Prefect-Ray task runner."""

    def __init__(self, ray_address: str | None = None) -> None:
        self.ray_address = ray_address

    def run(self, pipeline: Pipeline, initial_input: BaseModel, **kwargs: object) -> object:
        try:
            from prefect import flow, task
            from prefect_ray.task_runners import RayTaskRunner
        except ImportError as exc:
            raise BackendUnavailableError(
                "Prefect-Ray requires `pip install runweaver[prefect,ray]`"
            ) from exc

        runner = RayTaskRunner(address=self.ray_address)
        execute = task(LocalExecutor().run, persist_result=False)

        @flow(task_runner=runner, name=f"runweaver-ray:{pipeline.name}")
        def compiled():
            return execute(pipeline, initial_input, **kwargs)

        return compiled()
