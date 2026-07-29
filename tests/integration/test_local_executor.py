from __future__ import annotations

from pydantic import BaseModel
from runweaver import (
    Experiment,
    LocalBackend,
    LocalExecutionConfig,
    LocalExecutor,
    MaterializationMode,
    Pipeline,
    function_block,
)
from runweaver.execution import RunContext


class Numbers(BaseModel):
    items: list[int]


class Total(BaseModel):
    total: int


def double(inputs: Numbers, context: RunContext) -> Numbers:
    return Numbers(items=[2 * item for item in inputs.items])


def total(inputs: Numbers, context: RunContext) -> Total:
    return Total(total=sum(inputs.items))


def test_local_sequential_pipeline_is_typed_and_ephemeral(tmp_path) -> None:
    pipeline = (
        Pipeline("numbers")
        .then(function_block(double, inputs=Numbers, outputs=Numbers))
        .then(function_block(total, inputs=Numbers, outputs=Total))
    )
    executor = LocalExecutor(LocalExecutionConfig(
        materialization=MaterializationMode.EPHEMERAL,
        work_dir=tmp_path / "work",
        artifact_root=str(tmp_path / "artifacts"),
        install_signal_handlers=False,
    ))
    result = executor.run(pipeline, Numbers(items=[1, 2, 3]))
    assert result.final_output == Total(total=12)
    assert not (tmp_path / "artifacts" / "objects" / "sha256").exists()


def test_process_map_preserves_order_and_collects(tmp_path) -> None:
    pipeline = Pipeline("map").then(
        function_block(double, inputs=Numbers, outputs=Numbers),
        map_over="items",
        parallelism=2,
    )
    executor = LocalExecutor(LocalExecutionConfig(
        backend=LocalBackend.PROCESSES,
        materialization=MaterializationMode.EPHEMERAL,
        work_dir=tmp_path / "work",
        artifact_root=str(tmp_path / "artifacts"),
        install_signal_handlers=False,
    ))
    result = executor.run(pipeline, Numbers(items=[4, 1, 3]))
    assert result.final_output == Numbers(items=[8, 2, 6])


def test_durable_second_run_uses_cache(durable_config: LocalExecutionConfig) -> None:
    pipeline = Pipeline("cache").then(
        function_block(double, inputs=Numbers, outputs=Numbers)
    )
    executor = LocalExecutor(durable_config)
    experiment = Experiment(name="cache")
    first = executor.run(pipeline, Numbers(items=[2]), experiment=experiment)
    second = executor.run(pipeline, Numbers(items=[2]), experiment=experiment)
    assert first.cache_hits == ()
    assert second.cache_hits == ("double",)
    assert second.final_output == Numbers(items=[4])
