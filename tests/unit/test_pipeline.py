from __future__ import annotations

import pytest
from pydantic import BaseModel
from runweaver import Pipeline, PipelineValidationError, PortCompatibilityError, function_block
from runweaver.execution import RunContext


class A(BaseModel):
    value: int


class B(BaseModel):
    value: int
    note: str = ""


class C(BaseModel):
    text: str


def identity(inputs: A, context: RunContext) -> A:
    return inputs


def augment(inputs: A, context: RunContext) -> B:
    return B(value=inputs.value)


def incompatible(inputs: C, context: RunContext) -> C:
    return inputs


def test_sequential_graph_validates_and_orders() -> None:
    pipeline = (
        Pipeline("ok")
        .then(function_block(identity, inputs=A, outputs=A))
        .then(function_block(augment, inputs=A, outputs=B))
        .validate()
    )
    assert [node.block.spec.name for node in pipeline.topological_order()] == [
        "identity",
        "augment",
    ]


def test_port_mismatch_fails_before_execution() -> None:
    pipeline = (
        Pipeline("bad")
        .then(function_block(identity, inputs=A, outputs=A))
        .then(function_block(incompatible, inputs=C, outputs=C))
    )
    with pytest.raises(PortCompatibilityError, match="required field"):
        pipeline.validate()


def test_cycle_is_rejected() -> None:
    block = function_block(identity, inputs=A, outputs=A)
    pipeline = Pipeline("cycle")
    pipeline.add(block, id="a", depends_on=("b",))
    pipeline.add(block, id="b", depends_on=("a",))
    with pytest.raises(PipelineValidationError, match="cycle"):
        pipeline.validate()
