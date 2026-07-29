"""Small dependency-free blocks used by quickstarts and declarative smoke tests."""

from __future__ import annotations

from pydantic import BaseModel

from runweaver.domain import BlockRole, function_block
from runweaver.execution import RunContext


class Values(BaseModel):
    values: list[float]


class Summary(BaseModel):
    values: list[float]
    mean: float


def constant_factory(parameters: dict[str, object]) -> object:
    values = [float(value) for value in parameters.get("values", [1.0, 2.0, 3.0])]

    def constant(inputs: Values, context: RunContext) -> Values:
        return Values(values=values)

    return function_block(
        constant,
        inputs=Values,
        outputs=Values,
        name="constant",
        role=BlockRole.GENERATION,
    )


def scale_factory(parameters: dict[str, object]) -> object:
    factor = float(parameters.get("factor", 1.0))

    def scale(inputs: Values, context: RunContext) -> Values:
        return Values(values=[factor * value for value in inputs.values])

    return function_block(
        scale,
        inputs=Values,
        outputs=Values,
        name="scale",
        role=BlockRole.PROCESSING,
    )


def mean_factory(parameters: dict[str, object]) -> object:
    def summarize(inputs: Values, context: RunContext) -> Summary:
        return Summary(
            values=inputs.values,
            mean=sum(inputs.values) / max(1, len(inputs.values)),
        )

    return function_block(
        summarize,
        inputs=Values,
        outputs=Summary,
        name="mean",
        role=BlockRole.EVALUATION,
    )


BUILTIN_BLOCKS = {
    "runweaver.constant": constant_factory,
    "runweaver.scale": scale_factory,
    "runweaver.mean": mean_factory,
}
