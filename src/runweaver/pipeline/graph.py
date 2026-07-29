"""Typed DAG builder and preflight validation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from runweaver.domain.blocks import Block
from runweaver.domain.models import ExecutionPattern, ResourceRequirements
from runweaver.exceptions import PipelineValidationError, PortCompatibilityError


@dataclass(frozen=True)
class PipelineNode:
    id: str
    block: Block[Any, Any]
    dependencies: tuple[str, ...]
    map_over: str | None = None
    parallelism: int = 1
    resources: ResourceRequirements | None = None


@dataclass(frozen=True)
class PipelineEdge:
    source: str
    target: str


class Pipeline:
    """A convenient sequential DSL backed by a general directed acyclic graph."""

    def __init__(self, name: str, *, version: str = "1") -> None:
        self.name = name
        self.version = version
        self._nodes: dict[str, PipelineNode] = {}
        self._last: tuple[str, ...] = ()

    @property
    def nodes(self) -> tuple[PipelineNode, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[PipelineEdge, ...]:
        return tuple(
            PipelineEdge(source=dependency, target=node.id)
            for node in self._nodes.values()
            for dependency in node.dependencies
        )

    def then(
        self,
        block: Block[Any, Any],
        *,
        id: str | None = None,
        map_over: str | None = None,
        parallelism: int = 1,
        resources: ResourceRequirements | None = None,
    ) -> Pipeline:
        node_id = id or block.spec.name
        if node_id in self._nodes:
            suffix = 2
            while f"{node_id}_{suffix}" in self._nodes:
                suffix += 1
            node_id = f"{node_id}_{suffix}"
        self.add(
            block,
            id=node_id,
            depends_on=self._last,
            map_over=map_over,
            parallelism=parallelism,
            resources=resources,
        )
        self._last = (node_id,)
        return self

    def add(
        self,
        block: Block[Any, Any],
        *,
        id: str,
        depends_on: tuple[str, ...] = (),
        map_over: str | None = None,
        parallelism: int = 1,
        resources: ResourceRequirements | None = None,
    ) -> Pipeline:
        if id in self._nodes:
            raise PipelineValidationError(f"duplicate block id: {id}")
        if parallelism < 1:
            raise PipelineValidationError(f"parallelism must be >= 1 for {id}")
        self._nodes[id] = PipelineNode(
            id=id,
            block=block,
            dependencies=tuple(depends_on),
            map_over=map_over,
            parallelism=parallelism,
            resources=resources,
        )
        self._last = (id,)
        return self

    def join(self, *node_ids: str) -> Pipeline:
        """Select several existing nodes as dependencies of the next ``then`` call."""

        missing = set(node_ids) - self._nodes.keys()
        if missing:
            raise PipelineValidationError(f"unknown join nodes: {sorted(missing)}")
        self._last = tuple(node_ids)
        return self

    def topological_order(self) -> tuple[PipelineNode, ...]:
        indegree = {node_id: 0 for node_id in self._nodes}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for node in self._nodes.values():
            for dependency in node.dependencies:
                if dependency not in self._nodes:
                    raise PipelineValidationError(
                        f"block {node.id!r} depends on unknown block {dependency!r}"
                    )
                indegree[node.id] += 1
                outgoing[dependency].append(node.id)
        ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        ordered: list[PipelineNode] = []
        while ready:
            node_id = ready.popleft()
            ordered.append(self._nodes[node_id])
            for child in outgoing[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self._nodes):
            cyclic = sorted(node_id for node_id, degree in indegree.items() if degree)
            raise PipelineValidationError(f"pipeline contains a cycle: {cyclic}")
        return tuple(ordered)

    def validate(self) -> Pipeline:
        ordered = self.topological_order()
        if not ordered:
            raise PipelineValidationError("pipeline must contain at least one block")
        for node in ordered:
            if node.map_over is not None:
                field = node.block.input_type.model_fields.get(node.map_over)
                if field is None:
                    raise PipelineValidationError(
                        f"map field {node.map_over!r} does not exist on {node.id} input"
                    )
                origin = get_origin(field.annotation)
                if origin not in (list, tuple):
                    raise PipelineValidationError(
                        f"map field {node.map_over!r} on {node.id} must be a list or tuple"
                    )
            if len(node.dependencies) == 1:
                upstream = self._nodes[node.dependencies[0]]
                _validate_model_compatibility(upstream.block.output_type, node.block.input_type)
            elif len(node.dependencies) > 1:
                available: set[str] = set()
                for dependency in node.dependencies:
                    available.update(self._nodes[dependency].block.output_type.model_fields)
                required = {
                    name
                    for name, field in node.block.input_type.model_fields.items()
                    if field.is_required()
                }
                missing = required - available
                if missing:
                    raise PortCompatibilityError(
                        f"fan-in block {node.id!r} lacks required fields: {sorted(missing)}"
                    )
            if node.block.spec.execution_pattern == ExecutionPattern.STREAM:
                raise PipelineValidationError(
                    f"stream block {node.id!r} is unsupported until backpressure semantics are configured"
                )
        return self

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "nodes": [
                {
                    "id": node.id,
                    "block": node.block.spec.model_dump(mode="json"),
                    "depends_on": list(node.dependencies),
                    "map_over": node.map_over,
                    "parallelism": node.parallelism,
                    "resources": (node.resources or node.block.spec.resources).model_dump(mode="json"),
                }
                for node in self.topological_order()
            ],
        }


def _validate_model_compatibility(output_type: type[BaseModel], input_type: type[BaseModel]) -> None:
    output_fields = output_type.model_fields
    for name, input_field in input_type.model_fields.items():
        if not input_field.is_required() and name not in output_fields:
            continue
        output_field = output_fields.get(name)
        if output_field is None:
            raise PortCompatibilityError(
                f"{output_type.__name__} does not provide required field "
                f"{input_type.__name__}.{name}"
            )
        if not _annotation_compatible(output_field.annotation, input_field.annotation):
            raise PortCompatibilityError(
                f"incompatible field {name}: {output_field.annotation!r} -> "
                f"{input_field.annotation!r}"
            )


def _annotation_compatible(output: object, input_: object) -> bool:
    if output == input_ or input_ is Any:
        return True
    output_origin = get_origin(output)
    input_origin = get_origin(input_)
    if output_origin == input_origin and output_origin is not None:
        output_args = get_args(output)
        input_args = get_args(input_)
        return len(output_args) == len(input_args) and all(
            _annotation_compatible(out, inp)
            for out, inp in zip(output_args, input_args, strict=True)
        )
    try:
        return isinstance(output, type) and isinstance(input_, type) and issubclass(output, input_)
    except TypeError:
        return False
