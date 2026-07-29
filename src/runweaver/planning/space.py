"""Parameter transforms, conditions and safe constraint evaluation."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from runweaver.domain.models import (
    BooleanParameter,
    CategoricalParameter,
    DerivedParameter,
    FixedParameter,
    FloatParameter,
    IntegerParameter,
    OrdinalParameter,
    Parameter,
    ParameterSpace,
)
from runweaver.exceptions import PlanningError


def from_unit(parameter: Parameter, unit: float) -> object:
    """Map ``[0, 1]`` into a parameter's physical domain."""

    value = float(np.clip(unit, 0.0, np.nextafter(1.0, 0.0)))
    if isinstance(parameter, FloatParameter):
        if parameter.log:
            physical = math.exp(math.log(parameter.low) + value * (math.log(parameter.high) - math.log(parameter.low)))
        else:
            physical = parameter.low + value * (parameter.high - parameter.low)
        if parameter.quantization:
            step = float(parameter.quantization)
            physical = round(physical / step) * step
        return float(np.clip(physical, parameter.low, parameter.high))
    if isinstance(parameter, IntegerParameter):
        if parameter.log:
            physical = math.exp(math.log(parameter.low) + value * (math.log(parameter.high) - math.log(parameter.low)))
            result = round(physical)
        else:
            result = parameter.low + math.floor(value * (parameter.high - parameter.low + 1))
        step = int(parameter.quantization or 1)
        result = parameter.low + round((result - parameter.low) / step) * step
        return int(np.clip(result, parameter.low, parameter.high))
    if isinstance(parameter, CategoricalParameter | OrdinalParameter):
        return parameter.values[min(len(parameter.values) - 1, int(value * len(parameter.values)))]
    if isinstance(parameter, BooleanParameter):
        return bool(value >= 0.5)
    if isinstance(parameter, FixedParameter):
        return parameter.value
    if isinstance(parameter, DerivedParameter):
        raise PlanningError(f"derived parameter {parameter.name!r} cannot be sampled directly")
    raise PlanningError(f"unsupported parameter type: {type(parameter).__name__}")


def to_unit(parameter: Parameter, physical: object) -> float:
    """Map a physical value back to a normalized unit interval."""

    if isinstance(parameter, FloatParameter):
        if not isinstance(physical, (int, float)):
            raise PlanningError(f"{parameter.name!r} requires a numeric value")
        value = float(physical)
        if parameter.log:
            return (math.log(value) - math.log(parameter.low)) / (math.log(parameter.high) - math.log(parameter.low))
        return (value - parameter.low) / (parameter.high - parameter.low)
    if isinstance(parameter, IntegerParameter):
        if not isinstance(physical, (int, float)):
            raise PlanningError(f"{parameter.name!r} requires a numeric value")
        value = int(physical)
        if parameter.log:
            return (math.log(value) - math.log(parameter.low)) / (math.log(parameter.high) - math.log(parameter.low))
        return float((value - parameter.low) / max(1, parameter.high - parameter.low))
    if isinstance(parameter, CategoricalParameter | OrdinalParameter):
        return parameter.values.index(physical) / max(1, len(parameter.values) - 1)
    if isinstance(parameter, BooleanParameter):
        return 1.0 if physical else 0.0
    if isinstance(parameter, FixedParameter):
        return 0.0
    raise PlanningError(f"parameter {parameter.name!r} has no inverse unit transform")


def resolve_row(space: ParameterSpace, units: Mapping[str, float]) -> dict[str, object]:
    values: dict[str, object] = {}
    for parameter in space.parameters:
        if parameter.activation and not parameter.activation.matches(values):
            continue
        if isinstance(parameter, DerivedParameter):
            names = {path: values[path] for path in parameter.source_paths}
            values[parameter.resolved_path] = _safe_expression(parameter.expression, names)
        elif isinstance(parameter, FixedParameter):
            values[parameter.resolved_path] = parameter.value
        else:
            values[parameter.resolved_path] = from_unit(parameter, units[parameter.name])
    validate_constraints(space, values)
    return values


def validate_constraints(space: ParameterSpace, values: Mapping[str, object]) -> None:
    for constraint in space.constraints:
        if not bool(_safe_expression(constraint.expression, values)):
            raise PlanningError(constraint.message)


_ALLOWED_BINARY: dict[type[ast.operator], Callable[[Any, Any], object]] = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
    ast.Mod: lambda left, right: left % right,
}
_ALLOWED_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], object]] = {
    ast.Eq: lambda left, right: left == right,
    ast.NotEq: lambda left, right: left != right,
    ast.Lt: lambda left, right: left < right,
    ast.LtE: lambda left, right: left <= right,
    ast.Gt: lambda left, right: left > right,
    ast.GtE: lambda left, right: left >= right,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


def _safe_expression(expression: str, names: Mapping[str, object]) -> object:
    """Evaluate a small data-only expression language, never Python calls."""

    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> object:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise PlanningError(f"unknown name in expression: {node.id}")
            return names[node.id]
        if isinstance(node, ast.List):
            return [visit(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(visit(item) for item in node.elts)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY:
            return _ALLOWED_BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            value = visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -value  # type: ignore[operator]
            if isinstance(node.op, ast.UAdd):
                return +value  # type: ignore[operator]
            return not value
        if isinstance(node, ast.BoolOp):
            values = [bool(visit(item)) for item in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for operator, comparator in zip(node.ops, node.comparators, strict=True):
                right = visit(comparator)
                if type(operator) not in _ALLOWED_COMPARE or not _ALLOWED_COMPARE[type(operator)](left, right):
                    return False
                left = right
            return True
        raise PlanningError(f"unsupported expression syntax: {ast.dump(node, include_attributes=False)}")

    return visit(tree)
