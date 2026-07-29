from __future__ import annotations

import pytest
from runweaver import (
    ActivationCondition,
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    LatinHypercubePlanner,
    ParameterConstraint,
    ParameterSpace,
    PlanningRequest,
    SobolPlanner,
)
from runweaver.domain.models import ExperimentHistory
from runweaver.planning.space import from_unit, to_unit


@pytest.mark.parametrize("unit", [0.0, 0.1, 0.5, 0.999999])
def test_float_normalization_round_trip(unit: float) -> None:
    parameter = FloatParameter(name="lr", low=1e-4, high=1e-1, log=True)
    physical = from_unit(parameter, unit)
    assert to_unit(parameter, physical) == pytest.approx(unit)


def test_lhs_plans_are_immutable_deterministic_and_conditional() -> None:
    space = ParameterSpace(
        parameters=(
            CategoricalParameter(name="method", values=("a", "b")),
            FloatParameter(
                name="strength",
                low=0.0,
                high=1.0,
                activation=ActivationCondition(parameter="method", value="b"),
            ),
            IntegerParameter(name="count", low=1, high=5),
        ),
        constraints=(ParameterConstraint(expression="count >= 1"),),
    )
    planner = LatinHypercubePlanner(12, seed=42)
    first = planner.propose(space, ExperimentHistory())
    second = planner.propose(space, ExperimentHistory())
    assert [plan.fingerprint for plan in first] == [plan.fingerprint for plan in second]
    assert len({plan.fingerprint for plan in first}) == 12
    assert all(
        ("strength" in plan.parameters) == (plan.parameters["method"] == "b")
        for plan in first
    )


def test_sobol_stays_inside_physical_bounds() -> None:
    space = ParameterSpace(
        parameters=(
            FloatParameter(name="x", low=-5, high=2),
            IntegerParameter(name="n", low=3, high=9),
        )
    )
    plans = SobolPlanner(16, seed=7).propose(
        space,
        ExperimentHistory(),
        PlanningRequest(n_trials=16, seed=7),
    )
    assert all(-5 <= float(plan.parameters["x"]) <= 2 for plan in plans)
    assert all(3 <= int(plan.parameters["n"]) <= 9 for plan in plans)
