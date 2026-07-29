from __future__ import annotations

from pathlib import Path

import pytest
from runweaver import (
    BooleanParameter,
    CategoricalParameter,
    DerivedParameter,
    DesignMatrixPlanner,
    FixedParameter,
    FloatParameter,
    FullFactorialPlanner,
    GridPlanner,
    HaltonPlanner,
    IntegerParameter,
    OrdinalParameter,
    ParameterConstraint,
    ParameterSpace,
    PlanningError,
    PlanningRequest,
    RandomPlanner,
)
from runweaver.domain.models import ExperimentHistory
from runweaver.planning.space import from_unit, resolve_row, to_unit


def mixed_space() -> ParameterSpace:
    return ParameterSpace(parameters=(
        FloatParameter(name="x", low=1e-3, high=1, log=True, quantization=0.001),
        IntegerParameter(name="n", low=1, high=16, log=True),
        CategoricalParameter(name="kind", values=("a", "b")),
        BooleanParameter(name="flag"),
        OrdinalParameter(name="level", values=("low", "mid", "high")),
        FixedParameter(name="fixed", value=3),
        DerivedParameter(name="twice", source_paths=("n",), expression="n * 2"),
    ), constraints=(ParameterConstraint(expression="twice >= 2 and fixed == 3"),))


def test_random_grid_factorial_and_halton() -> None:
    space = mixed_space()
    history = ExperimentHistory()
    random_plans = RandomPlanner(5, seed=1).propose(space, history)
    repeated_plans = RandomPlanner(5, seed=1).propose(space, history)
    assert len(random_plans) == 5
    assert [(plan.id, plan.fingerprint) for plan in random_plans] == [
        (plan.id, plan.fingerprint) for plan in repeated_plans
    ]
    assert all(plan.parameters["twice"] == 2 * plan.parameters["n"] for plan in random_plans)
    tiny = ParameterSpace(parameters=(
        BooleanParameter(name="flag"),
        CategoricalParameter(name="kind", values=("a", "b")),
    ))
    assert len(GridPlanner(2).propose(tiny, history)) == 4
    assert len(FullFactorialPlanner(2).propose(tiny, history)) == 4
    assert len(HaltonPlanner(7, seed=4).propose(tiny, history)) == 7


def test_design_matrix_round_trip_and_validation(tmp_path: Path) -> None:
    rows = [{"x": 1, "kind": "a"}, {"x": 2, "kind": "b"}]
    planner = DesignMatrixPlanner(rows, seed=3)
    csv_path = tmp_path / "design.csv"
    planner.to_csv(csv_path)
    loaded = DesignMatrixPlanner.from_csv(csv_path)
    space = ParameterSpace(parameters=(
        IntegerParameter(name="x", low=0, high=3),
        CategoricalParameter(name="kind", values=("a", "b")),
    ))
    assert len(loaded.propose(space, ExperimentHistory())) == 2
    with pytest.raises(PlanningError, match="fewer rows"):
        loaded.propose(
            space,
            ExperimentHistory(),
            PlanningRequest(n_trials=3),
        )


def test_all_unit_transforms_and_expression_failures() -> None:
    categorical = CategoricalParameter(name="c", values=("a", "b", "c"))
    ordinal = OrdinalParameter(name="o", values=(1, 2, 3))
    boolean = BooleanParameter(name="b")
    fixed = FixedParameter(name="f", value="constant")
    integer = IntegerParameter(name="n", low=1, high=100, log=True)
    assert from_unit(categorical, 0.5) == "b"
    assert to_unit(categorical, "c") == 1
    assert to_unit(ordinal, 2) == 0.5
    assert from_unit(boolean, 0.8) is True
    assert to_unit(boolean, False) == 0
    assert from_unit(fixed, 0.2) == "constant"
    assert to_unit(fixed, "constant") == 0
    assert 1 <= from_unit(integer, 0.4) <= 100

    space = ParameterSpace(
        parameters=(FloatParameter(name="x", low=0, high=1),),
        constraints=(ParameterConstraint(expression="missing > 0", message="missing"),),
    )
    with pytest.raises(PlanningError, match="unknown name"):
        resolve_row(space, {"x": 0.2})
