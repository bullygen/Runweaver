# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""Tutorial 8: select elites from Tutorial 6 and create a local second round."""

# %%
from __future__ import annotations

from runweaver import (
    CategoricalParameter,
    EliteZoomStrategy,
    FloatParameter,
    IntegerParameter,
    LatinHypercubePlanner,
    ParameterSpace,
    PlanningRequest,
)
from runweaver.domain.models import ExperimentHistory


def design(seed: int = 17, n_trials: int = 20):
    space = ParameterSpace(parameters=(
        FloatParameter(name="temperature", low=250.0, high=450.0),
        FloatParameter(name="rate", low=1e-4, high=1e-1, log=True),
        IntegerParameter(name="passes", low=1, high=8),
        CategoricalParameter(name="catalyst", values=("A", "B", "C")),
    ))
    plans = LatinHypercubePlanner(n_trials, seed=seed).propose(
        space,
        ExperimentHistory(),
        PlanningRequest(n_trials=n_trials, seed=seed, pipeline_version="toy-objective-v1"),
    )
    return space, plans


def objective(parameters: dict[str, object]) -> float:
    return (
        ((float(parameters["temperature"]) - 335.0) / 100.0) ** 2
        + (float(parameters["rate"]) - 0.02) ** 2
        + 0.01 * int(parameters["passes"])
    )


def main() -> None:
    space, plans = design(seed=17, n_trials=20)
    ranked = sorted(plans, key=lambda plan: objective(dict(plan.parameters)))
    child = EliteZoomStrategy(top_k=5, expansion_factor=1.4).refine(
        space,
        ExperimentHistory(),
        ranked[:5],
    )
    second = LatinHypercubePlanner(12, seed=71).propose(
        child,
        ExperimentHistory(),
        PlanningRequest(n_trials=12, seed=71, pipeline_version="toy-objective-v1"),
    )
    old_temperature = space.parameters[0]
    new_temperature = child.parameters[0]
    print("global temperature:", old_temperature.low, old_temperature.high)
    print("local temperature:", new_temperature.low, new_temperature.high)
    print("next-round points:", len(second))
    print("global exploration fraction:", child.metadata["global_exploration_fraction"])


if __name__ == "__main__":
    main()
