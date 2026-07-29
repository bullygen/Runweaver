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
"""Tutorial 7: persistent Optuna TPE plans, intermediate pruning and restart."""

# %%
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

from runweaver import FloatParameter, IntegerParameter, ParameterSpace
from runweaver.domain.models import ExperimentHistory
from runweaver.integrations.optuna import OptunaPlanner


def main() -> None:
    if importlib.util.find_spec("optuna") is None:
        print("SKIP: install `runweaver[optuna]` to execute adaptive HPO")
        return
    root = Path(tempfile.mkdtemp(prefix="runweaver-t07-"))
    storage = f"sqlite:///{root / 'optuna.db'}"
    space = ParameterSpace(parameters=(
        FloatParameter(name="x", low=-5, high=5),
        IntegerParameter(name="degree", low=1, high=4),
    ))
    planner = OptunaPlanner(
        5,
        study_name="toy-persistent",
        storage=storage,
        sampler="tpe",
        seed=23,
    )
    plans = planner.propose(space, ExperimentHistory())
    for plan in plans:
        x = float(plan.parameters["x"])
        degree = int(plan.parameters["degree"])
        objective = (x - 1.25) ** 2 + 0.05 * degree
        planner.report_intermediate(plan, objective + 1.0, step=0)
        planner.tell(plan, objective)
    restarted = OptunaPlanner(
        2,
        study_name="toy-persistent",
        storage=storage,
        sampler="tpe",
        seed=23,
    )
    next_plans = restarted.propose(space, ExperimentHistory())
    print("completed:", len(plans), "new after restart:", len(next_plans))
    print("persistent storage:", storage)


if __name__ == "__main__":
    main()
