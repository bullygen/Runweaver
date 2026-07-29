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
"""Tutorial 6: Latin-hypercube design in normalized and physical domains."""

# %%
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from runweaver import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    LatinHypercubePlanner,
    ParameterSpace,
    PlanningRequest,
)
from runweaver.domain.models import ExperimentHistory
from runweaver.planning.space import to_unit


def design(seed: int = 17, n_trials: int = 12):
    space = ParameterSpace(
        version="doe-v1",
        parameters=(
            FloatParameter(name="temperature", low=250.0, high=450.0, unit="K"),
            FloatParameter(name="rate", low=1e-4, high=1e-1, log=True),
            IntegerParameter(name="passes", low=1, high=8),
            CategoricalParameter(name="catalyst", values=("A", "B", "C")),
        ),
    )
    plans = LatinHypercubePlanner(n_trials, seed=seed).propose(
        space,
        ExperimentHistory(),
        PlanningRequest(n_trials=n_trials, seed=seed, pipeline_version="toy-objective-v1"),
    )
    return space, plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args, _ = parser.parse_known_args()
    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="runweaver-t06-"))
    root.mkdir(parents=True, exist_ok=True)
    space, plans = design()
    output = root / "design.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plans[0].parameters))
        writer.writeheader()
        writer.writerows(plan.parameters for plan in plans)
    temperature = space.parameters[0]
    normalized = [to_unit(temperature, plan.parameters["temperature"]) for plan in plans]
    print("trial count:", len(plans))
    print("normalized temperature range:", min(normalized), max(normalized))
    print("design matrix:", output)


if __name__ == "__main__":
    main()
