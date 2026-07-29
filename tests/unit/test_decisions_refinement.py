from __future__ import annotations

from runweaver import (
    BestFeasiblePolicy,
    CategoricalParameter,
    EliteZoomStrategy,
    ExperimentHistory,
    FloatParameter,
    IntegerParameter,
    MetricDirection,
    MetricRecord,
    NeighborhoodGridStrategy,
    ParameterSpace,
    ParetoFrontPolicy,
    ReplicationStrategy,
    RobustnessStrategy,
    RunState,
    ThresholdPolicy,
    TopKPolicy,
    TrialPlan,
    TrialResult,
    TrustRegionStrategy,
)
from runweaver.artifacts import fingerprint


def trial(index: int, x: float, quality: float, latency: float, feasible: float = 1.0):
    plan = TrialPlan(
        parameters={"x": x, "n": index + 2, "kind": "a" if index < 2 else "b"},
        seed=index,
        pipeline_version="1",
        planner_id="test",
        planner_version="1",
        fingerprint=fingerprint(index),
    )
    return TrialResult(
        trial_plan=plan,
        state=RunState.COMPLETED,
        metrics=(
            MetricRecord(name="quality", value=quality),
            MetricRecord(name="latency", value=latency),
            MetricRecord(name="feasible", value=feasible),
        ),
    )


def history() -> ExperimentHistory:
    return ExperimentHistory(trials=(
        trial(0, 1.0, 0.7, 2.0),
        trial(1, 2.0, 0.9, 4.0),
        trial(2, 7.0, 0.95, 12.0, feasible=0.0),
    ))


def test_decision_policies_cover_objectives_constraints_and_pareto() -> None:
    data = history()
    top = TopKPolicy("quality", k=2).decide(data)
    assert top.selected_trial_ids == (
        data.trials[2].trial_plan.id,
        data.trials[1].trial_plan.id,
    )
    minimum = TopKPolicy("latency", direction=MetricDirection.MINIMIZE).decide(data)
    assert minimum.selected_trial_ids == (data.trials[0].trial_plan.id,)
    feasible = BestFeasiblePolicy(
        "quality",
        constraints={"feasible": ("ge", 1.0)},
    ).decide(data)
    assert feasible.selected_trial_ids == (data.trials[1].trial_plan.id,)
    threshold = ThresholdPolicy("quality", 0.8).decide(data)
    assert len(threshold.selected_trial_ids) == 2
    pareto = ParetoFrontPolicy({
        "quality": MetricDirection.MAXIMIZE,
        "latency": MetricDirection.MINIMIZE,
    }).decide(data)
    assert len(pareto.selected_trial_ids) == 3


def test_all_refinement_strategies_return_versioned_children() -> None:
    data = history()
    selected = [item.trial_plan for item in data.trials[:2]]
    space = ParameterSpace(
        version="global",
        parameters=(
            FloatParameter(name="x", low=0, high=10),
            IntegerParameter(name="n", low=1, high=12),
            CategoricalParameter(name="kind", values=("a", "b", "c")),
        ),
    )
    elite = EliteZoomStrategy(top_k=2).refine(space, data, selected)
    assert elite.parent_version == "global"
    assert elite.parameters[0].high < 10
    assert elite.parameters[2].values == ("a",)
    neighborhood = NeighborhoodGridStrategy(0.2).refine(space, data, selected)
    assert neighborhood.parameters[0].low == 0
    trust = TrustRegionStrategy(improved=False).refine(space, data, selected)
    assert trust.metadata["improved"] is False
    replication = ReplicationStrategy(4).refine(space, data, selected)
    assert replication.metadata["replicates"] == 4
    robustness = RobustnessStrategy(perturbation_fraction=0.02).refine(space, data, selected)
    assert robustness.metadata["refinement"] == "robustness"
