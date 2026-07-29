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
"""Tutorial 12: migrate article_v1/article_v2 image workflows to Runweaver.

This is a small architectural migration example. It retains the former
generation → D1 → ... → D6 → evaluation shape, deterministic designs,
conditional article_v2 parameters and local refinement. It does not claim
numerical equivalence with the removed scientific detector.
"""

# %%
from __future__ import annotations

import argparse
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from runweaver import (
    ActivationCondition,
    ArtifactRef,
    BestFeasiblePolicy,
    BlockRole,
    CategoricalParameter,
    EliteZoomStrategy,
    Experiment,
    FloatParameter,
    LatinHypercubePlanner,
    LocalExecutionConfig,
    LocalExecutor,
    MaterializationMode,
    MetricDirection,
    MetricRecord,
    ParameterSpace,
    Pipeline,
    SobolPlanner,
    Study,
    function_block,
)
from runweaver.execution import RunContext
from scipy import ndimage, optimize


class ArticleState(BaseModel):
    protocol: str
    image_refs: list[ArtifactRef] = []
    truth_radii: list[float] = []
    edge_refs: list[ArtifactRef] = []
    vote_refs: list[ArtifactRef] = []
    candidates: list[list[float]] = []
    initial: list[list[float]] = []
    fits: list[list[float]] = []
    final: list[list[float]] = []
    radius_mae: float | None = None
    detection_rate: float | None = None


def array_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def save_array(context: RunContext, array: np.ndarray, *, role: str, index: int) -> ArtifactRef:
    return context.artifact_store.put_bytes(
        array_bytes(np.asarray(array)),
        media_type="application/x-npy",
        serializer_id="numpy-npy",
        metadata={"semantic_role": role, "index": index},
        lineage={
            "trial_id": str(context.trial_id),
            "block_run_id": context.block_run_id,
            "partition_id": context.partition_id,
        },
    )


def load_array(context: RunContext, ref: ArtifactRef) -> np.ndarray:
    return np.load(io.BytesIO(context.artifact_store.read_bytes(ref)), allow_pickle=False)


# %%
def generate_images(inputs: ArticleState, context: RunContext) -> ArticleState:
    rng = context.child_rng("article-generation")
    yy, xx = np.mgrid[:48, :48]
    refs, truth = [], []
    for index in range(6):
        radius = float(rng.uniform(7.0, 15.0))
        radial = np.hypot(xx - 23.5, yy - 23.5)
        angle = np.arctan2(yy - 23.5, xx - 23.5)
        ring = np.exp(-0.5 * ((radial - radius) / 1.25) ** 2)
        ring *= 1.0 + 0.25 * np.cos(angle - rng.uniform(-np.pi, np.pi))
        background = 0.004 * xx + 0.002 * yy
        noise = rng.normal(0, 0.07 if inputs.protocol == "article_v1" else 0.10, ring.shape)
        image = ring + background + noise
        refs.append(save_array(context, image.astype(np.float32), role="synthetic_image", index=index))
        truth.append(radius)
    return inputs.model_copy(update={"image_refs": refs, "truth_radii": truth})


def d1_edges(inputs: ArticleState, context: RunContext) -> ArticleState:
    method = str(context.parameters.get("edge_method", "sobel"))
    quantile = float(context.parameters.get("edge_quantile", 0.84))
    smoothing = float(context.parameters.get("smoothing", 1.0))
    refs = []
    for index, image_ref in enumerate(inputs.image_refs):
        image = load_array(context, image_ref)
        if method == "log":
            score = np.abs(ndimage.gaussian_laplace(image, sigma=smoothing))
        else:
            smooth = ndimage.gaussian_filter(image, sigma=smoothing)
            score = np.hypot(ndimage.sobel(smooth, axis=0), ndimage.sobel(smooth, axis=1))
        threshold = float(np.quantile(score, quantile))
        edges = (score >= threshold).astype(np.float32)
        refs.append(save_array(context, edges, role="d1_edges", index=index))
    return inputs.model_copy(update={"edge_refs": refs})


def d2_radial_votes(inputs: ArticleState, context: RunContext) -> ArticleState:
    yy, xx = np.mgrid[:48, :48]
    radius_bins = np.rint(np.hypot(xx - 23.5, yy - 23.5)).astype(int)
    refs = []
    for index, edge_ref in enumerate(inputs.edge_refs):
        edges = load_array(context, edge_ref)
        votes = np.bincount(radius_bins.ravel(), weights=edges.ravel(), minlength=34)
        counts = np.bincount(radius_bins.ravel(), minlength=34)
        profile = votes / np.maximum(counts, 1)
        refs.append(save_array(context, profile.astype(np.float32), role="d2_votes", index=index))
    return inputs.model_copy(update={"vote_refs": refs})


def d3_candidates(inputs: ArticleState, context: RunContext) -> ArticleState:
    candidates = []
    for ref in inputs.vote_refs:
        profile = load_array(context, ref)
        search = profile[4:21]
        best = int(np.argmax(search)) + 4
        if inputs.protocol == "article_v2":
            masked = search.copy()
            masked[max(0, best - 4): min(len(masked), best)] = -np.inf
            second = int(np.argmax(masked)) + 4
            candidates.append([float(best), float(second)])
        else:
            candidates.append([float(best)])
    return inputs.model_copy(update={"candidates": candidates})


def d4_initialize(inputs: ArticleState, context: RunContext) -> ArticleState:
    initialized = [[float(np.clip(radius + 0.2, 4, 21)) for radius in row] for row in inputs.candidates]
    return inputs.model_copy(update={"initial": initialized})


def d5_fit(inputs: ArticleState, context: RunContext) -> ArticleState:
    yy, xx = np.mgrid[:48, :48]
    radial = np.hypot(xx - 23.5, yy - 23.5)
    fitted = []
    for image_ref, initialized in zip(inputs.image_refs, inputs.initial, strict=True):
        image = load_array(context, image_ref)
        row = []
        for estimate in initialized:
            def objective(radius: float, image_: np.ndarray = image) -> float:
                weight = np.exp(-0.5 * ((radial - radius) / 1.1) ** 2)
                return -float(np.sum(weight * image_) / np.sum(weight))

            result = optimize.minimize_scalar(
                objective,
                bounds=(max(4.0, estimate - 2.5), min(21.0, estimate + 2.5)),
                method="bounded",
                options={"xatol": 0.02},
            )
            row.append(float(result.x))
        fitted.append(row)
    return inputs.model_copy(update={"fits": fitted})


def d6_prune(inputs: ArticleState, context: RunContext) -> ArticleState:
    limit = 1 if inputs.protocol == "article_v1" else 2
    final = [
        sorted({round(radius, 4) for radius in row if np.isfinite(radius)})[:limit]
        for row in inputs.fits
    ]
    return inputs.model_copy(update={"final": final})


def evaluate(inputs: ArticleState, context: RunContext) -> ArticleState:
    errors = [
        min(abs(candidate - truth) for candidate in candidates)
        for truth, candidates in zip(inputs.truth_radii, inputs.final, strict=True)
        if candidates
    ]
    radius_mae = float(np.mean(errors)) if errors else float("inf")
    detection_rate = float(np.mean([error <= 2.5 for error in errors])) if errors else 0.0
    context.report_metric(MetricRecord(
        name="radius_mae",
        value=radius_mae,
        direction=MetricDirection.MINIMIZE,
        split="synthetic",
        uncertainty={"std": float(np.std(errors)) if errors else 0.0},
        unit="pixels",
    ))
    context.report_metric(MetricRecord(
        name="detection_rate",
        value=detection_rate,
        direction=MetricDirection.MAXIMIZE,
        split="synthetic",
    ))
    return inputs.model_copy(update={"radius_mae": radius_mae, "detection_rate": detection_rate})


def build_pipeline() -> Pipeline:
    return (
        Pipeline("article-image-workflow", version="1")
        .then(function_block(generate_images, inputs=ArticleState, outputs=ArticleState, role=BlockRole.GENERATION))
        .then(function_block(d1_edges, inputs=ArticleState, outputs=ArticleState, role=BlockRole.PREPROCESSING))
        .then(function_block(d2_radial_votes, inputs=ArticleState, outputs=ArticleState, role=BlockRole.PROCESSING))
        .then(function_block(d3_candidates, inputs=ArticleState, outputs=ArticleState, role=BlockRole.PROCESSING))
        .then(function_block(d4_initialize, inputs=ArticleState, outputs=ArticleState, role=BlockRole.PROCESSING))
        .then(function_block(d5_fit, inputs=ArticleState, outputs=ArticleState, role=BlockRole.PROCESSING))
        .then(function_block(d6_prune, inputs=ArticleState, outputs=ArticleState, role=BlockRole.PROCESSING))
        .then(function_block(evaluate, inputs=ArticleState, outputs=ArticleState, role=BlockRole.EVALUATION))
    )


def protocol_space(protocol: str) -> ParameterSpace:
    if protocol == "article_v1":
        return ParameterSpace(version="article-v1", parameters=(
            FloatParameter(name="edge_quantile", low=0.76, high=0.91),
            FloatParameter(name="smoothing", low=0.6, high=1.5),
        ))
    return ParameterSpace(version="article-v2", parameters=(
        CategoricalParameter(name="edge_method", values=("sobel", "log")),
        FloatParameter(
            name="edge_quantile",
            low=0.74,
            high=0.93,
            activation=ActivationCondition(parameter="edge_method", value="sobel"),
        ),
        FloatParameter(name="smoothing", low=0.5, high=1.8),
    ))


def run_protocol(protocol: str, root: Path, n_trials: int) -> dict[str, object]:
    space = protocol_space(protocol)
    planner = (
        LatinHypercubePlanner(n_trials, seed=101)
        if protocol == "article_v1"
        else SobolPlanner(n_trials, seed=202)
    )
    executor = LocalExecutor(LocalExecutionConfig(
        materialization=MaterializationMode.DURABLE,
        work_dir=root / protocol / "work",
        artifact_root=str(root / protocol / "artifacts"),
        state_database_url=f"sqlite:///{root / protocol / 'state.db'}",
        install_signal_handlers=False,
    ))
    study = Study(
        experiment=Experiment(name=protocol, seed=239),
        pipeline=build_pipeline(),
        planner=planner,
        parameter_space=space,
        initial_input=lambda plan: ArticleState(protocol=protocol),
        executor=executor,
        decision_policy=BestFeasiblePolicy(
            objective="radius_mae",
            direction=MetricDirection.MINIMIZE,
            constraints={"detection_rate": ("ge", 0.5)},
        ),
        refinement=EliteZoomStrategy(top_fraction=0.4),
    )
    result = study.run()
    completed = [trial for trial in result.history.trials if trial.state.value == "completed"]
    summary = {
        "protocol": protocol,
        "trial_count": len(result.history.trials),
        "completed": len(completed),
        "selected_trial_ids": (
            [str(item) for item in result.decision.selected_trial_ids] if result.decision else []
        ),
        "refined_space_version": result.refined_space.version if result.refined_space else None,
        "metrics": [
            {
                "trial_id": str(trial.trial_plan.id),
                "parameters": dict(trial.trial_plan.parameters),
                "radius_mae": next(
                    (metric.value for metric in trial.metrics if metric.name == "radius_mae"),
                    None,
                ),
                "detection_rate": next(
                    (metric.value for metric in trial.metrics if metric.name == "detection_rate"),
                    None,
                ),
            }
            for trial in completed
        ],
    }
    output = root / protocol / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# %%
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["article_v1", "article_v2", "both"], default="both")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--out")
    args, _ = parser.parse_known_args()
    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="runweaver-article-"))
    protocols = ["article_v1", "article_v2"] if args.protocol == "both" else [args.protocol]
    for protocol in protocols:
        summary = run_protocol(protocol, root, args.trials)
        print(
            protocol,
            f"completed={summary['completed']}/{summary['trial_count']}",
            f"selected={len(summary['selected_trial_ids'])}",
            f"refined={summary['refined_space_version']}",
        )
    print(f"artifacts and summaries: {root}")


if __name__ == "__main__":
    main()
