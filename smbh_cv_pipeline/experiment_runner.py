from __future__ import annotations

"""Resumable multi-start hyperparameter experiments around the fixed D1--D6 methods."""

import argparse
import csv
import json
import logging
import math
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .cli import STAGE_FUNCS, _pool_map, setup_logging
from .config import PipelineConfig
from .io_utils import dump_json, read_json, run_paths
from .statistics import run_statistics


def _fingerprint(data: Any) -> str:
    import hashlib
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sample_dimension(rng: np.random.Generator, definition: Mapping[str, Any], unit: float) -> Any:
    kind = definition["type"]
    if kind == "float":
        return float(definition["low"] + unit * (definition["high"] - definition["low"]))
    if kind == "log_float":
        lo, hi = math.log(float(definition["low"])), math.log(float(definition["high"]))
        return float(math.exp(lo + unit * (hi - lo)))
    if kind == "int":
        lo, hi = int(definition["low"]), int(definition["high"])
        return int(min(hi, lo + math.floor(unit * (hi - lo + 1))))
    if kind == "choice":
        values = definition["values"]
        return values[min(len(values) - 1, int(unit * len(values)))]
    raise ValueError(f"Unknown search dimension type: {kind}")


def create_candidates(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create one baseline plus Latin-hypercube samples for each independent start."""
    candidates: List[Dict[str, Any]] = [{"candidate_id": "baseline", "start": -1, "trial": -1, "params": {}}]
    names = list(spec["search_space"])
    starts, per_start = int(spec["starts"]), int(spec["trials_per_start"])
    root_seed = int(spec["search_seed"])
    for start in range(starts):
        rng = np.random.default_rng(root_seed + 1009 * start)
        units: Dict[str, np.ndarray] = {}
        for name in names:
            values = (np.arange(per_start) + rng.random(per_start)) / per_start
            rng.shuffle(values)
            units[name] = values
        for trial in range(per_start):
            params = {name: _sample_dimension(rng, spec["search_space"][name], float(units[name][trial])) for name in names}
            for target, source in spec.get("derived_equal", {}).items():
                params[target] = params[source]
            if "d3_max_candidates" in params:
                params["d4_max_init_candidates"] = int(params["d3_max_candidates"])
            candidate_id = f"s{start + 1:02d}_t{trial + 1:02d}_{_fingerprint(params)[:8]}"
            candidates.append({"candidate_id": candidate_id, "start": start, "trial": trial, "params": params})
    return candidates


def _load_config(path: str | Path) -> PipelineConfig:
    data = read_json(Path(path))
    allowed = {field.name for field in fields(PipelineConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown PipelineConfig keys in {path}: {sorted(unknown)}")
    return PipelineConfig(**data)


def _select_dataset_items(dataset_split: Path, phase: Mapping[str, Any]) -> List[Dict[str, Any]]:
    items = list(read_json(dataset_split / "manifest.json")["items"])
    folds = phase.get("folds")
    if folds is not None:
        allowed = {int(x) for x in folds}
        items = [row for row in items if int(row.get("fold", -1)) in allowed]
    items.sort(key=lambda row: row["image_id"])
    max_images = phase.get("max_images")
    if max_images is not None:
        # The design already randomizes factors over image IDs; fixed selection keeps
        # every candidate on exactly the same images (paired comparison).
        items = items[: int(max_images)]
    if not items:
        raise ValueError(f"Dataset selection is empty for {dataset_split}")
    return items


def _prepare_run(
    dataset_split: Path,
    run_dir: Path,
    items: Sequence[Mapping[str, Any]],
    cfg: PipelineConfig,
    restart: bool,
    signature: Mapping[str, Any],
) -> None:
    if restart and run_dir.exists():
        shutil.rmtree(run_dir)
    signature_path = run_dir / "run_signature.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not signature_path.exists():
            raise RuntimeError(f"Existing run has no signature; use --restart after checking the target: {run_dir}")
        existing = read_json(signature_path)
        if existing.get("fingerprint") != signature.get("fingerprint"):
            raise RuntimeError(f"Run inputs/config changed; refusing to mix artifacts. Use --restart: {run_dir}")
    paths = run_paths(run_dir)
    for item in items:
        source = (dataset_split / "images" / item["image_id"]).resolve()
        target = paths["images"] / item["image_id"]
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source, target_is_directory=True)
    manifest_items = [{
        "image_id": item["image_id"], "status": "ok", "true_n": item.get("true_n"),
        "family_id": item.get("family_id"), "fold": item.get("fold"), "factors": item.get("factors", {}),
    } for item in items]
    dump_json(paths["manifest"], {"source_dataset": str(dataset_split), "items": manifest_items})
    dump_json(paths["config"], cfg.to_dict())
    dump_json(signature_path, signature)


_STAGE_OUTPUT = {
    "d1": ("d1", "d1_edges.npz"),
    "d2": ("d2", "d2_sparse_accumulator.npz"),
    "d3": ("d3", "d3_candidates.json"),
    "d4": ("d4", "d4_initial_params.json"),
    "d5": ("d5", "d5_fit.json"),
    "d6": ("d6", "d6_final.json"),
}


def _missing_stage_image_ids(cfg: PipelineConfig, stage: str, image_ids: Sequence[str]) -> List[str]:
    paths = run_paths(cfg.out)
    key, filename = _STAGE_OUTPUT[stage]
    return [
        image_id
        for image_id in image_ids
        if not (paths[key] / image_id / filename).is_file()
    ]


def _run_one(
    spec: Mapping[str, Any], phase_name: str, candidate: Mapping[str, Any], algorithm_seed: int,
    restart: bool = False,
) -> Dict[str, Any]:
    phase = spec["phases"][phase_name]
    dataset_split = Path(spec["dataset_root"]) / phase["split"]
    items = _select_dataset_items(dataset_split, phase)
    candidate_id = str(candidate["candidate_id"])
    run_dir = Path(spec["output_root"]) / phase_name / candidate_id / f"algorithm_seed_{algorithm_seed}"
    cfg = _load_config(spec["base_config"])
    for key, value in candidate.get("params", {}).items():
        if not hasattr(cfg, key):
            raise ValueError(f"Candidate contains unknown config key: {key}")
        setattr(cfg, key, value)
    for key, value in phase.get("config_overrides", {}).items():
        if not hasattr(cfg, key):
            raise ValueError(f"Phase override contains unknown config key: {key}")
        setattr(cfg, key, value)
    cfg.out = str(run_dir)
    cfg.seed = int(algorithm_seed)
    cfg.n_images = len(items)
    cfg.make_plots = False
    cfg.save_arrays = bool(phase.get("save_arrays", False))
    dataset_manifest = read_json(dataset_split / "manifest.json")
    signature_payload = {
        "phase": phase_name,
        "candidate_id": candidate_id,
        "candidate_params": candidate.get("params", {}),
        "algorithm_seed": int(algorithm_seed),
        "dataset_plan_hash": dataset_manifest.get("plan_hash"),
        "selected_images": [[item["image_id"], item.get("image_sha256")] for item in items],
        "pipeline_config": cfg.to_dict(),
        "phase_config": phase,
    }
    signature = {"fingerprint": _fingerprint(signature_payload), "payload": signature_payload}
    _prepare_run(dataset_split, run_dir, items, cfg, restart=restart, signature=signature)
    image_ids = [str(item["image_id"]) for item in items]
    workers = spec.get("workers", {})
    for stage in ("d1", "d2", "d3", "d4", "d5", "d6"):
        missing_image_ids = _missing_stage_image_ids(cfg, stage, image_ids)
        if not missing_image_ids:
            continue
        logging.info(
            "Resuming %s candidate=%s seed=%d: %d/%d images pending",
            stage.upper(), candidate_id, algorithm_seed, len(missing_image_ids), len(image_ids),
        )
        _pool_map(
            stage,
            STAGE_FUNCS[stage],
            cfg,
            missing_image_ids,
            int(workers.get(stage, 1)),
            total_items=len(image_ids),
        )
    summary_path = run_dir / "statistics" / "summary.json"
    if not summary_path.exists():
        run_statistics(cfg)
    summary = read_json(summary_path)
    dump_json(run_dir / "experiment_metadata.json", {
        "phase": phase_name,
        "candidate": candidate,
        "algorithm_seed": algorithm_seed,
        "dataset_split": str(dataset_split),
        "n_images": len(items),
        "config_fingerprint": _fingerprint(cfg.to_dict()),
    })
    return summary


def _finite(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        try:
            value = float(value)
            if math.isfinite(value):
                out.append(value)
        except (TypeError, ValueError):
            pass
    return out


def _objective(run_summaries: Sequence[Mapping[str, Any]], objective: Mapping[str, Any]) -> Dict[str, float | bool]:
    f1 = _finite(row.get("article_f1") for row in run_summaries)
    f1_low = _finite((row.get("article_f1_ci95") or [None])[0] for row in run_summaries)
    precision = _finite(row.get("article_precision") for row in run_summaries)
    recall = _finite(row.get("article_recall") for row in run_summaries)
    null_fppi = _finite(row.get("null_fppi") for row in run_summaries)
    if not f1 or not f1_low or not precision or not recall:
        return {"score": float("-inf"), "feasible": False}
    mean_f1 = float(np.mean(f1))
    std_f1 = float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0
    worst_ci_low = float(min(f1_low))
    max_null_fppi = float(max(null_fppi)) if null_fppi else float("inf")
    feasible = min(precision) >= float(objective["precision_floor"]) and min(recall) >= float(objective["recall_floor"])
    score = worst_ci_low - float(objective["seed_std_penalty"]) * std_f1 - float(objective["null_fppi_penalty"]) * max_null_fppi
    if not feasible:
        score -= float(objective["infeasible_penalty"])
    return {
        "score": float(score), "feasible": bool(feasible), "mean_f1": mean_f1,
        "std_f1_across_algorithm_seeds": std_f1, "worst_f1_ci95_low": worst_ci_low,
        "min_precision": float(min(precision)), "min_recall": float(min(recall)),
        "max_null_fppi": max_null_fppi,
    }


def summarize_phase(spec: Mapping[str, Any], phase_name: str, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    root = Path(spec["output_root"]) / phase_name
    ranking = []
    for candidate in candidates:
        candidate_dir = root / candidate["candidate_id"]
        summaries = []
        for path in sorted(candidate_dir.glob("algorithm_seed_*/statistics/summary.json")):
            summaries.append(read_json(path))
        if not summaries:
            continue
        ranking.append({
            "candidate_id": candidate["candidate_id"], "params": candidate.get("params", {}),
            "n_runs": len(summaries), **_objective(summaries, spec["objective"]),
        })
    ranking.sort(key=lambda row: float(row["score"]), reverse=True)
    result = {"phase": phase_name, "objective": spec["objective"], "ranking": ranking}
    root.mkdir(parents=True, exist_ok=True)
    dump_json(root / "ranking.json", result)
    if ranking:
        fields_out = ["rank", "candidate_id", "score", "feasible", "n_runs", "mean_f1", "std_f1_across_algorithm_seeds", "worst_f1_ci95_low", "min_precision", "min_recall", "max_null_fppi", "params"]
        with open(root / "ranking.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields_out)
            writer.writeheader()
            for rank, row in enumerate(ranking, 1):
                writer.writerow({**row, "rank": rank, "params": json.dumps(row["params"], sort_keys=True)})
    return result


def _candidate_subset(spec: Mapping[str, Any], phase_name: str, candidates: Sequence[Mapping[str, Any]], frozen: str | None) -> List[Dict[str, Any]]:
    phase = spec["phases"][phase_name]
    source = phase.get("source", "all")
    by_id = {row["candidate_id"]: dict(row) for row in candidates}
    if source == "all":
        return list(by_id.values())
    if source.startswith("top_from:"):
        previous = source.split(":", 1)[1]
        ranking = read_json(Path(spec["output_root"]) / previous / "ranking.json")["ranking"]
        return [by_id[row["candidate_id"]] for row in ranking[: int(phase["top_k"])] if row["candidate_id"] in by_id]
    if source == "frozen":
        if not frozen:
            raise ValueError(f"Phase {phase_name} requires --frozen")
        data = read_json(Path(frozen))
        return [{"candidate_id": data["candidate_id"], "params": data["params"], "start": None, "trial": None}]
    raise ValueError(f"Unknown candidate source: {source}")


def freeze_best(spec: Mapping[str, Any], phase_name: str, output: str | Path) -> Dict[str, Any]:
    ranking_path = Path(spec["output_root"]) / phase_name / "ranking.json"
    ranking = read_json(ranking_path)["ranking"]
    if not ranking:
        raise ValueError(f"No completed candidates in {ranking_path}")
    best = ranking[0]
    base = _load_config(spec["base_config"])
    for key, value in best["params"].items():
        setattr(base, key, value)
    result = {
        "format": "smbh-cv-frozen-hyperparameters-v1",
        "source_phase": phase_name,
        "candidate_id": best["candidate_id"],
        "params": best["params"],
        "selection_metrics": {key: value for key, value in best.items() if key not in ("params",)},
        "full_pipeline_config": base.to_dict(),
        "config_fingerprint": _fingerprint(base.to_dict()),
        "warning": "Do not change this file after opening the test split; create a new protocol version instead.",
    }
    dump_json(Path(output), result)
    return result


def load_spec(path: str | Path) -> Dict[str, Any]:
    spec = read_json(Path(path))
    base = _load_config(spec["base_config"])
    for key, expected in spec.get("locked_method_hypotheses", {}).items():
        actual = getattr(base, key, None)
        if actual != expected:
            raise ValueError(f"Locked method {key} must be {expected!r}, found {actual!r}")
        if key in spec.get("search_space", {}):
            raise ValueError(f"Locked method {key} may not appear in search_space")
    return spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and execute resumable multi-start D1-D6 hyperparameter experiments.")
    parser.add_argument("--spec", default="experiments/article_v1/search_plan.json")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    run = sub.add_parser("run")
    run.add_argument("--phase", required=True)
    run.add_argument("--max-candidates", type=int)
    run.add_argument("--frozen")
    run.add_argument("--restart", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--phase", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--phase", default="validation")
    freeze.add_argument("--out", default="experiments/article_v1/frozen_config.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    spec = load_spec(args.spec)
    candidates = create_candidates(spec)
    plan_path = Path(spec["output_root"]) / "candidate_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(plan_path, {"spec_fingerprint": _fingerprint(spec), "candidates": candidates})
    if args.command == "plan":
        print(json.dumps({"candidate_count": len(candidates), "path": str(plan_path)}, ensure_ascii=False))
        return 0
    if args.command == "summarize":
        result = summarize_phase(spec, args.phase, candidates)
        print(json.dumps({"phase": args.phase, "completed_candidates": len(result["ranking"])}, ensure_ascii=False))
        return 0
    if args.command == "freeze":
        result = freeze_best(spec, args.phase, args.out)
        print(json.dumps({"candidate_id": result["candidate_id"], "out": args.out}, ensure_ascii=False))
        return 0
    selected = _candidate_subset(spec, args.phase, candidates, args.frozen)
    if args.max_candidates is not None:
        selected = selected[: args.max_candidates]
    seeds = [int(x) for x in spec["phases"][args.phase]["algorithm_seeds"]]
    for candidate_index, candidate in enumerate(selected, 1):
        for seed in seeds:
            logging.info(
                "Running candidate %d/%d id=%s seed=%d",
                candidate_index, len(selected), candidate["candidate_id"], seed,
            )
            _run_one(spec, args.phase, candidate, seed, restart=args.restart)
    result = summarize_phase(spec, args.phase, selected)
    print(json.dumps({"phase": args.phase, "completed_candidates": len(result["ranking"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
