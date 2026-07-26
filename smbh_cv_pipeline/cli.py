from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import psutil

from .config import PipelineConfig
from .generation import generate_one, write_manifest
from .io_utils import clean_run_dir, dump_json, image_ids_from_manifest, read_json, run_paths
from . import d1_preprocess, d2_hough, d3_candidates, d4_initialization, d5_fit, d6_prune, visualization
from .statistics import run_statistics

STAGE_FUNCS: Dict[str, Callable[[Dict[str, Any], str], Dict[str, Any]]] = {
    "d1": d1_preprocess.process_one,
    "d2": d2_hough.process_one,
    "d3": d3_candidates.process_one,
    "d4": d4_initialization.process_one,
    "d5": d5_fit.process_one,
    "d6": d6_prune.process_one,
}
STAGE_ORDER = ["generate", "d1", "d2", "d3", "d4", "d5", "d6", "stats", "viz"]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def _cfg_from_file(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_cfg(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    file_cfg = _cfg_from_file(args.config)
    for k, v in file_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            logging.warning("Ignoring unknown config key in file: %s", k)
    # Explicit CLI overrides only.
    keys = [
        "out", "seed", "n_images", "nx", "ny", "dataset_mode", "n_true_min", "n_true_max", "noise_sigma",
        "beam_fwhm_px", "noise_model", "corr_noise_length_px", "background_gradient", "partial_ring_prob", "overlap_mode", "reconstruction_method",
        "d1_mask_method", "d1_edge_method", "d1_threshold_method", "d1_morphology_method", "d1_edge_sigmas", "d1_tau", "d1_percentile", "d1_min_area", "d1_skeletonize", "d1_jet_percentile", "d1_rect_half_size", "d1_legacy_badpix", "d1_legacy_max_iter",
        "d2_method", "d2_vote_weight", "d2_votes_fraction", "d2_max_votes", "d2_min_triangle_area", "d2_center_bin_px", "d2_radius_bin_px", "d2_r_min", "d2_r_max", "d2_n_sectors",
        "d3_method", "d3_threshold_method", "d3_relative_threshold", "d3_quantile", "d3_bootstrap_quantile", "d3_bootstrap_repeats", "d3_nms_dx", "d3_nms_dr", "d3_max_candidates", "d3_expected_max_rings", "d3_top_k_for_ph", "d3_ph_memory_limit_gb",
        "d4_method", "d4_strip_half_width", "d4_radial_window", "d4_n_angle_bins", "d4_max_init_candidates",
        "d5_engine", "d5_loss", "d5_residual_weighting", "d5_max_nfev", "d5_fit_background", "d5_f_scale", "d5_fit_stride", "d5_local_then_global",
        "d6_method", "d6_min_artifacts", "d6_amp_min", "d6_delta_bic_threshold", "d6_fdr_alpha", "d6_fdr_no_covar_action", "d6_merge_center_px", "d6_merge_radius_px",
        "max_match_cost", "max_center_distance_px", "max_center_error_fraction", "max_radius_error_fraction", "min_annular_iou", "make_plots", "save_arrays", "viz_dpi", "viz_max_accumulator_points",
    ]
    for key in keys:
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)
    return cfg


def _pool_map(stage: str, func: Callable[[Dict[str, Any], str], Dict[str, Any]], cfg: PipelineConfig, image_ids: Sequence[str], workers: int) -> List[Dict[str, Any]]:
    workers = max(1, int(workers))
    logging.info("Starting %s with workers=%d for %d images", stage.upper(), workers, len(image_ids))
    t0 = time.perf_counter()
    if workers == 1:
        results = [func(cfg.to_dict(), image_id) for image_id in image_ids]
    else:
        ctx = mp.get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as ex:
            futs = [ex.submit(func, cfg.to_dict(), image_id) for image_id in image_ids]
            for fut in as_completed(futs):
                results.append(fut.result())
    elapsed = time.perf_counter() - t0
    dump_json(Path(cfg.out) / "runtime" / f"{stage}_stage_summary.json", {"stage": stage, "workers": workers, "n_items": len(image_ids), "wall_time_s": elapsed, "results": results})
    logging.info("Finished %s in %.3fs", stage.upper(), elapsed)
    return results


def run_generate(cfg: PipelineConfig, workers: int) -> List[Dict[str, Any]]:
    run_paths(cfg.out)
    logging.info("Starting GENERATE with workers=%d for %d images", workers, cfg.n_images)
    t0 = time.perf_counter()
    indices = list(range(cfg.n_images))
    if workers <= 1:
        items = [generate_one(cfg.to_dict(), i) for i in indices]
    else:
        ctx = mp.get_context("spawn")
        items = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as ex:
            futs = [ex.submit(generate_one, cfg.to_dict(), i) for i in indices]
            for fut in as_completed(futs):
                items.append(fut.result())
    items = sorted(items, key=lambda x: x["image_id"])
    write_manifest(cfg, items)
    elapsed = time.perf_counter() - t0
    dump_json(Path(cfg.out) / "runtime" / "generate_stage_summary.json", {"stage": "generate", "workers": workers, "n_items": len(items), "wall_time_s": elapsed, "results": items})
    logging.info("Finished GENERATE in %.3fs", elapsed)
    return items


def validate_worker_plan(cfg: PipelineConfig, args: argparse.Namespace) -> None:
    mem_total_gb = psutil.virtual_memory().total / (1024 ** 3)
    if cfg.d3_method in ("ph_safe_proxy", "ph_cripser_dense") and args.workers_d3 > 1:
        est_max = max(1, int(math.floor(mem_total_gb / 10.0)))
        if args.workers_d3 > est_max and not args.allow_unsafe_d3_workers:
            logging.warning("D3 dense/PH-safe mode assumes up to 10 GB per worker. Capping workers_d3 from %d to %d. Use --allow-unsafe-d3-workers to override.", args.workers_d3, est_max)
            args.workers_d3 = est_max


def run_command(command: str, cfg: PipelineConfig, args: argparse.Namespace) -> None:
    paths = run_paths(cfg.out)
    dump_json(paths["config"], cfg.to_dict())
    validate_worker_plan(cfg, args)
    if command == "generate":
        run_generate(cfg, args.workers_generate)
        return
    if command in STAGE_FUNCS:
        image_ids = image_ids_from_manifest(cfg.out)
        _pool_map(command, STAGE_FUNCS[command], cfg, image_ids, getattr(args, f"workers_{command}"))
        return
    if command == "stats":
        t0 = time.perf_counter()
        summary = run_statistics(cfg)
        dump_json(Path(cfg.out) / "runtime" / "stats_stage_summary.json", {"stage": "stats", "workers": 1, "wall_time_s": time.perf_counter() - t0})
        logging.info("Statistics summary: %s", summary)
        return
    if command == "viz":
        image_ids = image_ids_from_manifest(cfg.out)
        _pool_map("viz", visualization.process_one, cfg, image_ids, args.workers_viz)
        summary = visualization.run_visualization(cfg, image_ids)
        dump_json(Path(cfg.out) / "runtime" / "viz_stage_summary.json", {"stage": "viz", "workers": args.workers_viz, "summary": summary})
        logging.info("Visualization summary: %s", summary)
        return
    if command == "all":
        if args.clean:
            clean_run_dir(cfg.out)
            run_paths(cfg.out)
        run_generate(cfg, args.workers_generate)
        image_ids = image_ids_from_manifest(cfg.out)
        for st in ["d1", "d2", "d3", "d4", "d5", "d6"]:
            _pool_map(st, STAGE_FUNCS[st], cfg, image_ids, getattr(args, f"workers_{st}"))
        t0 = time.perf_counter()
        run_statistics(cfg)
        dump_json(Path(cfg.out) / "runtime" / "stats_stage_summary.json", {"stage": "stats", "workers": 1, "wall_time_s": time.perf_counter() - t0})
        if cfg.make_plots:
            _pool_map("viz", visualization.process_one, cfg, image_ids, args.workers_viz)
            visualization.run_visualization(cfg, image_ids)
        return
    if command == "benchmark_one":
        args.clean = True
        clean_run_dir(cfg.out)
        cfg.n_images = 1
        run_paths(cfg.out)
        run_generate(cfg, args.workers_generate)
        image_ids = image_ids_from_manifest(cfg.out)
        for st in ["d1", "d2", "d3", "d4", "d5", "d6"]:
            _pool_map(st, STAGE_FUNCS[st], cfg, image_ids, getattr(args, f"workers_{st}"))
        summary = run_statistics(cfg)
        dump_json(Path(cfg.out) / "runtime" / "benchmark_one_summary.json", summary)
        if cfg.make_plots:
            _pool_map("viz", visualization.process_one, cfg, image_ids, args.workers_viz)
            visualization.run_visualization(cfg, image_ids)
        return
    raise ValueError(f"Unknown command: {command}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Modular D1-D6 SMBH ring morphology pipeline with per-stage workers and branch arguments.")
    p.add_argument("command", choices=["all", "benchmark_one", "generate", "d1", "d2", "d3", "d4", "d5", "d6", "stats", "viz"])
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--log-level", default="INFO")

    # Worker controls.
    p.add_argument("--workers-generate", type=int, default=1)
    p.add_argument("--workers-d1", type=int, default=1)
    p.add_argument("--workers-d2", type=int, default=1)
    p.add_argument("--workers-d3", type=int, default=1)
    p.add_argument("--workers-d4", type=int, default=1)
    p.add_argument("--workers-d5", type=int, default=1)
    p.add_argument("--workers-d6", type=int, default=1)
    p.add_argument("--workers-viz", type=int, default=1)
    p.add_argument("--allow-unsafe-d3-workers", action="store_true")

    # General / generator.
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-images", type=int, default=None)
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--dataset-mode", choices=["simple_model", "degraded_model"], default=None)
    p.add_argument("--n-true-min", type=int, default=None)
    p.add_argument("--n-true-max", type=int, default=None)
    p.add_argument("--noise-sigma", type=float, default=None)
    p.add_argument("--beam-fwhm-px", type=float, default=None)
    p.add_argument("--noise-model", choices=["none", "gaussian", "correlated"], default=None)
    p.add_argument("--corr-noise-length-px", type=float, default=None)
    p.add_argument("--background-gradient", type=float, default=None)
    p.add_argument("--partial-ring-prob", type=float, default=None)
    p.add_argument("--overlap-mode", choices=["random", "clustered"], default=None)
    p.add_argument("--reconstruction-method", default=None)

    # D1 branch args.
    p.add_argument("--d1-mask-method", choices=["none", "percentile_cc", "rect_max"], default=None)
    p.add_argument("--d1-edge-method", choices=["legacy_laplace_positive", "log_mad", "dog", "sobel", "laplace_sign"], default=None)
    p.add_argument("--d1-threshold-method", choices=["positive", "mad", "percentile", "sauvola"], default=None)
    p.add_argument("--d1-morphology-method", choices=["none", "open_close", "remove_small", "legacy_control_open"], default=None)
    p.add_argument("--d1-edge-sigmas", default=None)
    p.add_argument("--d1-tau", type=float, default=None)
    p.add_argument("--d1-percentile", type=float, default=None)
    p.add_argument("--d1-min-area", type=int, default=None)
    p.add_argument("--d1-skeletonize", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--d1-jet-percentile", type=float, default=None)
    p.add_argument("--d1-rect-half-size", type=int, default=None)
    p.add_argument("--d1-legacy-badpix", type=float, default=None)
    p.add_argument("--d1-legacy-max-iter", type=int, default=None)

    # D2 branch args.
    p.add_argument("--d2-method", choices=["sparse_rht", "sparse_rht_stratified"], default=None)
    p.add_argument("--d2-vote-weight", choices=["none", "mean", "product"], default=None)
    p.add_argument("--d2-votes-fraction", type=float, default=None)
    p.add_argument("--d2-max-votes", type=int, default=None)
    p.add_argument("--d2-min-triangle-area", type=float, default=None)
    p.add_argument("--d2-center-bin-px", type=float, default=None)
    p.add_argument("--d2-radius-bin-px", type=float, default=None)
    p.add_argument("--d2-r-min", type=float, default=None)
    p.add_argument("--d2-r-max", type=float, default=None)
    p.add_argument("--d2-n-sectors", type=int, default=None)

    # D3 branch args.
    p.add_argument("--d3-method", choices=["peak_nms", "bootstrap_nms", "ph_safe_proxy", "ph_union_find_sparse", "ph_cripser_dense"], default=None)
    p.add_argument("--d3-threshold-method", choices=["relative", "quantile", "bootstrap"], default=None)
    p.add_argument("--d3-relative-threshold", type=float, default=None)
    p.add_argument("--d3-quantile", type=float, default=None)
    p.add_argument("--d3-bootstrap-quantile", type=float, default=None)
    p.add_argument("--d3-bootstrap-repeats", type=int, default=None)
    p.add_argument("--d3-nms-dx", type=float, default=None)
    p.add_argument("--d3-nms-dr", type=float, default=None)
    p.add_argument("--d3-max-candidates", type=int, default=None)
    p.add_argument("--d3-expected-max-rings", type=int, default=None)
    p.add_argument("--d3-top-k-for-ph", type=int, default=None)
    p.add_argument("--d3-ph-memory-limit-gb", type=float, default=None)

    # D4-D6 branch args.
    p.add_argument("--d4-method", choices=["profile_harmonic", "constants"], default=None)
    p.add_argument("--d4-strip-half-width", type=float, default=None)
    p.add_argument("--d4-radial-window", type=float, default=None)
    p.add_argument("--d4-n-angle-bins", type=int, default=None)
    p.add_argument("--d4-max-init-candidates", type=int, default=None)

    p.add_argument("--d5-engine", choices=["scipy"], default=None)
    p.add_argument("--d5-loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default=None)
    p.add_argument("--d5-residual-weighting", choices=["uniform", "ring_support", "inverse_background"], default=None)
    p.add_argument("--d5-max-nfev", type=int, default=None)
    p.add_argument("--d5-fit-background", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--d5-f-scale", type=float, default=None)
    p.add_argument("--d5-fit-stride", type=int, default=None)
    p.add_argument("--d5-local-then-global", action=argparse.BooleanOptionalAction, default=None)

    p.add_argument("--d6-method", choices=["none", "amplitude", "delta_bic", "merge_delta_bic", "fdr_bh_amplitude", "merge_fdr_bh_amplitude", "amplitude_merge_delta_bic", "amplitude_merge_fdr_bh_amplitude"], default=None)
    p.add_argument("--d6-min-artifacts", type=int, default=None)
    p.add_argument("--d6-amp-min", type=float, default=None)
    p.add_argument("--d6-delta-bic-threshold", type=float, default=None)
    p.add_argument("--d6-fdr-alpha", type=float, default=None)
    p.add_argument("--d6-fdr-no-covar-action", choices=["keep_all", "amplitude", "drop_min_amplitude"], default=None)
    p.add_argument("--d6-merge-center-px", type=float, default=None)
    p.add_argument("--d6-merge-radius-px", type=float, default=None)

    p.add_argument("--max-match-cost", type=float, default=None)
    p.add_argument("--max-center-distance-px", type=float, default=None)
    p.add_argument("--max-center-error-fraction", type=float, default=None)
    p.add_argument("--max-radius-error-fraction", type=float, default=None)
    p.add_argument("--min-annular-iou", type=float, default=None)
    p.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=None, help="If true, command all/benchmark_one also runs the visualization stage after stats.")
    p.add_argument("--save-arrays", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--viz-dpi", type=int, default=None)
    p.add_argument("--viz-max-accumulator-points", type=int, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    cfg = _update_cfg(PipelineConfig(), args)
    try:
        run_command(args.command, cfg, args)
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
