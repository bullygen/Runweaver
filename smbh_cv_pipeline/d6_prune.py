from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import math

import numpy as np

from .config import PipelineConfig
from .io_utils import dump_json, read_json, run_paths, measured, write_worker_metric
from .models import meshgrid_xy, ring_model, smbh_model_image


def _merge_duplicates(rings: List[Dict[str, float]], cfg: PipelineConfig) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    if len(rings) <= 1:
        return rings, []
    order = sorted(range(len(rings)), key=lambda i: float(rings[i].get("source_confidence", rings[i].get("A", 0.0))), reverse=True)
    kept: List[Dict[str, float]] = []
    removed: List[Dict[str, Any]] = []
    for idx in order:
        r = rings[idx]
        duplicate_of = None
        for j, k in enumerate(kept):
            if np.hypot(r["x0"] - k["x0"], r["y0"] - k["y0"]) <= cfg.d6_merge_center_px and abs(r["R"] - k["R"]) <= cfg.d6_merge_radius_px:
                duplicate_of = j
                break
        if duplicate_of is None:
            kept.append(r)
        else:
            removed.append({"index": idx, "reason": "duplicate_merge", "duplicate_of_kept_index": duplicate_of, "ring": r})
    return kept, removed


def _ring_contributions(image: np.ndarray, rings: Sequence[Dict[str, float]], env: Dict[str, float], cfg: PipelineConfig) -> List[Dict[str, float]]:
    ny, nx = image.shape
    X, Y = meshgrid_xy(nx, ny)
    full = smbh_model_image(X, Y, rings, env)
    resid = full - image
    rss_full = float(np.sum(resid ** 2))
    out = []
    for i, r in enumerate(rings):
        no_ring = full - ring_model(X, Y, r)
        rss_without = float(np.sum((no_ring - image) ** 2))
        delta_rss = rss_without - rss_full
        # Positive delta means ring improves fit.
        n = image.size
        delta_bic_proxy = float(n * np.log(max(rss_without, 1e-300) / max(rss_full, 1e-300)) - 7 * np.log(n))
        out.append({"index": i, "delta_rss": delta_rss, "delta_bic_proxy": delta_bic_proxy, "A": float(r["A"])})
    return out


def _normal_sf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _bh_fdr_mask(pvalues: Sequence[float], alpha: float) -> List[bool]:
    p = np.asarray(pvalues, dtype=np.float64)
    ok = np.isfinite(p)
    keep = np.zeros(len(p), dtype=bool)
    if not np.any(ok):
        return keep.tolist()
    idx = np.where(ok)[0]
    p_ok = p[idx]
    order = np.argsort(p_ok)
    sorted_p = p_ok[order]
    m = len(sorted_p)
    thresholds = float(alpha) * np.arange(1, m + 1, dtype=np.float64) / max(m, 1)
    passed = sorted_p <= thresholds
    if np.any(passed):
        k = int(np.max(np.where(passed)[0]))
        cutoff = sorted_p[k]
        keep[idx] = p_ok <= cutoff
    return keep.tolist()


def _stderr_for_ring(ring: Dict[str, float], stderr: Sequence[Dict[str, Any]], fallback_index: int) -> Dict[str, Any] | None:
    """Return the stderr record corresponding to a ring after optional filtering/merge.

    Rings are tagged with source_fit_index before D6 pruning. This keeps FDR amplitude
    p-values aligned with the covariance entries from D5 even after amplitude filtering
    or duplicate merging changes the order and length of the ring list.
    """
    idx = ring.get("source_fit_index", fallback_index)
    try:
        idx_i = int(idx)
    except Exception:
        idx_i = fallback_index
    if 0 <= idx_i < len(stderr) and isinstance(stderr[idx_i], dict):
        return stderr[idx_i]
    if 0 <= fallback_index < len(stderr) and isinstance(stderr[fallback_index], dict):
        return stderr[fallback_index]
    return None


def _amplitude_pvalues(rings: Sequence[Dict[str, float]], stderr: Sequence[Dict[str, Any]]) -> List[float]:
    pvals: List[float] = []
    for i, r in enumerate(rings):
        er = _stderr_for_ring(r, stderr, i)
        A = float(r.get("A", 0.0))
        se = er.get("A") if isinstance(er, dict) else None
        if se is None or not np.isfinite(float(se)) or float(se) <= 0:
            pvals.append(float("nan"))
        else:
            pvals.append(_normal_sf(A / float(se)))
    return pvals


def _apply_amplitude_threshold(current: List[Dict[str, float]], cfg: PipelineConfig, reason_prefix: str = "amplitude") -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    """Drop rings with fitted amplitude below cfg.d6_amp_min.

    The strongest rings are forced to remain if the threshold would reduce the result
    below cfg.d6_min_artifacts. This helper is used both as a standalone D6 method and
    as a pre-filter for chained amplitude+merge methods.
    """
    history: List[Dict[str, Any]] = []
    if len(current) <= cfg.d6_min_artifacts:
        return current, history
    keep_mask = [float(r.get("A", 0.0)) >= float(cfg.d6_amp_min) for r in current]
    if sum(keep_mask) < cfg.d6_min_artifacts:
        order = np.argsort([-float(r.get("A", 0.0)) for r in current])
        force = set(int(i) for i in order[: cfg.d6_min_artifacts])
    else:
        force = set()
    final: List[Dict[str, float]] = []
    for i, r in enumerate(current):
        keep_i = bool(keep_mask[i]) or i in force
        if keep_i:
            final.append(r)
        else:
            history.append({
                "index": i,
                "source_fit_index": r.get("source_fit_index"),
                "reason": f"{reason_prefix}_below_threshold",
                "A": float(r.get("A", 0.0)),
                "amp_min": float(cfg.d6_amp_min),
            })
    return final, history


def _apply_fdr_bh_amplitude(current: List[Dict[str, float]], fit: Dict[str, Any], cfg: PipelineConfig) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []
    if len(current) <= cfg.d6_min_artifacts:
        return current, history
    stderr = fit.get("stderr_artifacts", [])
    cov_ok = bool(fit.get("covar_available", False)) and len(stderr) >= len(fit.get("artifacts", []))
    if not cov_ok:
        action = cfg.d6_fdr_no_covar_action
        history.append({"reason": "fdr_no_covariance", "action": action})
        if action == "keep_all":
            return current, history
        if action == "amplitude":
            return _apply_amplitude_threshold(current, cfg, reason_prefix="fdr_fallback_amplitude")
        if action == "drop_min_amplitude" and len(current) > cfg.d6_min_artifacts:
            drop = int(np.argmin([float(r.get("A", 0.0)) for r in current]))
            final = [r for i, r in enumerate(current) if i != drop]
            history.append({"index": drop, "source_fit_index": current[drop].get("source_fit_index"), "reason": "fdr_fallback_drop_min_amplitude", "A": current[drop].get("A")})
            return final, history
        raise ValueError(f"Unknown d6_fdr_no_covar_action: {action}")

    pvals = _amplitude_pvalues(current, stderr)
    keep = _bh_fdr_mask(pvals, cfg.d6_fdr_alpha)
    final: List[Dict[str, float]] = []
    if sum(keep) < cfg.d6_min_artifacts:
        order = np.argsort(np.nan_to_num(np.asarray(pvals), nan=np.inf))
        force = set(int(i) for i in order[: cfg.d6_min_artifacts])
    else:
        force = set()
    for i, r in enumerate(current):
        keep_i = bool(keep[i]) or i in force
        record = {"fdr_pvalue_A": pvals[i], "fdr_alpha": cfg.d6_fdr_alpha}
        if keep_i:
            final.append(r | record)
        else:
            history.append({"index": i, "source_fit_index": r.get("source_fit_index"), "reason": "benjamini_hochberg_fdr_not_significant", **record, "A": r.get("A")})
    return final, history


def process_one(cfg_dict: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    cfg = PipelineConfig(**cfg_dict)
    paths = run_paths(cfg.out)
    metric: Dict[str, Any] = {}
    with measured("D6", image_id) as metric:
        fit = read_json(paths["d5"] / image_id / "d5_fit.json")
        image = np.load(paths["images"] / image_id / "image.npy").astype(np.float64)
        rings = [dict(r) | {"source_fit_index": i} for i, r in enumerate(fit.get("artifacts", []))]
        env = dict(fit.get("env", {}))
        history: List[Dict[str, Any]] = []
        if cfg.d6_method == "none":
            final = rings
        else:
            current = rings

            # Chained pruning modes: remove very weak fitted amplitudes before resolving
            # duplicates and then applying the selected statistical/model-selection rule.
            if cfg.d6_method in ("amplitude_merge_delta_bic", "amplitude_merge_fdr_bh_amplitude"):
                current, amp_history = _apply_amplitude_threshold(current, cfg, reason_prefix="prefilter_amplitude")
                history.extend(amp_history)

            if cfg.d6_method in ("merge_delta_bic", "merge_fdr_bh_amplitude", "amplitude_merge_delta_bic", "amplitude_merge_fdr_bh_amplitude"):
                current, removed = _merge_duplicates(current, cfg)
                history.extend(removed)

            if cfg.d6_method == "amplitude":
                final, amp_history = _apply_amplitude_threshold(current, cfg, reason_prefix="amplitude")
                history.extend(amp_history)
            elif cfg.d6_method in ("delta_bic", "merge_delta_bic", "amplitude_merge_delta_bic"):
                contrib = _ring_contributions(image, current, env, cfg)
                final = []
                # Keep rings whose contribution passes proxy BIC, but never below min_artifacts.
                sorted_by_strength = sorted(contrib, key=lambda x: x["delta_bic_proxy"], reverse=True)
                force_keep = {c["index"] for c in sorted_by_strength[: cfg.d6_min_artifacts]}
                for c, r in zip(contrib, current):
                    keep = c["delta_bic_proxy"] > cfg.d6_delta_bic_threshold or c["index"] in force_keep
                    if keep:
                        final.append(r | {"delta_bic_proxy": c["delta_bic_proxy"], "delta_rss": c["delta_rss"]})
                    else:
                        history.append({"index": c["index"], "source_fit_index": r.get("source_fit_index"), "reason": "delta_bic_proxy_below_threshold", **c})
            elif cfg.d6_method in ("fdr_bh_amplitude", "merge_fdr_bh_amplitude", "amplitude_merge_fdr_bh_amplitude"):
                final, fdr_history = _apply_fdr_bh_amplitude(current, fit, cfg)
                history.extend(fdr_history)
            else:
                raise ValueError(f"Unknown D6 method: {cfg.d6_method}")
        ny, nx = image.shape
        X, Y = meshgrid_xy(nx, ny)
        model = smbh_model_image(X, Y, final, env)
        residual = model - image
        out_dir = paths["d6"] / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.save_arrays:
            np.savez_compressed(out_dir / "d6_final_arrays.npz", model=model.astype(np.float32), residual=residual.astype(np.float32))
        result = {
            "image_id": image_id,
            "method": cfg.d6_method,
            "input_n": len(rings),
            "final_n": len(final),
            "final_artifacts": final,
            "env": env,
            "selection_history": history,
            "fit_source": str(paths["d5"] / image_id / "d5_fit.json"),
            "fit_success": fit.get("success"),
            "fit_statistics": fit.get("fit_statistics", {}),
        }
        dump_json(out_dir / "d6_final.json", result)
        metric.update({"input_n": len(rings), "final_n": len(final), "pruned_n": len(rings) - len(final)})
    write_worker_metric(cfg.out, metric)
    return {"image_id": image_id, "status": "ok", "final_n": metric.get("final_n")}
