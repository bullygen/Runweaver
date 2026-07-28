from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kurtosis, normaltest, probplot, skew

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from .config import PARAM_NAMES, PipelineConfig
from .io_utils import dump_json, image_ids_from_manifest, read_json, run_paths
from .models import angle_diff, meshgrid_xy, smbh_model_image


def _finite(vals: Iterable[Any]) -> np.ndarray:
    out = []
    for v in vals:
        try:
            f = float(v)
            if np.isfinite(f):
                out.append(f)
        except Exception:
            pass
    return np.asarray(out, dtype=float)


def _safe_mean(vals: Iterable[Any]) -> float:
    arr = _finite(vals)
    return float(np.mean(arr)) if len(arr) else float("nan")


def _safe_median(vals: Iterable[Any]) -> float:
    arr = _finite(vals)
    return float(np.median(arr)) if len(arr) else float("nan")


def _annular_iou(t: Dict[str, float], f: Dict[str, float], nx: int, ny: int) -> float:
    X, Y = meshgrid_xy(nx, ny)
    rt = np.sqrt((X - t["x0"]) ** 2 + (Y - t["y0"]) ** 2)
    rf = np.sqrt((X - f["x0"]) ** 2 + (Y - f["y0"]) ** 2)
    mt = np.abs(rt - t["R"]) <= max(t["sigma"], 1.0)
    mf = np.abs(rf - f["R"]) <= max(f["sigma"], 1.0)
    inter = int(np.sum(mt & mf))
    union = int(np.sum(mt | mf))
    return float(inter / union) if union else 0.0


def _artifact_cost(t: Dict[str, float], f: Dict[str, float], bounds: Dict[str, Sequence[float]], nx: int, ny: int) -> float:
    diag = float(np.hypot(nx, ny))
    terms = []
    for p in ["A", "R", "sigma", "B"]:
        lo, hi = bounds[p]
        terms.append(((f[p] - t[p]) / max(hi - lo, 1e-9)) ** 2)
    terms.append((angle_diff(f["phi"], t["phi"]) / np.pi) ** 2)
    terms.append(((f["x0"] - t["x0"]) / diag * 4.0) ** 2)
    terms.append(((f["y0"] - t["y0"]) / diag * 4.0) ** 2)
    return float(np.sqrt(np.sum(terms)))


def _match(true_rings: Sequence[Dict[str, float]], fit_rings: Sequence[Dict[str, float]], cfg: PipelineConfig, nx: int, ny: int) -> Tuple[List[Dict[str, Any]], int, int, int]:
    if not true_rings or not fit_rings:
        return [], 0, len(fit_rings), len(true_rings)
    cost = np.zeros((len(true_rings), len(fit_rings)), dtype=float)
    for i, t in enumerate(true_rings):
        for j, f in enumerate(fit_rings):
            cost[i, j] = _artifact_cost(t, f, cfg.fit_bounds, nx, ny)
    rows, cols = linear_sum_assignment(cost)
    matches = []
    tp = 0
    for i, j in zip(rows, cols):
        center_distance = float(np.hypot(fit_rings[j]["x0"] - true_rings[i]["x0"], fit_rings[j]["y0"] - true_rings[i]["y0"]))
        iou = _annular_iou(true_rings[i], fit_rings[j], nx, ny)
        is_tp = bool(cost[i, j] <= cfg.max_match_cost and center_distance <= cfg.max_center_distance_px)
        tp += int(is_tp)
        matches.append({"true_index": int(i), "fit_index": int(j), "cost": float(cost[i, j]), "center_distance": center_distance, "annular_iou": iou, "is_tp": is_tp})
    return matches, tp, len(fit_rings) - tp, len(true_rings) - tp


def _match_article(true_rings: Sequence[Dict[str, float]], fit_rings: Sequence[Dict[str, float]], cfg: PipelineConfig, nx: int, ny: int) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Article-primary object matching based on observable ring geometry.

    Unlike the legacy seven-parameter cost used by the historical spreadsheet,
    this criterion does not require a model-mismatch profile to recover the
    generator's amplitude/asymmetry parametrization.  Thresholds are dimensionless
    center/radius errors plus annular mask IoU and are frozen in PipelineConfig.
    """
    if not true_rings or not fit_rings:
        return [], 0, len(fit_rings), len(true_rings)
    cost = np.zeros((len(true_rings), len(fit_rings)), dtype=float)
    diagnostics: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for i, truth in enumerate(true_rings):
        scale = max(abs(float(truth["R"])), 1e-9)
        for j, fit in enumerate(fit_rings):
            center_fraction = float(np.hypot(fit["x0"] - truth["x0"], fit["y0"] - truth["y0"]) / scale)
            radius_fraction = float(abs(fit["R"] - truth["R"]) / scale)
            iou = _annular_iou(truth, fit, nx, ny)
            diagnostics[(i, j)] = (center_fraction, radius_fraction, iou)
            cost[i, j] = float(np.sqrt(center_fraction ** 2 + radius_fraction ** 2 + (1.0 - iou) ** 2))
    rows, cols = linear_sum_assignment(cost)
    matches: List[Dict[str, Any]] = []
    tp = 0
    for i, j in zip(rows, cols):
        center_fraction, radius_fraction, iou = diagnostics[(int(i), int(j))]
        is_tp = bool(
            center_fraction <= cfg.max_center_error_fraction
            and radius_fraction <= cfg.max_radius_error_fraction
            and iou >= cfg.min_annular_iou
        )
        tp += int(is_tp)
        matches.append({
            "true_index": int(i), "fit_index": int(j), "cost": float(cost[i, j]),
            "center_error_fraction": center_fraction, "radius_error_fraction": radius_fraction,
            "center_distance": center_fraction * max(abs(float(true_rings[i]["R"])), 1e-9),
            "annular_iou": iou, "is_tp": is_tp, "match_protocol": "article_geometric_iou_v1",
        })
    return matches, tp, len(fit_rings) - tp, len(true_rings) - tp


def _candidate_stage_recall(cands: Sequence[Dict[str, float]], true_rings: Sequence[Dict[str, float]], cfg: PipelineConfig, nx: int, ny: int) -> float:
    """Recall of geometric D3 proposals before D4/D5 nuisance parameters exist.

    The previous implementation filled A, sigma, B and phi with constants and
    then used the seven-parameter D5/D6 matching cost.  That can report a lower
    D3 recall than final recall even when the correct center/radius proposal was
    present.  D3 only hypothesizes (x0, y0, R), so this diagnostic must only use
    those explicitly available quantities.
    """
    if not true_rings:
        return float("nan")
    if not cands:
        return 0.0
    cost = np.zeros((len(true_rings), len(cands)), dtype=float)
    for i, truth in enumerate(true_rings):
        for j, cand in enumerate(cands):
            center = np.hypot(float(cand["x0"]) - truth["x0"], float(cand["y0"]) - truth["y0"])
            radius = abs(float(cand["R"]) - truth["R"])
            cost[i, j] = np.hypot(center / max(cfg.max_center_distance_px, 1e-9), radius / max(cfg.max_center_distance_px, 1e-9))
    rows, cols = linear_sum_assignment(cost)
    tp = 0
    for i, j in zip(rows, cols):
        center = np.hypot(float(cands[j]["x0"]) - true_rings[i]["x0"], float(cands[j]["y0"]) - true_rings[i]["y0"])
        radius = abs(float(cands[j]["R"]) - true_rings[i]["R"])
        tp += int(center <= cfg.max_center_distance_px and radius <= cfg.max_center_distance_px)
    return float(tp / len(true_rings))


def _counts_to_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else float("nan")
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _bootstrap_detection_ci(rows: Sequence[Dict[str, Any]], repeats: int = 2000, seed: int = 91573, prefix: str = "") -> Dict[str, List[float]]:
    """Cluster bootstrap over images, preserving within-image ring dependence."""
    if not rows:
        return {key: [float("nan"), float("nan")] for key in ("precision", "recall", "f1", "null_fppi")}
    counts = np.asarray([[r[f"{prefix}tp"], r[f"{prefix}fp"], r[f"{prefix}fn"], int(r["true_n"] == 0), r[f"{prefix}fp"] if r["true_n"] == 0 else 0] for r in rows], dtype=int)
    rng = np.random.default_rng(seed)
    values: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": [], "null_fppi": []}
    n = len(rows)
    for _ in range(max(1, repeats)):
        sample = counts[rng.integers(0, n, size=n)].sum(axis=0)
        metrics = _counts_to_metrics(int(sample[0]), int(sample[1]), int(sample[2]))
        for key in ("precision", "recall", "f1"):
            if np.isfinite(metrics[key]):
                values[key].append(metrics[key])
        if sample[3] > 0:
            values["null_fppi"].append(float(sample[4] / sample[3]))
    result: Dict[str, List[float]] = {}
    for key, vals in values.items():
        arr = _finite(vals)
        result[key] = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if len(arr) else [float("nan"), float("nan")]
    return result


def _factor_slices(per_image: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    factor_keys = sorted({key for row in per_image for key in row if key.startswith("factor_")})
    out: List[Dict[str, Any]] = []
    for key in factor_keys:
        values = sorted({str(row.get(key)) for row in per_image})
        for value in values:
            rows = [row for row in per_image if str(row.get(key)) == value]
            tp = int(sum(int(row["article_tp"]) for row in rows))
            fp = int(sum(int(row["article_fp"]) for row in rows))
            fn = int(sum(int(row["article_fn"]) for row in rows))
            n_null = sum(int(row["true_n"] == 0) for row in rows)
            null_fp = sum(int(row["article_fp"]) for row in rows if row["true_n"] == 0)
            metrics = _counts_to_metrics(tp, fp, fn)
            seed_bytes = hashlib.sha256(f"{key}\0{value}".encode("utf-8")).digest()[:4]
            ci = _bootstrap_detection_ci(rows, repeats=500, seed=int.from_bytes(seed_bytes, "little"), prefix="article_")
            out.append({
                "match_protocol": "article_geometric_iou_v1",
                "factor": key.removeprefix("factor_"),
                "level": value,
                "n_images": len(rows),
                "n_positive_images": len(rows) - n_null,
                "n_null_images": n_null,
                "tp": tp, "fp": fp, "fn": fn,
                **metrics,
                "precision_ci95_low": ci["precision"][0], "precision_ci95_high": ci["precision"][1],
                "recall_ci95_low": ci["recall"][0], "recall_ci95_high": ci["recall"][1],
                "f1_ci95_low": ci["f1"][0], "f1_ci95_high": ci["f1"][1],
                "null_fppi": float(null_fp / n_null) if n_null else float("nan"),
            })
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _parameter_metrics(param_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    param_metrics: List[Dict[str, Any]] = []
    for p in PARAM_NAMES:
        rows = [r for r in param_rows if r["param"] == p and r["is_tp"]]
        err = _finite(r["error"] for r in rows)
        true = _finite(r["true"] for r in rows)
        fit = _finite(r["fit"] for r in rows)
        if len(err):
            tss = float(np.sum((true - np.mean(true)) ** 2)) if len(true) else 0.0
            r2 = float(1.0 - np.sum((fit - true) ** 2) / tss) if tss > 0 and len(fit) == len(true) else float("nan")
            corr = float(np.corrcoef(true, fit)[0, 1]) if len(true) > 1 and np.std(true) > 0 and np.std(fit) > 0 else float("nan")
            param_metrics.append({
                "param": p,
                "n": int(len(err)),
                "bias": float(np.mean(err)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "median_abs_error": float(np.median(np.abs(err))),
                "std_error": float(np.std(err, ddof=1)) if len(err) > 1 else 0.0,
                "corr": corr,
                "r2": r2,
                "true_mean": float(np.mean(true)) if len(true) else float("nan"),
                "fit_mean": float(np.mean(fit)) if len(fit) else float("nan"),
                "true_std": float(np.std(true, ddof=1)) if len(true) > 1 else 0.0,
                "fit_std": float(np.std(fit, ddof=1)) if len(fit) > 1 else 0.0,
            })
        else:
            param_metrics.append({"param": p, "n": 0})
    return param_metrics


def _save_fig(fig: Any, path: Path, dpi: int) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return {"path": str(path), "filename": path.name}


def _hist_plot(outdir: Path, name: str, values: np.ndarray, xlabel: str, title: str, dpi: int, bins: int = 25) -> List[Dict[str, Any]]:
    if plt is None or len(values) == 0:
        return []
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    ax.hist(values, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    return [_save_fig(fig, outdir / f"{name}.png", dpi)]


def _scatter_identity_plot(outdir: Path, name: str, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, dpi: int) -> List[Dict[str, Any]]:
    if plt is None or len(x) == 0 or len(y) == 0:
        return []
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    ax.scatter(x, y, s=18)
    if len(x) and len(y):
        mn = float(min(np.min(x), np.min(y)))
        mx = float(max(np.max(x), np.max(y)))
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return [_save_fig(fig, outdir / f"{name}.png", dpi)]


def make_aggregate_plots(
    stats_dir: Path,
    per_image_rows: Sequence[Dict[str, Any]],
    match_rows: Sequence[Dict[str, Any]],
    param_error_rows: Sequence[Dict[str, Any]],
    residual_chunks: Sequence[np.ndarray],
    summary: Dict[str, Any],
    dpi: int = 140,
) -> List[Dict[str, Any]]:
    """Create aggregate PNG diagnostics under statistics/plots.

    The function intentionally uses only files already produced by D1-D6 and the
    rows computed by run_statistics. It does not read per-image heavy artifacts.
    """
    if plt is None:
        return []
    outdir = stats_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    ok = list(per_image_rows)

    def arr(key: str) -> np.ndarray:
        return _finite(r.get(key) for r in ok)

    # Global detection count summary.
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    labels = ["TP", "FP", "FN"]
    vals = [summary.get("article_tp", summary.get("global_tp", 0)), summary.get("article_fp", summary.get("global_fp", 0)), summary.get("article_fn", summary.get("global_fn", 0))]
    ax.bar(labels, vals)
    ax.set_ylabel("rings")
    ax.set_title("Global detection counts")
    for i, v in enumerate(vals):
        ax.text(i, float(v), str(v), ha="center", va="bottom")
    manifest.append(_save_fig(fig, outdir / "global_detection_counts.png", dpi))

    # True/candidate/fit counts per image.
    idx = np.arange(len(ok))
    if len(idx):
        fig, ax = plt.subplots(figsize=(max(6.0, 0.12 * len(ok) + 4.0), 4.2), constrained_layout=True)
        ax.plot(idx, arr("true_n"), marker="o", linewidth=1, label="true_n")
        ax.plot(idx, arr("candidate_n"), marker="o", linewidth=1, label="candidate_n")
        ax.plot(idx, arr("fit_n"), marker="o", linewidth=1, label="fit_n")
        ax.set_xlabel("image index")
        ax.set_ylabel("count")
        ax.set_title("Ring counts by image")
        ax.legend()
        manifest.append(_save_fig(fig, outdir / "counts_by_image.png", dpi))

    true_n = arr("true_n")
    cand_n = arr("candidate_n")
    fit_n = arr("fit_n")
    if len(true_n) and len(fit_n):
        manifest.extend(_scatter_identity_plot(outdir, "count_true_vs_fit", true_n, fit_n, "true N", "fit N", "Artifact count: true vs fit", dpi))
        manifest.extend(_hist_plot(outdir, "count_error_hist", fit_n - true_n, "fit N - true N", "Count error", dpi, bins=max(5, int(np.ptp(fit_n - true_n) + 3)) if len(fit_n) else 10))
    if len(true_n) and len(cand_n):
        manifest.extend(_scatter_identity_plot(outdir, "count_true_vs_candidates", true_n, cand_n, "true N", "candidate N", "Artifact count: true vs D3 candidates", dpi))

    # Per-image detection and fit-quality distributions.
    for key, xlabel, title in [
        ("article_precision", "article precision", "Per-image article precision"),
        ("article_recall", "article recall", "Per-image article recall"),
        ("article_f1", "article F1", "Per-image article F1"),
        ("precision", "precision", "Per-image precision"),
        ("recall", "recall", "Per-image recall"),
        ("f1", "F1", "Per-image F1"),
        ("candidate_stage_recall", "candidate-stage recall", "D3 candidate-stage recall"),
        ("r2", "R²", "Fit R²"),
        ("rmse", "RMSE", "Fit RMSE"),
        ("mae", "MAE", "Fit MAE"),
        ("redchi_uncalibrated", "uncalibrated reduced chi-square", "Uncalibrated reduced chi-square"),
    ]:
        vals = arr(key)
        manifest.extend(_hist_plot(outdir, f"{key}_hist", vals, xlabel, title, dpi))

    # Compact detection metrics bar plot.
    metric_vals = {
        "article precision": summary.get("article_precision", summary.get("global_precision")),
        "article recall": summary.get("article_recall", summary.get("global_recall")),
        "article F1": summary.get("article_f1", summary.get("global_f1")),
        "mean precision": summary.get("mean_precision"),
        "mean recall": summary.get("mean_recall"),
        "mean F1": summary.get("mean_f1"),
        "D3 recall": summary.get("mean_candidate_stage_recall"),
    }
    metric_labels = [k for k, v in metric_vals.items() if v is not None and np.isfinite(float(v))]
    if metric_labels:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        vals = [float(metric_vals[k]) for k in metric_labels]
        ax.bar(np.arange(len(metric_labels)), vals)
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=35, ha="right")
        ax.set_ylim(0.0, max(1.0, max(vals) * 1.1))
        ax.set_ylabel("metric")
        ax.set_title("Detection metric summary")
        manifest.append(_save_fig(fig, outdir / "detection_metric_summary.png", dpi))

    # Match diagnostics.
    m_cost = _finite(r.get("cost") for r in match_rows)
    m_cd = _finite(r.get("center_distance") for r in match_rows)
    m_iou = _finite(r.get("annular_iou") for r in match_rows)
    manifest.extend(_hist_plot(outdir, "match_cost_hist", m_cost, "matching cost", "Matched-pair cost", dpi))
    manifest.extend(_hist_plot(outdir, "center_distance_hist", m_cd, "center distance [px]", "Matched-pair center distance", dpi))
    manifest.extend(_hist_plot(outdir, "annular_iou_hist", m_iou, "annular IoU", "Matched-pair annular IoU", dpi))
    if len(m_cd) and len(m_iou):
        fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
        n = min(len(m_cd), len(m_iou))
        ax.scatter(m_cd[:n], m_iou[:n], s=18)
        ax.set_xlabel("center distance [px]")
        ax.set_ylabel("annular IoU")
        ax.set_title("Annular IoU vs center distance")
        manifest.append(_save_fig(fig, outdir / "annular_iou_vs_center_distance.png", dpi))

    # Residual diagnostics.
    if residual_chunks:
        resid = np.concatenate([np.asarray(x, dtype=float).ravel() for x in residual_chunks])
        resid = resid[np.isfinite(resid)]
        if len(resid):
            manifest.extend(_hist_plot(outdir, "global_residual_hist", resid, "fit - data", "Global residuals", dpi, bins=80))
            fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
            try:
                osm, osr = probplot(resid, dist="norm", fit=False)
                ax.scatter(osm, osr, s=2)
                ax.set_xlabel("normal quantiles")
                ax.set_ylabel("residual quantiles")
                ax.set_title("Global residual QQ")
                manifest.append(_save_fig(fig, outdir / "global_residual_qq.png", dpi))
            except Exception:
                plt.close(fig)

    # Parameter-level diagnostics on true-positive pairs only.
    tp_rows = [r for r in param_error_rows if bool(r.get("is_tp", False))]
    for p in PARAM_NAMES:
        rows = [r for r in tp_rows if r.get("param") == p]
        if not rows:
            continue
        t = _finite(r.get("true") for r in rows)
        f = _finite(r.get("fit") for r in rows)
        e = _finite(r.get("error") for r in rows)
        n = min(len(t), len(f), len(e))
        if n == 0:
            continue
        t, f, e = t[:n], f[:n], e[:n]
        fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
        ax.hist(t, bins=25, alpha=0.55, label="true")
        ax.hist(f, bins=25, alpha=0.55, label="fit")
        ax.set_xlabel(p)
        ax.set_ylabel("rings")
        ax.set_title(f"{p}: true and fit distributions")
        ax.legend()
        manifest.append(_save_fig(fig, outdir / f"param_{p}_true_fit_hist.png", dpi))
        manifest.extend(_scatter_identity_plot(outdir, f"param_{p}_true_vs_fit", t, f, f"true {p}", f"fit {p}", f"{p}: true vs fit", dpi))
        manifest.extend(_hist_plot(outdir, f"param_{p}_error_hist", e, f"fit - true {p}", f"{p}: error", dpi))

    # Parameter metric bars.
    metric_rows = summary.get("parameter_metrics_tp", [])
    for metric in ["mae", "rmse", "median_abs_error", "r2"]:
        labels = []
        vals = []
        for row in metric_rows:
            v = row.get(metric)
            if v is not None:
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        labels.append(str(row.get("param")))
                        vals.append(fv)
                except Exception:
                    pass
        if labels:
            fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
            ax.bar(labels, vals)
            ax.set_ylabel(metric)
            ax.set_title(f"Parameter {metric}")
            manifest.append(_save_fig(fig, outdir / f"parameter_{metric}_bar.png", dpi))

    dump_json(stats_dir / "plot_manifest.json", {"plot_count": len(manifest), "plots": manifest})
    return manifest


def run_statistics(cfg: PipelineConfig) -> Dict[str, Any]:
    paths = run_paths(cfg.out)
    image_ids = image_ids_from_manifest(cfg.out)
    per_image: List[Dict[str, Any]] = []
    matches_rows: List[Dict[str, Any]] = []
    param_rows: List[Dict[str, Any]] = []
    residual_chunks: List[np.ndarray] = []
    global_tp = global_fp = global_fn = 0
    article_tp = article_fp = article_fn = 0
    for image_id in image_ids:
        truth = read_json(paths["images"] / image_id / "truth.json")
        final = read_json(paths["d6"] / image_id / "d6_final.json")
        image = np.load(paths["images"] / image_id / "image.npy").astype(float)
        ny, nx = image.shape
        true_rings = truth.get("true_artifacts", [])
        fit_rings = final.get("final_artifacts", [])
        _legacy_matches, tp, fp, fn = _match(true_rings, fit_rings, cfg, nx, ny)
        matches, a_tp, a_fp, a_fn = _match_article(true_rings, fit_rings, cfg, nx, ny)
        global_tp += tp
        global_fp += fp
        global_fn += fn
        article_tp += a_tp
        article_fp += a_fp
        article_fn += a_fn
        legacy_metrics = _counts_to_metrics(tp, fp, fn)
        article_metrics = _counts_to_metrics(a_tp, a_fp, a_fn)
        precision, recall, f1 = legacy_metrics["precision"], legacy_metrics["recall"], legacy_metrics["f1"]
        cands = read_json(paths["d3"] / image_id / "d3_candidates.json").get("candidates", [])
        cand_recall = _candidate_stage_recall(cands, true_rings, cfg, nx, ny)
        X, Y = meshgrid_xy(nx, ny)
        model = smbh_model_image(X, Y, fit_rings, final.get("env", {}))
        resid = (model - image).ravel()
        residual_chunks.append(resid)
        row = {
            "image_id": image_id,
            "split": truth.get("split", "legacy"),
            "fold": truth.get("fold"),
            "family_id": truth.get("family_id", image_id),
            "true_n": len(true_rings),
            "candidate_n": len(cands),
            "fit_n": len(fit_rings),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "article_tp": a_tp,
            "article_fp": a_fp,
            "article_fn": a_fn,
            "article_precision": article_metrics["precision"],
            "article_recall": article_metrics["recall"],
            "article_f1": article_metrics["f1"],
            "candidate_stage_recall": cand_recall,
            "n_error": len(fit_rings) - len(true_rings),
            "r2": final.get("fit_statistics", {}).get("r2"),
            "rmse": final.get("fit_statistics", {}).get("rmse"),
            "mae": final.get("fit_statistics", {}).get("mae"),
            "redchi_uncalibrated": final.get("fit_statistics", {}).get("redchi_uncalibrated"),
        }
        for factor_name, factor_value in truth.get("factors", {}).items():
            row[f"factor_{factor_name}"] = factor_value
        per_image.append(row)
        for m in matches:
            mrow = {"image_id": image_id, **m}
            if m["true_index"] < len(true_rings) and m["fit_index"] < len(fit_rings):
                t = true_rings[m["true_index"]]
                f = fit_rings[m["fit_index"]]
                for p in PARAM_NAMES:
                    err = angle_diff(f[p], t[p]) if p == "phi" else float(f[p] - t[p])
                    mrow[f"true_{p}"] = t[p]
                    mrow[f"fit_{p}"] = f[p]
                    mrow[f"err_{p}"] = err
                    param_rows.append({"image_id": image_id, "param": p, "true": t[p], "fit": f[p], "error": err, "abs_error": abs(err), "is_tp": m["is_tp"], "match_protocol": "article_geometric_iou_v1"})
            matches_rows.append(mrow)
    global_metrics = _counts_to_metrics(global_tp, global_fp, global_fn)
    precision_g, recall_g, f1_g = global_metrics["precision"], global_metrics["recall"], global_metrics["f1"]
    article_global_metrics = _counts_to_metrics(article_tp, article_fp, article_fn)
    param_metrics = _parameter_metrics(param_rows)
    resid_all = np.concatenate(residual_chunks) if residual_chunks else np.array([], dtype=float)
    residual_summary: Dict[str, Any] = {}
    if len(resid_all):
        residual_summary = {
            "global_residual_mean": float(np.mean(resid_all)),
            "global_residual_std": float(np.std(resid_all, ddof=1)) if len(resid_all) > 1 else 0.0,
            "global_residual_skew": float(skew(resid_all)) if len(resid_all) >= 3 else float("nan"),
            "global_residual_excess_kurtosis": float(kurtosis(resid_all, fisher=True)) if len(resid_all) >= 4 else float("nan"),
        }
        try:
            stat, p = normaltest(resid_all) if len(resid_all) >= 8 else (float("nan"), float("nan"))
            residual_summary["global_normaltest_stat"] = float(stat)
            residual_summary["global_normaltest_p"] = float(p)
        except Exception:
            residual_summary["global_normaltest_stat"] = float("nan")
            residual_summary["global_normaltest_p"] = float("nan")
    n_null_images = sum(int(row["true_n"] == 0) for row in per_image)
    n_positive_images = len(per_image) - n_null_images
    null_fp_total = sum(int(row["article_fp"]) for row in per_image if row["true_n"] == 0)
    null_images_with_fp = sum(int(row["article_fp"] > 0) for row in per_image if row["true_n"] == 0)
    ci = _bootstrap_detection_ci(per_image)
    article_ci = _bootstrap_detection_ci(per_image, prefix="article_")
    factor_slices = _factor_slices(per_image)
    eligible_slice_f1 = [
        float(row["f1"]) for row in factor_slices
        if int(row["n_images"]) >= 20 and int(row["tp"]) + int(row["fn"]) > 0 and np.isfinite(float(row["f1"]))
    ]
    eligible_slice_f1_low = [
        float(row["f1_ci95_low"]) for row in factor_slices
        if int(row["n_images"]) >= 20 and int(row["tp"]) + int(row["fn"]) > 0 and np.isfinite(float(row["f1_ci95_low"]))
    ]
    summary = {
        "n_images_total": len(image_ids),
        "n_positive_images": n_positive_images,
        "n_null_images": n_null_images,
        "global_tp": global_tp,
        "global_fp": global_fp,
        "global_fn": global_fn,
        "global_precision": precision_g,
        "global_recall": recall_g,
        "global_f1": f1_g,
        "global_precision_ci95": ci["precision"],
        "global_recall_ci95": ci["recall"],
        "global_f1_ci95": ci["f1"],
        "article_match_protocol": "article_geometric_iou_v1",
        "article_match_thresholds": {
            "max_center_error_fraction": cfg.max_center_error_fraction,
            "max_radius_error_fraction": cfg.max_radius_error_fraction,
            "min_annular_iou": cfg.min_annular_iou,
        },
        "article_tp": article_tp,
        "article_fp": article_fp,
        "article_fn": article_fn,
        "article_precision": article_global_metrics["precision"],
        "article_recall": article_global_metrics["recall"],
        "article_f1": article_global_metrics["f1"],
        "article_precision_ci95": article_ci["precision"],
        "article_recall_ci95": article_ci["recall"],
        "article_f1_ci95": article_ci["f1"],
        "null_fp_total": null_fp_total,
        "null_fppi": float(null_fp_total / n_null_images) if n_null_images else float("nan"),
        "null_fppi_ci95": article_ci["null_fppi"],
        "null_image_fpr": float(null_images_with_fp / n_null_images) if n_null_images else float("nan"),
        "worst_eligible_factor_f1": min(eligible_slice_f1) if eligible_slice_f1 else float("nan"),
        "worst_eligible_factor_f1_ci95_low": min(eligible_slice_f1_low) if eligible_slice_f1_low else float("nan"),
        "mean_true_n": _safe_mean(r["true_n"] for r in per_image),
        "mean_candidate_n": _safe_mean(r["candidate_n"] for r in per_image),
        "mean_fit_n": _safe_mean(r["fit_n"] for r in per_image),
        "mean_n_error": _safe_mean(r["n_error"] for r in per_image),
        "mean_precision": _safe_mean(r["precision"] for r in per_image),
        "mean_recall": _safe_mean(r["recall"] for r in per_image),
        "mean_f1": _safe_mean(r["f1"] for r in per_image),
        "mean_article_precision": _safe_mean(r["article_precision"] for r in per_image),
        "mean_article_recall": _safe_mean(r["article_recall"] for r in per_image),
        "mean_article_f1": _safe_mean(r["article_f1"] for r in per_image),
        "mean_candidate_stage_recall": _safe_mean(r["candidate_stage_recall"] for r in per_image),
        "mean_r2": _safe_mean(r["r2"] for r in per_image),
        "median_rmse": _safe_median(r["rmse"] for r in per_image),
        "median_mae": _safe_median(r["mae"] for r in per_image),
        "parameter_metrics_tp": param_metrics,
        **residual_summary,
    }
    _write_csv(paths["stats"] / "per_image_metrics.csv", per_image)
    _write_csv(paths["stats"] / "matched_artifacts.csv", matches_rows)
    _write_csv(paths["stats"] / "parameter_errors_long.csv", param_rows)
    _write_csv(paths["stats"] / "parameter_metrics.csv", param_metrics)
    _write_csv(paths["stats"] / "factor_metrics.csv", factor_slices)
    plots = (
        make_aggregate_plots(
            paths["stats"],
            per_image,
            matches_rows,
            param_rows,
            residual_chunks,
            summary,
            dpi=cfg.viz_dpi,
        )
        if cfg.make_plots
        else []
    )
    if not cfg.make_plots:
        dump_json(paths["stats"] / "plot_manifest.json", {
            "plot_count": 0,
            "plots": [],
            "disabled_by_config": True,
        })
    summary["aggregate_plot_count"] = len(plots)
    dump_json(paths["stats"] / "summary.json", summary)
    return summary
