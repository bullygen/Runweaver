from __future__ import annotations

"""
Visualization stage for the modular SMBH CV pipeline.

The plotting layout follows the original project graph-handling convention
(__USE_graphHandling.py): every diagnostic compares the source image with the
intermediate result of a concrete stage and writes a deterministic PNG artifact.
This module is deliberately separated from D1-D6 so that heavy stages can free
memory before figures are produced.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .config import PipelineConfig
from .io_utils import dump_json, read_json, run_paths, measured, write_worker_metric
from .models import meshgrid_xy, smbh_model_image


def _viz_dir(paths: Mapping[str, Path], image_id: str) -> Path:
    out = paths["viz"] / image_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return read_json(path)


def _safe_load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def _safe_load_npz_array(path: Path, key: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        return np.load(path)[key]
    except Exception:
        return None


def _imshow(ax: plt.Axes, data: np.ndarray, title: str, *, cmap: str = "gray", add_colorbar: bool = True, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    im = ax.imshow(data, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(im, cax=cax, orientation="vertical")


def _overlay_rings(ax: plt.Axes, rings: Sequence[Mapping[str, Any]], *, edgecolor: str, label: str, linewidth: float = 1.4, linestyle: str = "-") -> None:
    added = False
    for r in rings:
        if not all(k in r for k in ("x0", "y0", "R")):
            continue
        circ = Circle((float(r["x0"]), float(r["y0"])), float(r["R"]), fill=False, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, alpha=0.9)
        ax.add_patch(circ)
        added = True
        if "sigma" in r:
            sigma = max(float(r.get("sigma", 0.0)), 0.0)
            if sigma > 0:
                ax.add_patch(Circle((float(r["x0"]), float(r["y0"])), max(0.1, float(r["R"]) - sigma), fill=False, edgecolor=edgecolor, linewidth=0.7, linestyle=":", alpha=0.55))
                ax.add_patch(Circle((float(r["x0"]), float(r["y0"])), float(r["R"]) + sigma, fill=False, edgecolor=edgecolor, linewidth=0.7, linestyle=":", alpha=0.55))
    if added:
        # Empty proxy for a clean legend entry.
        ax.plot([], [], color=edgecolor, linestyle=linestyle, linewidth=linewidth, label=label)


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _truth(paths: Mapping[str, Path], image_id: str) -> Dict[str, Any]:
    return read_json(paths["images"] / image_id / "truth.json")


def _image(paths: Mapping[str, Path], image_id: str) -> np.ndarray:
    return np.load(paths["images"] / image_id / "image.npy")


def _model_from_rings(image: np.ndarray, rings: Sequence[Mapping[str, Any]], env: Optional[Mapping[str, Any]]) -> np.ndarray:
    ny, nx = image.shape
    X, Y = meshgrid_xy(nx, ny)
    return smbh_model_image(X, Y, [dict(r) for r in rings], dict(env or {}))


def plot_generation(paths: Mapping[str, Path], image_id: str, dpi: int) -> Optional[str]:
    truth = _truth(paths, image_id)
    image_dir = paths["images"] / image_id
    image = _image(paths, image_id)
    clean = _safe_load_npy(image_dir / "clean_image.npy")
    pre = _safe_load_npy(image_dir / "truth_clean_pre_degradation.npy")
    noise = _safe_load_npy(image_dir / "noise.npy")
    rings = truth.get("true_artifacts", [])

    fig, ax = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    vmax = float(np.max(image)) if image.size else None
    vmin = float(np.min(image)) if image.size else None
    _imshow(ax[0, 0], image, "Generated image", vmin=vmin, vmax=vmax)
    _overlay_rings(ax[0, 0], rings, edgecolor="tab:green", label="true rings")
    ax[0, 0].legend(loc="upper right", fontsize=8)
    if clean is not None:
        _imshow(ax[0, 1], clean, "Clean image after degradation", vmin=vmin, vmax=vmax)
        _overlay_rings(ax[0, 1], rings, edgecolor="tab:green", label="true rings")
    else:
        ax[0, 1].axis("off")
    if pre is not None:
        _imshow(ax[1, 0], pre, "Clean image before degradation")
    else:
        ax[1, 0].axis("off")
    if noise is not None:
        _imshow(ax[1, 1], noise, "Noise map")
    else:
        ax[1, 1].axis("off")
    fig.suptitle(f"Generation diagnostics: {image_id}")
    out = _viz_dir(paths, image_id) / "00_generation.png"
    _save(fig, out, dpi)
    return str(out)


def plot_d1(paths: Mapping[str, Path], image_id: str, dpi: int) -> Optional[str]:
    d1_npz = paths["d1"] / image_id / "d1_edges.npz"
    if not d1_npz.exists():
        return None
    image = _image(paths, image_id)
    data = np.load(d1_npz)
    binary = data["binary"].astype(bool)
    weights = data["weights"].astype(float)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    _imshow(ax[0, 0], image, "Original image")
    _imshow(ax[0, 1], weights, "D1 weighted edge response", cmap="magma")
    _imshow(ax[1, 0], binary.astype(float), "D1 binary/skeleton map")
    _imshow(ax[1, 1], image, "D1 edges over image")
    yy, xx = np.nonzero(binary)
    if len(xx):
        ax[1, 1].scatter(xx, yy, s=1, alpha=0.55, label="edge pixels")
        ax[1, 1].legend(loc="upper right", fontsize=8)
    meta = _safe_load_json(paths["d1"] / image_id / "d1_meta.json") or {}
    fig.suptitle(f"D1 preprocessing: {image_id}; edge pixels={meta.get('morphology', {}).get('after_skeleton_nonzero', len(xx))}")
    out = _viz_dir(paths, image_id) / "01_D1_preprocess.png"
    _save(fig, out, dpi)
    return str(out)


def _top_rows(rows: np.ndarray, limit: int) -> np.ndarray:
    if len(rows) == 0:
        return rows.reshape(0, 4)
    order = np.argsort(rows[:, 3])[::-1]
    return rows[order[: min(limit, len(rows))]]


def _projection_maps_from_sparse(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build continuous 2D projections of sparse RHT votes.

    The maps are sums over the third coordinate, matching the original project
    graph-handling idea: XY=sum_R, XR=sum_Y, YR=sum_X.
    """
    if len(rows) == 0:
        z = np.zeros((1, 1), dtype=np.float32)
        return z, z, z, {"empty": True}
    x = np.rint(rows[:, 0]).astype(int)
    y = np.rint(rows[:, 1]).astype(int)
    r = np.rint(rows[:, 2]).astype(int)
    v = rows[:, 3].astype(float)
    nx = int(max(x.max() + 1, 1))
    ny = int(max(y.max() + 1, 1))
    nr = int(max(r.max() + 1, 1))
    xy = np.zeros((ny, nx), dtype=np.float32)
    xr = np.zeros((nr, nx), dtype=np.float32)
    yr = np.zeros((nr, ny), dtype=np.float32)
    np.add.at(xy, (y, x), v)
    np.add.at(xr, (r, x), v)
    np.add.at(yr, (r, y), v)
    return xy, xr, yr, {"empty": False, "nx": nx, "ny": ny, "nr": nr}


def _local_maxima_sparse(rows: np.ndarray, max_points: int = 50, dx: float = 5.0, dr: float = 4.0) -> np.ndarray:
    if len(rows) == 0:
        return rows.reshape(0, 4)
    rows = rows[np.argsort(rows[:, 3])[::-1]]
    selected = []
    for row in rows:
        x, y, r, _ = row
        keep = True
        for s in selected:
            if np.hypot(float(x) - float(s[0]), float(y) - float(s[1])) <= dx and abs(float(r) - float(s[2])) <= dr:
                keep = False
                break
        if keep:
            selected.append(row)
            if len(selected) >= max_points:
                break
    return np.asarray(selected, dtype=float).reshape(-1, 4)


def plot_d2(paths: Mapping[str, Path], image_id: str, dpi: int, max_points: int) -> Optional[str]:
    acc_path = paths["d2"] / image_id / "d2_sparse_accumulator.npz"
    if not acc_path.exists():
        return None
    rows = np.load(acc_path)["accumulator"].astype(float)
    xy, xr, yr, _ = _projection_maps_from_sparse(rows)
    maxima = _local_maxima_sparse(rows, max_points=min(80, max_points), dx=5.0, dr=4.0)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(f"D2 Randomized Hough accumulator projections: {image_id}; sparse bins={len(rows)}")

    im = ax[0, 0].imshow(xy, origin="lower", interpolation="nearest", cmap="magma")
    ax[0, 0].set_title("XY heatmap: sum over R")
    ax[0, 0].set_xlabel("x0, px")
    ax[0, 0].set_ylabel("y0, px")
    '''
    if len(maxima):
        ax[0, 0].scatter(maxima[:, 0], maxima[:, 1], s=24, marker="x", c="cyan", linewidths=0.9, label="local maxima")
        ax[0, 0].legend(loc="upper right", fontsize=8)
    '''
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.02, label="votes")

    im = ax[0, 1].imshow(xr, origin="lower", interpolation="nearest", cmap="magma", aspect="auto")
    ax[0, 1].set_title("XR heatmap: sum over Y")
    ax[0, 1].set_xlabel("x0, px")
    ax[0, 1].set_ylabel("R, px")
    '''
    if len(maxima):
        ax[0, 1].scatter(maxima[:, 0], maxima[:, 2], s=24, marker="x", c="cyan", linewidths=0.9)
    '''
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.02, label="votes")

    im = ax[0, 2].imshow(yr, origin="lower", interpolation="nearest", cmap="magma", aspect="auto")
    ax[0, 2].set_title("YR heatmap: sum over X")
    ax[0, 2].set_xlabel("y0, px")
    ax[0, 2].set_ylabel("R, px")
    '''
    if len(maxima):
        ax[0, 2].scatter(maxima[:, 1], maxima[:, 2], s=24, marker="x", c="cyan", linewidths=0.9)
    '''
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046, pad=0.02, label="votes")

    if len(rows) == 0:
        for a in ax[1, :]:
            a.text(0.5, 0.5, "empty accumulator", ha="center", va="center")
            a.axis("off")
    else:
        ax[1, 0].hist(rows[:, 3], bins=60)
        ax[1, 0].set_title("Vote-score distribution")
        ax[1, 0].set_xlabel("votes")
        ax[1, 0].set_ylabel("bins")
        ax[1, 1].hist(rows[:, 2], bins=60)
        ax[1, 1].set_title("Radius-bin distribution")
        ax[1, 1].set_xlabel("R, px")
        ax[1, 1].set_ylabel("bins")
        top = _top_rows(rows, max_points)
        ax[1, 2].plot(np.arange(1, len(top) + 1), top[:, 3])
        ax[1, 2].set_title(f"Top {len(top)} vote scores")
        ax[1, 2].set_xlabel("rank")
        ax[1, 2].set_ylabel("votes")

    out = _viz_dir(paths, image_id) / "02_D2_hough.png"
    _save(fig, out, dpi)
    return str(out)

def plot_d3(paths: Mapping[str, Path], image_id: str, dpi: int) -> Optional[str]:
    data = _safe_load_json(paths["d3"] / image_id / "d3_candidates.json")
    if data is None:
        return None
    image = _image(paths, image_id)
    candidates = data.get("candidates", [])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    _imshow(ax[0], image, "Candidates over image")
    _overlay_rings(ax[0], candidates, edgecolor="tab:orange", label="D3 candidates")
    if candidates:
        ax[0].legend(loc="upper right", fontsize=8)
        scores = [float(c.get("score", 0.0)) for c in candidates]
        ax[1].bar(np.arange(len(scores)), scores)
        ax[1].set_xlabel("candidate index")
        ax[1].set_ylabel("score")
        ax[1].set_title("Candidate scores")
        ax[2].scatter([c["R"] for c in candidates], scores)
        ax[2].set_xlabel("R, px")
        ax[2].set_ylabel("score")
        ax[2].set_title("Score vs radius")
    else:
        ax[1].text(0.5, 0.5, "no candidates", ha="center", va="center")
        ax[2].text(0.5, 0.5, "no candidates", ha="center", va="center")
    fig.suptitle(f"D3 candidates: {image_id}; threshold={data.get('score_threshold')}")
    out = _viz_dir(paths, image_id) / "03_D3_candidates.png"
    _save(fig, out, dpi)
    return str(out)


def plot_d4(paths: Mapping[str, Path], image_id: str, dpi: int) -> Optional[str]:
    data = _safe_load_json(paths["d4"] / image_id / "d4_initial_params.json")
    if data is None:
        return None
    image = _image(paths, image_id)
    rings = data.get("initial_artifacts", [])
    env = data.get("initial_env", {})
    model = _model_from_rings(image, rings, env)
    resid = model - image
    fig, ax = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    _imshow(ax[0, 0], image, "D4 initial rings over image")
    _overlay_rings(ax[0, 0], rings, edgecolor="tab:orange", label="D4 init rings")
    if rings:
        ax[0, 0].legend(loc="upper right", fontsize=8)
    _imshow(ax[0, 1], model, "Initial model image")
    _imshow(ax[1, 0], resid, "Initial model - image", cmap="coolwarm")
    if rings:
        ax[1, 1].bar(np.arange(len(rings)), [float(r.get("source_confidence", 0.0)) for r in rings])
        ax[1, 1].set_title("Source confidence from D3")
        ax[1, 1].set_xlabel("ring index")
        ax[1, 1].set_ylabel("confidence")
    else:
        ax[1, 1].text(0.5, 0.5, "no initial rings", ha="center", va="center")
    fig.suptitle(f"D4 initialization: {image_id}; initial_n={len(rings)}")
    out = _viz_dir(paths, image_id) / "04_D4_initialization.png"
    _save(fig, out, dpi)
    return str(out)


def plot_d5(paths: Mapping[str, Path], image_id: str, dpi: int) -> Optional[str]:
    data = _safe_load_json(paths["d5"] / image_id / "d5_fit.json")
    if data is None:
        return None
    image = _image(paths, image_id)
    arrays_path = Path(data.get("arrays_path", paths["d5"] / image_id / "d5_fit_arrays.npz"))
    model = _safe_load_npz_array(arrays_path, "model")
    residual = _safe_load_npz_array(arrays_path, "residual")
    weights = _safe_load_npz_array(arrays_path, "weights")
    if model is None:
        model = _model_from_rings(image, data.get("artifacts", []), data.get("env", {}))
    if residual is None:
        residual = model - image
    fig, ax = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    _imshow(ax[0, 0], image, "D5 fitted rings over image")
    _overlay_rings(ax[0, 0], data.get("artifacts", []), edgecolor="tab:red", label="D5 fitted rings")
    if data.get("artifacts"):
        ax[0, 0].legend(loc="upper right", fontsize=8)
    _imshow(ax[0, 1], model, "D5 fitted model")
    _imshow(ax[1, 0], residual, "D5 residual: model - image", cmap="coolwarm")
    if weights is not None:
        _imshow(ax[1, 1], weights, "D5 residual weights", cmap="viridis")
    else:
        ax[1, 1].text(0.5, 0.5, "weights not saved", ha="center", va="center")
    stats = data.get("fit_statistics", {})
    fig.suptitle(f"D5 fit: {image_id}; fit_n={len(data.get('artifacts', []))}; R²={stats.get('r2')}")
    out = _viz_dir(paths, image_id) / "05_D5_fit.png"
    _save(fig, out, dpi)
    return str(out)


def plot_d6(paths: Mapping[str, Path], image_id: str, dpi: int) -> Optional[str]:
    data = _safe_load_json(paths["d6"] / image_id / "d6_final.json")
    if data is None:
        return None
    truth = _truth(paths, image_id)
    image = _image(paths, image_id)
    arrays = paths["d6"] / image_id / "d6_final_arrays.npz"
    model = _safe_load_npz_array(arrays, "model")
    residual = _safe_load_npz_array(arrays, "residual")
    if model is None:
        model = _model_from_rings(image, data.get("final_artifacts", []), data.get("env", {}))
    if residual is None:
        residual = model - image
    fig, ax = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    _imshow(ax[0, 0], image, "Final result over image")
    _overlay_rings(ax[0, 0], truth.get("true_artifacts", []), edgecolor="tab:green", label="truth", linestyle="--")
    _overlay_rings(ax[0, 0], data.get("final_artifacts", []), edgecolor="tab:red", label="final")
    ax[0, 0].legend(loc="upper right", fontsize=8)
    _imshow(ax[0, 1], model, "D6 final model")
    _imshow(ax[1, 0], residual, "D6 residual: model - image", cmap="coolwarm")
    history = data.get("selection_history", [])
    if history:
        reasons: Dict[str, int] = {}
        for h in history:
            reasons[str(h.get("reason", "unknown"))] = reasons.get(str(h.get("reason", "unknown")), 0) + 1
        ax[1, 1].bar(list(reasons.keys()), list(reasons.values()))
        ax[1, 1].tick_params(axis="x", labelrotation=30)
        ax[1, 1].set_title("Pruning/merge decisions")
        ax[1, 1].set_ylabel("count")
    else:
        ax[1, 1].text(0.5, 0.5, "no pruning history", ha="center", va="center")
        ax[1, 1].set_title("Pruning/merge decisions")
    fig.suptitle(f"Final D6 result: {image_id}; true={truth.get('true_n')}; final={data.get('final_n')}")
    out = _viz_dir(paths, image_id) / "06_D6_final.png"
    _save(fig, out, dpi)
    return str(out)


def plot_pipeline_overview(paths: Mapping[str, Path], image_id: str, dpi: int) -> Optional[str]:
    image = _image(paths, image_id)
    truth = _truth(paths, image_id)
    d1 = _safe_load_npz_array(paths["d1"] / image_id / "d1_edges.npz", "binary")
    d3 = _safe_load_json(paths["d3"] / image_id / "d3_candidates.json") or {}
    d4 = _safe_load_json(paths["d4"] / image_id / "d4_initial_params.json") or {}
    d5 = _safe_load_json(paths["d5"] / image_id / "d5_fit.json") or {}
    d6 = _safe_load_json(paths["d6"] / image_id / "d6_final.json") or {}
    d6_model = _safe_load_npz_array(paths["d6"] / image_id / "d6_final_arrays.npz", "model")
    d6_resid = _safe_load_npz_array(paths["d6"] / image_id / "d6_final_arrays.npz", "residual")

    fig, ax = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    _imshow(ax[0, 0], image, "Input image")
    _overlay_rings(ax[0, 0], truth.get("true_artifacts", []), edgecolor="tab:green", label="truth", linestyle="--")
    ax[0, 0].legend(loc="upper right", fontsize=8)
    if d1 is not None:
        _imshow(ax[0, 1], d1.astype(float), "D1 binary edges")
    else:
        ax[0, 1].axis("off")
    _imshow(ax[0, 2], image, "D3 candidates")
    _overlay_rings(ax[0, 2], d3.get("candidates", []), edgecolor="tab:orange", label="candidates")
    if d3.get("candidates"):
        ax[0, 2].legend(loc="upper right", fontsize=8)
    _imshow(ax[0, 3], image, "D4 initial rings")
    _overlay_rings(ax[0, 3], d4.get("initial_artifacts", []), edgecolor="tab:orange", label="initial")
    if d4.get("initial_artifacts"):
        ax[0, 3].legend(loc="upper right", fontsize=8)
    _imshow(ax[1, 0], image, "D5 fitted rings")
    _overlay_rings(ax[1, 0], d5.get("artifacts", []), edgecolor="tab:red", label="fit")
    if d5.get("artifacts"):
        ax[1, 0].legend(loc="upper right", fontsize=8)
    _imshow(ax[1, 1], image, "D6 final rings")
    _overlay_rings(ax[1, 1], truth.get("true_artifacts", []), edgecolor="tab:green", label="truth", linestyle="--")
    _overlay_rings(ax[1, 1], d6.get("final_artifacts", []), edgecolor="tab:red", label="final")
    ax[1, 1].legend(loc="upper right", fontsize=8)
    if d6_model is not None:
        _imshow(ax[1, 2], d6_model, "Final model")
    else:
        ax[1, 2].axis("off")
    if d6_resid is not None:
        _imshow(ax[1, 3], d6_resid, "Final residual", cmap="coolwarm")
    else:
        ax[1, 3].axis("off")
    fig.suptitle(f"Pipeline overview: {image_id}")
    out = _viz_dir(paths, image_id) / "pipeline_overview.png"
    _save(fig, out, dpi)
    return str(out)


def process_one(cfg_dict: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    cfg = PipelineConfig(**cfg_dict)
    paths = run_paths(cfg.out)
    metric: Dict[str, Any] = {}
    with measured("VIZ", image_id) as metric:
        produced: Dict[str, Optional[str]] = {}
        produced["generation"] = plot_generation(paths, image_id, cfg.viz_dpi)
        produced["d1"] = plot_d1(paths, image_id, cfg.viz_dpi)
        produced["d2"] = plot_d2(paths, image_id, cfg.viz_dpi, cfg.viz_max_accumulator_points)
        produced["d3"] = plot_d3(paths, image_id, cfg.viz_dpi)
        produced["d4"] = plot_d4(paths, image_id, cfg.viz_dpi)
        produced["d5"] = plot_d5(paths, image_id, cfg.viz_dpi)
        produced["d6"] = plot_d6(paths, image_id, cfg.viz_dpi)
        produced["overview"] = plot_pipeline_overview(paths, image_id, cfg.viz_dpi)
        produced = {k: v for k, v in produced.items() if v is not None}
        dump_json(paths["viz"] / image_id / "visualization_manifest.json", {"image_id": image_id, "figures": produced})
        metric.update({"n_figures": len(produced)})
    write_worker_metric(cfg.out, metric)
    return {"image_id": image_id, "status": "ok", "n_figures": metric.get("n_figures")}


def run_visualization(cfg: PipelineConfig, image_ids: Sequence[str]) -> Dict[str, Any]:
    paths = run_paths(cfg.out)
    manifests = []
    for image_id in image_ids:
        m = _safe_load_json(paths["viz"] / image_id / "visualization_manifest.json")
        if m is not None:
            manifests.append(m)
    out = {"n_images": len(image_ids), "n_visualized": len(manifests), "items": manifests}
    dump_json(paths["viz"] / "visualization_summary.json", out)
    return out
