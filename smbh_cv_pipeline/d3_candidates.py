from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .config import PipelineConfig
from .io_utils import dump_json, run_paths, measured, write_worker_metric


def _nms(rows: np.ndarray, score_threshold: float, cfg: PipelineConfig) -> List[Dict[str, float]]:
    if len(rows) == 0:
        return []
    rows = rows[np.argsort(rows[:, 3])[::-1]]
    selected: List[Dict[str, float]] = []
    max_candidates = int(min(cfg.d3_max_candidates, max(1, 3 * cfg.d3_expected_max_rings)))
    for x, y, r, score in rows:
        if float(score) < score_threshold:
            continue
        keep = True
        for c in selected:
            if np.hypot(float(x) - c["x0"], float(y) - c["y0"]) <= cfg.d3_nms_dx and abs(float(r) - c["R"]) <= cfg.d3_nms_dr:
                keep = False
                break
        if keep:
            selected.append({"x0": float(x), "y0": float(y), "R": float(r), "score": float(score)})
            if len(selected) >= max_candidates:
                break
    return selected


def _threshold(rows: np.ndarray, cfg: PipelineConfig) -> float:
    if len(rows) == 0:
        return float("inf")
    scores = rows[:, 3].astype(float)
    if cfg.d3_threshold_method == "relative":
        return float(cfg.d3_relative_threshold * np.max(scores))
    if cfg.d3_threshold_method == "quantile":
        return float(np.quantile(scores, cfg.d3_quantile))
    if cfg.d3_threshold_method == "bootstrap":
        rng = np.random.default_rng(1009)
        vals = []
        for _ in range(max(1, cfg.d3_bootstrap_repeats)):
            sample = rng.choice(scores, size=min(len(scores), max(50, len(scores) // 4)), replace=True)
            vals.append(float(np.max(sample)))
        return float(np.quantile(vals, cfg.d3_bootstrap_quantile))
    raise ValueError(f"Unknown D3 threshold method: {cfg.d3_threshold_method}")


def _ph_safe_proxy(rows: np.ndarray, cfg: PipelineConfig, nx: int, ny: int) -> List[Dict[str, float]]:
    if len(rows) == 0:
        return []
    top_k = min(int(cfg.d3_top_k_for_ph), len(rows))
    top = rows[np.argsort(rows[:, 3])[::-1]][:top_k]
    est_dense_gb = (nx * ny * max(1, int(cfg.d2_r_max - cfg.d2_r_min + 1)) * 8) / (1024 ** 3)
    threshold = _threshold(top, cfg)
    out = _nms(top, threshold, cfg)
    for c in out:
        c["method_note"] = "ph_safe_proxy_sparse_topk_no_dense_complex" if est_dense_gb > cfg.d3_ph_memory_limit_gb else "ph_safe_proxy_dense_allowed_but_nms_used"
        c["estimated_dense_gb"] = float(est_dense_gb)
    return out


def _find(parent: np.ndarray, i: int) -> int:
    root = i
    while parent[root] != root:
        root = int(parent[root])
    while parent[i] != i:
        j = int(parent[i])
        parent[i] = root
        i = j
    return root


def _union_find_sparse_ph(rows: np.ndarray, cfg: PipelineConfig) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """0D superlevel persistence on occupied sparse accumulator cells only.

    Each sparse accumulator bin is a vertex. Two vertices are adjacent if their
    integer parameter-bin coordinates differ by at most one in x, y and R. Vertices
    enter from high score to low score. When two active components become adjacent,
    the component with lower birth score dies at the current score.
    """
    if len(rows) == 0:
        return [], {"n_vertices": 0, "n_pairs": 0}
    top_k = min(int(cfg.d3_top_k_for_ph), len(rows)) if cfg.d3_top_k_for_ph > 0 else len(rows)
    rows = rows[np.argsort(rows[:, 3])[::-1]][:top_k].astype(np.float64, copy=False)

    xb = np.rint(rows[:, 0] / max(cfg.d2_center_bin_px, 1e-9)).astype(np.int32)
    yb = np.rint(rows[:, 1] / max(cfg.d2_center_bin_px, 1e-9)).astype(np.int32)
    rb = np.rint(rows[:, 2] / max(cfg.d2_radius_bin_px, 1e-9)).astype(np.int32)
    coords = list(zip(xb.tolist(), yb.tolist(), rb.tolist()))
    key_to_idx = {c: i for i, c in enumerate(coords)}

    n = len(rows)
    parent = np.arange(n, dtype=np.int32)
    active = np.zeros(n, dtype=bool)
    birth = rows[:, 3].astype(np.float64).copy()
    peak_idx = np.arange(n, dtype=np.int32)
    order = np.argsort(rows[:, 3])[::-1]

    pairs: List[Dict[str, float]] = []
    neigh = [(dx, dy, dr) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dr in (-1, 0, 1) if not (dx == dy == dr == 0)]
    for idx in order:
        idx = int(idx)
        active[idx] = True
        x, y, r = coords[idx]
        current_score = float(rows[idx, 3])
        for dx, dy, dr in neigh:
            j = key_to_idx.get((x + dx, y + dy, r + dr))
            if j is None or not active[j]:
                continue
            ri = _find(parent, idx)
            rj = _find(parent, int(j))
            if ri == rj:
                continue
            # The component born at the larger score survives. In a tie, keep the one
            # whose peak has the larger accumulator score and then smaller index.
            if (birth[ri] > birth[rj]) or (birth[ri] == birth[rj] and int(peak_idx[ri]) <= int(peak_idx[rj])):
                survivor, loser = ri, rj
            else:
                survivor, loser = rj, ri
            pidx = int(peak_idx[loser])
            persistence = float(max(0.0, birth[loser] - current_score))
            pairs.append({
                "x0": float(rows[pidx, 0]),
                "y0": float(rows[pidx, 1]),
                "R": float(rows[pidx, 2]),
                "score": persistence,
                "persistence": persistence,
                "birth": float(birth[loser]),
                "death": current_score,
                "source_score": float(rows[pidx, 3]),
            })
            parent[loser] = survivor
            if birth[loser] > birth[survivor]:
                birth[survivor] = birth[loser]
                peak_idx[survivor] = peak_idx[loser]
    # Add essential components. Their death is set to the minimum score in the sparse set.
    min_score = float(np.min(rows[:, 3]))
    roots = sorted({_find(parent, int(i)) for i in range(n) if active[i]})
    for root in roots:
        pidx = int(peak_idx[root])
        persistence = float(max(0.0, birth[root] - min_score))
        pairs.append({
            "x0": float(rows[pidx, 0]),
            "y0": float(rows[pidx, 1]),
            "R": float(rows[pidx, 2]),
            "score": persistence,
            "persistence": persistence,
            "birth": float(birth[root]),
            "death": min_score,
            "source_score": float(rows[pidx, 3]),
            "essential": True,
        })
    rows_for_nms = np.array([[p["x0"], p["y0"], p["R"], p["score"]] for p in pairs], dtype=np.float32)
    threshold = _threshold(rows_for_nms, cfg) if len(rows_for_nms) else float("inf")
    selected = _nms(rows_for_nms, threshold, cfg)
    # Copy persistence details into selected candidates by matching coordinates.
    details = {(round(p["x0"], 6), round(p["y0"], 6), round(p["R"], 6), round(p["score"], 6)): p for p in pairs}
    for c in selected:
        k = (round(c["x0"], 6), round(c["y0"], 6), round(c["R"], 6), round(c["score"], 6))
        if k in details:
            c.update(details[k])
        c["method_note"] = "ph_union_find_sparse_h0"
    meta = {"n_vertices": int(n), "n_pairs": int(len(pairs)), "score_threshold": float(threshold), "top_k": int(top_k)}
    return selected, meta


def _ph_cripser_dense(rows: np.ndarray, cfg: PipelineConfig, nx: int, ny: int) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    try:
        import cripser  # type: ignore
    except Exception as exc:
        raise RuntimeError("d3_method='ph_cripser_dense' requires optional package 'cripser'. Install it separately, or use ph_union_find_sparse.") from exc
    r_bins = int(round(cfg.d2_r_max / max(cfg.d2_radius_bin_px, 1e-9))) + 1
    est_dense_gb = (nx * ny * max(1, r_bins) * 4) / (1024 ** 3)
    if est_dense_gb > cfg.d3_ph_memory_limit_gb:
        raise MemoryError(f"Dense Cripser backend would allocate about {est_dense_gb:.2f} GB, above d3_ph_memory_limit_gb={cfg.d3_ph_memory_limit_gb}.")
    vol = np.zeros((nx, ny, r_bins), dtype=np.float32)
    xb = np.rint(rows[:, 0] / max(cfg.d2_center_bin_px, 1e-9)).astype(int)
    yb = np.rint(rows[:, 1] / max(cfg.d2_center_bin_px, 1e-9)).astype(int)
    rb = np.rint(rows[:, 2] / max(cfg.d2_radius_bin_px, 1e-9)).astype(int)
    ok = (xb >= 0) & (xb < nx) & (yb >= 0) & (yb < ny) & (rb >= 0) & (rb < r_bins)
    vol[xb[ok], yb[ok], rb[ok]] = rows[ok, 3].astype(np.float32)
    # Cripser computes cubical persistence on dense arrays. API variants differ;
    # this branch is intentionally guarded and documented as optional.
    pd = cripser.computePH(-vol, maxdim=0)  # type: ignore[attr-defined]
    _ = pd  # Coordinate extraction is not guaranteed by all cripser builds; use sparse PH for candidates.
    selected, meta = _union_find_sparse_ph(rows, cfg)
    meta.update({"backend": "cripser_dense_computed_diagram_then_sparse_locations", "estimated_dense_gb": float(est_dense_gb)})
    return selected, meta


def process_one(cfg_dict: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    cfg = PipelineConfig(**cfg_dict)
    paths = run_paths(cfg.out)
    metric: Dict[str, Any] = {}
    with measured("D3", image_id) as metric:
        rows = np.load(paths["d2"] / image_id / "d2_sparse_accumulator.npz")["accumulator"].astype(np.float32)
        image = np.load(paths["images"] / image_id / "image.npy")
        ny, nx = image.shape
        extra_meta: Dict[str, Any] = {}
        if cfg.d3_method in ("peak_nms", "bootstrap_nms"):
            effective_cfg = cfg
            if cfg.d3_method == "bootstrap_nms":
                effective_cfg.d3_threshold_method = "bootstrap"
            threshold = _threshold(rows, effective_cfg)
            candidates = _nms(rows, threshold, effective_cfg)
        elif cfg.d3_method == "ph_safe_proxy":
            threshold = _threshold(rows, cfg)
            candidates = _ph_safe_proxy(rows, cfg, nx=nx, ny=ny)
        elif cfg.d3_method == "ph_union_find_sparse":
            candidates, extra_meta = _union_find_sparse_ph(rows, cfg)
            threshold = float(extra_meta.get("score_threshold", float("nan")))
        elif cfg.d3_method == "ph_cripser_dense":
            candidates, extra_meta = _ph_cripser_dense(rows, cfg, nx=nx, ny=ny)
            threshold = float(extra_meta.get("score_threshold", float("nan")))
        else:
            raise ValueError(f"Unknown D3 method: {cfg.d3_method}")
        if candidates:
            max_score = max(c["score"] for c in candidates)
            for c in candidates:
                c["confidence"] = float(c["score"] / max_score) if max_score > 0 else 0.0
        out_dir = paths["d3"] / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_json(out_dir / "d3_candidates.json", {
            "image_id": image_id,
            "method": cfg.d3_method,
            "threshold_method": cfg.d3_threshold_method,
            "score_threshold": float(threshold),
            "n_input_bins": int(len(rows)),
            "n_candidates": int(len(candidates)),
            "candidates": candidates,
            "extra_meta": extra_meta,
        })
        metric.update({"input_bins": int(len(rows)), "n_candidates": int(len(candidates)), "score_threshold": float(threshold)})
    write_worker_metric(cfg.out, metric)
    return {"image_id": image_id, "status": "ok", "n_candidates": metric.get("n_candidates")}
