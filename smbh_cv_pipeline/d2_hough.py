from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import PipelineConfig
from .io_utils import dump_json, run_paths, measured, write_worker_metric


def _circle_from_three(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float] | None:
    # p = [y, x]
    x1, y1 = float(p1[1]), float(p1[0])
    x2, y2 = float(p2[1]), float(p2[0])
    x3, y3 = float(p3[1]), float(p3[0])
    a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    if abs(a) < 1e-9:
        return None
    b = ((x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1))
    c = ((x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2))
    cx = -b / (2.0 * a)
    cy = -c / (2.0 * a)
    r = math.hypot(x1 - cx, y1 - cy)
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r)):
        return None
    return cx, cy, r


def _triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    x1, y1 = float(p1[1]), float(p1[0])
    x2, y2 = float(p2[1]), float(p2[0])
    x3, y3 = float(p3[1]), float(p3[0])
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0


def _make_sectors(points: np.ndarray, nx: int, ny: int, n_sectors: int) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    cx, cy = (nx - 1.0) / 2.0, (ny - 1.0) / 2.0
    angles = np.arctan2(points[:, 0] - cy, points[:, 1] - cx)
    idx = np.floor(((angles + np.pi) / (2 * np.pi)) * n_sectors).astype(int)
    idx = np.clip(idx, 0, n_sectors - 1)
    return [np.where(idx == k)[0] for k in range(n_sectors)]


def _sample_triplet(rng: np.random.Generator, points: np.ndarray, sectors: List[np.ndarray], cfg: PipelineConfig) -> Tuple[int, int, int] | None:
    n = len(points)
    if n < 3:
        return None
    if cfg.d2_method == "sparse_rht":
        ids = rng.choice(n, size=3, replace=False)
        return int(ids[0]), int(ids[1]), int(ids[2])
    if cfg.d2_method == "sparse_rht_stratified":
        non_empty = [i for i, s in enumerate(sectors) if len(s) > 0]
        if len(non_empty) < 3:
            ids = rng.choice(n, size=3, replace=False)
            return int(ids[0]), int(ids[1]), int(ids[2])
        # Choose separated sectors to reduce local/collinear triples.
        for _ in range(10):
            chosen = rng.choice(non_empty, size=3, replace=False)
            if len(set(chosen)) == 3:
                ids = [int(rng.choice(sectors[c])) for c in chosen]
                return ids[0], ids[1], ids[2]
        ids = rng.choice(n, size=3, replace=False)
        return int(ids[0]), int(ids[1]), int(ids[2])
    raise ValueError(f"Unknown D2 method: {cfg.d2_method}")


def process_one(cfg_dict: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    cfg = PipelineConfig(**cfg_dict)
    paths = run_paths(cfg.out)
    metric: Dict[str, Any] = {}
    with measured("D2", image_id) as metric:
        d1 = np.load(paths["d1"] / image_id / "d1_edges.npz")
        binary = d1["binary"].astype(bool)
        weights_map = d1["weights"].astype(np.float32)
        ny, nx = binary.shape
        points = np.argwhere(binary).astype(np.float64)  # columns: y, x
        n_points = int(len(points))
        n_votes_target = int(min(cfg.d2_max_votes, max(0, cfg.d2_votes_fraction * (min(nx, ny) ** 3))))
        rng = np.random.default_rng(cfg.seed + 200000 + int(image_id.split("_")[-1]))

        if n_points < 3 or n_votes_target <= 0:
            rows = np.zeros((0, 4), dtype=np.float32)
            attempted = accepted = 0
        else:
            cx0, cy0 = (nx - 1.0) / 2.0, (ny - 1.0) / 2.0
            point_angles = np.arctan2(points[:, 0] - cy0, points[:, 1] - cx0)
            sector_id = np.floor(((point_angles + np.pi) / (2.0 * np.pi)) * cfg.d2_n_sectors).astype(np.int32)
            sector_id = np.clip(sector_id, 0, cfg.d2_n_sectors - 1)
            all_keys = []
            all_votes = []
            accepted = 0
            attempted = 0
            batch = min(max(8192, n_votes_target // 4), 65536)
            max_attempts = max(n_votes_target * 10, batch)
            while accepted < n_votes_target and attempted < max_attempts:
                b = min(batch, max_attempts - attempted)
                ids = rng.integers(0, n_points, size=(b, 3), endpoint=False)
                # Ensure distinct point indices.
                distinct = (ids[:, 0] != ids[:, 1]) & (ids[:, 0] != ids[:, 2]) & (ids[:, 1] != ids[:, 2])
                if cfg.d2_method == "sparse_rht_stratified":
                    s0, s1, s2 = sector_id[ids[:, 0]], sector_id[ids[:, 1]], sector_id[ids[:, 2]]
                    distinct &= (s0 != s1) & (s0 != s2) & (s1 != s2)
                elif cfg.d2_method != "sparse_rht":
                    raise ValueError(f"Unknown D2 method: {cfg.d2_method}")
                ids = ids[distinct]
                attempted += b
                if len(ids) == 0:
                    continue
                p1 = points[ids[:, 0]]
                p2 = points[ids[:, 1]]
                p3 = points[ids[:, 2]]
                x1, y1 = p1[:, 1], p1[:, 0]
                x2, y2 = p2[:, 1], p2[:, 0]
                x3, y3 = p3[:, 1], p3[:, 0]
                twice_area = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                ok = twice_area >= (2.0 * cfg.d2_min_triangle_area)
                a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
                ok &= np.abs(a) > 1e-9
                if not np.any(ok):
                    continue
                x1, y1, x2, y2, x3, y3, a = x1[ok], y1[ok], x2[ok], y2[ok], x3[ok], y3[ok], a[ok]
                ids_ok = ids[ok]
                bcoef = ((x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1))
                ccoef = ((x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2))
                cx = -bcoef / (2.0 * a)
                cy = -ccoef / (2.0 * a)
                r = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
                ok2 = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(r) & (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny) & (r >= cfg.d2_r_min) & (r <= cfg.d2_r_max)
                if not np.any(ok2):
                    continue
                cx, cy, r, ids_ok = cx[ok2], cy[ok2], r[ok2], ids_ok[ok2]
                xb = np.rint(cx / cfg.d2_center_bin_px).astype(np.int32)
                yb = np.rint(cy / cfg.d2_center_bin_px).astype(np.int32)
                rb = np.rint(r / cfg.d2_radius_bin_px).astype(np.int32)
                if cfg.d2_vote_weight == "none":
                    votes = np.ones(len(xb), dtype=np.float32)
                else:
                    w1 = weights_map[points[ids_ok[:, 0], 0].astype(np.int32), points[ids_ok[:, 0], 1].astype(np.int32)]
                    w2 = weights_map[points[ids_ok[:, 1], 0].astype(np.int32), points[ids_ok[:, 1], 1].astype(np.int32)]
                    w3 = weights_map[points[ids_ok[:, 2], 0].astype(np.int32), points[ids_ok[:, 2], 1].astype(np.int32)]
                    if cfg.d2_vote_weight == "mean":
                        votes = ((w1 + w2 + w3) / 3.0).astype(np.float32)
                    elif cfg.d2_vote_weight == "product":
                        votes = (w1 * w2 * w3).astype(np.float32)
                    else:
                        raise ValueError(f"Unknown D2 vote weight: {cfg.d2_vote_weight}")
                keys = np.stack([xb, yb, rb], axis=1)
                take = min(len(keys), n_votes_target - accepted)
                if take > 0:
                    all_keys.append(keys[:take])
                    all_votes.append(votes[:take])
                    accepted += take
            if all_keys:
                keys = np.concatenate(all_keys, axis=0)
                votes = np.concatenate(all_votes, axis=0)
                # Aggregate sparse bins.
                uniq, inv = np.unique(keys, axis=0, return_inverse=True)
                sums = np.bincount(inv, weights=votes).astype(np.float32)
                rows = np.column_stack([
                    uniq[:, 0].astype(np.float32) * cfg.d2_center_bin_px,
                    uniq[:, 1].astype(np.float32) * cfg.d2_center_bin_px,
                    uniq[:, 2].astype(np.float32) * cfg.d2_radius_bin_px,
                    sums,
                ]).astype(np.float32)
                order = np.argsort(rows[:, 3])[::-1]
                rows = rows[order]
            else:
                rows = np.zeros((0, 4), dtype=np.float32)
        out_dir = paths["d2"] / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "d2_sparse_accumulator.npz", accumulator=rows)
        meta = {
            "image_id": image_id,
            "method": cfg.d2_method,
            "vote_weight": cfg.d2_vote_weight,
            "n_edge_points": n_points,
            "requested_votes": n_votes_target,
            "attempted_triplets": int(attempted),
            "accepted_triplets": int(accepted),
            "n_accumulator_bins": int(len(rows)),
            "radius_bounds": [cfg.d2_r_min, cfg.d2_r_max],
            "binning": {"center_bin_px": cfg.d2_center_bin_px, "radius_bin_px": cfg.d2_radius_bin_px},
            "paths": {"accumulator_npz": str(out_dir / "d2_sparse_accumulator.npz")},
        }
        dump_json(out_dir / "d2_meta.json", meta)
        metric.update({"edge_points": n_points, "accepted_triplets": int(accepted), "accumulator_bins": int(len(rows))})
    write_worker_metric(cfg.out, metric)
    return {"image_id": image_id, "status": "ok", "accumulator_bins": metric.get("accumulator_bins")}
