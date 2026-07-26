from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from .config import ENV_PARAM_NAMES, PARAM_NAMES, PipelineConfig
from .io_utils import dump_json, read_json, run_paths, measured, write_worker_metric
from .models import clamp_params


def _estimate_env(image: np.ndarray, cfg: PipelineConfig) -> Dict[str, float]:
    ny, nx = image.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    baseline = float(np.percentile(image, 20))
    threshold = float(np.percentile(image, 99.5))
    mask = image >= threshold
    lab = measure.label(mask, connectivity=2)
    props = measure.regionprops(lab, intensity_image=image)
    if props:
        def _flux(p):
            img = getattr(p, "image_intensity", None)
            if img is None:
                img = p.intensity_image
            return float(img[p.image].sum())
        prop = max(props, key=_flux)
        yy, xx = prop.coords[:, 0], prop.coords[:, 1]
        weights = np.clip(image[yy, xx] - baseline, 1e-12, None)
        x0 = float(np.average(xx, weights=weights))
        y0 = float(np.average(yy, weights=weights))
        sx = float(max(np.sqrt(np.average((xx - x0) ** 2, weights=weights)), 1.0))
        sy = float(max(np.sqrt(np.average((yy - y0) ** 2, weights=weights)), 1.0))
    else:
        weights = np.clip(image - baseline, 0.0, None)
        total = float(np.sum(weights))
        if total > 0:
            x0 = float(np.sum(weights * x_grid) / total)
            y0 = float(np.sum(weights * y_grid) / total)
        else:
            x0, y0 = (nx - 1) / 2.0, (ny - 1) / 2.0
        sx = sy = 3.0
    env = {
        "j_A": float(max(np.percentile(image, 99.9) - baseline, 1e-5)),
        "j_sx": sx,
        "j_sy": sy,
        "j_phi": 0.0,
        "j_x0": x0,
        "j_y0": y0,
        "noise_sigma": float(cfg.noise_sigma),
    }
    bounded = clamp_params(env, cfg.fit_bounds, ENV_PARAM_NAMES)
    bounded["noise_sigma"] = float(cfg.noise_sigma)
    return bounded


def _sample_profile(image: np.ndarray, x0: float, y0: float, R: float, radial_window: float, n_r: int = 41, n_a: int = 144) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    radii = np.linspace(max(1.0, R - radial_window), R + radial_window, n_r)
    angles = np.linspace(-np.pi, np.pi, n_a, endpoint=False)
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    xs = x0 + rr * np.cos(aa)
    ys = y0 + rr * np.sin(aa)
    vals = ndi.map_coordinates(image, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(n_r, n_a)
    return radii, angles, vals


def _init_ring_profile(image: np.ndarray, cand: Dict[str, float], cfg: PipelineConfig) -> Dict[str, float]:
    x0, y0, R0 = cand["x0"], cand["y0"], cand["R"]
    if cfg.d4_method == "constants":
        th = {"A": 0.035, "R": R0, "sigma": 7.0, "B": 0.5, "phi": 0.0, "x0": x0, "y0": y0}
        return clamp_params(th, cfg.fit_bounds, PARAM_NAMES)
    if cfg.d4_method != "profile_harmonic":
        raise ValueError(f"Unknown D4 method: {cfg.d4_method}")
    radii, angles, vals = _sample_profile(image, x0, y0, R0, cfg.d4_radial_window, n_a=cfg.d4_n_angle_bins)
    # Background from outer profile quantile.
    bg = float(np.percentile(vals, 20))
    prof = np.median(vals - bg, axis=1)
    imax = int(np.argmax(prof))
    R = float(radii[imax])
    peak = float(max(prof[imax], 1e-9))
    half = 0.5 * peak
    left = imax
    while left > 0 and prof[left] > half:
        left -= 1
    right = imax
    while right < len(prof) - 1 and prof[right] > half:
        right += 1
    fwhm = float(max(radii[right] - radii[left], 1.0))
    sigma = float(np.clip(fwhm / 1.665109222, cfg.fit_bounds["sigma"][0], cfg.fit_bounds["sigma"][1]))

    # Angular first harmonic in annular strip near R.
    strip = vals[np.abs(radii - R) <= cfg.d4_strip_half_width]
    if strip.size == 0:
        strip_mean = np.maximum(vals[imax] - bg, 0.0)
    else:
        strip_mean = np.maximum(np.mean(strip, axis=0) - bg, 0.0)
    total = float(np.sum(strip_mean))
    if total > 0:
        c1 = float(np.sum(strip_mean * np.cos(angles)))
        s1 = float(np.sum(strip_mean * np.sin(angles)))
        # Model is 1 - B cos(angle - phi); maximum is opposite phi for B>0.
        phi = float(np.arctan2(s1, c1) + np.pi)
        phi = float((phi + np.pi) % (2 * np.pi) - np.pi)
        B = float(np.clip(2.0 * np.hypot(c1, s1) / total, 0.0, 1.0))
    else:
        phi, B = 0.0, 0.3
    # In model, peak angular/radial value is roughly A/(2pi)*(1+B). Use robust strip max.
    peak_intensity = float(np.percentile(strip_mean, 95)) if strip_mean.size else peak
    A = float(np.clip(2.0 * np.pi * peak_intensity / max(1.0 + B, 1e-6), cfg.fit_bounds["A"][0], cfg.fit_bounds["A"][1]))
    th = {"A": A, "R": R, "sigma": sigma, "B": B, "phi": phi, "x0": x0, "y0": y0}
    return clamp_params(th, cfg.fit_bounds, PARAM_NAMES)


def process_one(cfg_dict: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    cfg = PipelineConfig(**cfg_dict)
    paths = run_paths(cfg.out)
    metric: Dict[str, Any] = {}
    with measured("D4", image_id) as metric:
        image = np.load(paths["images"] / image_id / "image.npy").astype(np.float64)
        cand_data = read_json(paths["d3"] / image_id / "d3_candidates.json")
        candidates = cand_data.get("candidates", [])[: cfg.d4_max_init_candidates]
        init_rings = [_init_ring_profile(image, c, cfg) | {"source_score": float(c.get("score", 0.0)), "source_confidence": float(c.get("confidence", 0.0))} for c in candidates]
        env = _estimate_env(image, cfg)
        out_dir = paths["d4"] / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_json(out_dir / "d4_initial_params.json", {
            "image_id": image_id,
            "method": cfg.d4_method,
            "initial_n": len(init_rings),
            "initial_artifacts": init_rings,
            "initial_env": env,
            "source_candidates_path": str(paths["d3"] / image_id / "d3_candidates.json"),
        })
        metric.update({"initial_n": len(init_rings)})
    write_worker_metric(cfg.out, metric)
    return {"image_id": image_id, "status": "ok", "initial_n": metric.get("initial_n")}
