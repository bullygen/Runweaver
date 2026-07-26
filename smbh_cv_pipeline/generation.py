from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import PipelineConfig
from .io_utils import dump_json, run_paths, measured, write_worker_metric
from .models import meshgrid_xy, random_env, random_ring, smbh_model_image


def _partial_ring_apply(X: np.ndarray, Y: np.ndarray, ring: Dict[str, float], rng: np.random.Generator, prob: float) -> Dict[str, Any]:
    """Return angular mask metadata; actual simple generator keeps full rings unless degraded mode asks partials."""
    if rng.random() >= prob:
        return {"enabled": False}
    center = float(rng.uniform(-np.pi, np.pi))
    width = float(rng.uniform(np.pi / 2.0, 1.75 * np.pi))
    return {"enabled": True, "center": center, "width": width}


def _ring_with_optional_partial(X: np.ndarray, Y: np.ndarray, ring: Dict[str, float], partial: Dict[str, Any]) -> np.ndarray:
    from .models import ring_model, angle_diff
    img = ring_model(X, Y, ring)
    if not partial.get("enabled", False):
        return img
    ang = np.arctan2(Y - ring["y0"], X - ring["x0"])
    d = np.vectorize(angle_diff)(ang, partial["center"])
    return img * (np.abs(d) <= partial["width"] / 2.0)


def generate_one(cfg_dict: Dict[str, Any], image_index: int) -> Dict[str, Any]:
    cfg = PipelineConfig(**cfg_dict)
    paths = run_paths(cfg.out)
    image_id = f"image_{image_index:04d}"
    metric: Dict[str, Any] = {}
    with measured("generate", image_id) as metric:
        rng = np.random.default_rng(cfg.seed + image_index)
        X, Y = meshgrid_xy(cfg.nx, cfg.ny)

        n_true = int(rng.integers(cfg.n_true_min, cfg.n_true_max + 1))
        rings = [random_ring(rng, cfg.true_bounds) for _ in range(n_true)]
        if cfg.dataset_mode == "degraded_model" and cfg.overlap_mode == "clustered" and rings:
            # Deliberately create a more difficult case with nearby centers/radii.
            cx = float(rng.uniform(cfg.nx * 0.42, cfg.nx * 0.58))
            cy = float(rng.uniform(cfg.ny * 0.42, cfg.ny * 0.58))
            for ring in rings:
                ring["x0"] = float(np.clip(cx + rng.normal(0, 5.0), 0, cfg.nx - 1))
                ring["y0"] = float(np.clip(cy + rng.normal(0, 5.0), 0, cfg.ny - 1))

        env = random_env(rng, cfg.true_bounds, cfg.noise_sigma)
        clean = smbh_model_image(X, Y, [], env)
        partials = []
        for ring in rings:
            partial = _partial_ring_apply(X, Y, ring, rng, cfg.partial_ring_prob if cfg.dataset_mode == "degraded_model" else 0.0)
            partials.append(partial)
            clean += _ring_with_optional_partial(X, Y, ring, partial)

        degraded_clean = clean.copy()
        reconstruction_metadata: Dict[str, Any] = {"method": cfg.reconstruction_method}
        if cfg.dataset_mode == "degraded_model" and cfg.beam_fwhm_px > 0:
            sigma = cfg.beam_fwhm_px / 2.354820045
            degraded_clean = gaussian_filter(degraded_clean, sigma=sigma, mode="nearest")
            reconstruction_metadata["beam_fwhm_px"] = cfg.beam_fwhm_px
            reconstruction_metadata["beam_sigma_px"] = sigma
        if cfg.dataset_mode == "degraded_model" and cfg.background_gradient != 0:
            gx = np.linspace(-0.5, 0.5, cfg.nx)[None, :]
            gy = np.linspace(-0.5, 0.5, cfg.ny)[:, None]
            degraded_clean = degraded_clean + cfg.background_gradient * (gx + gy)
            reconstruction_metadata["background_gradient"] = cfg.background_gradient

        if cfg.noise_model == "none" or cfg.noise_sigma <= 0:
            noise = np.zeros_like(degraded_clean)
        elif cfg.noise_model == "correlated":
            raw = rng.normal(0.0, cfg.noise_sigma, size=degraded_clean.shape)
            noise = gaussian_filter(raw, sigma=max(cfg.corr_noise_length_px, 1e-6), mode="reflect")
            std = float(np.std(noise))
            if std > 0:
                noise *= cfg.noise_sigma / std
        else:
            noise = rng.normal(0.0, cfg.noise_sigma, size=degraded_clean.shape)
        image = degraded_clean + noise

        image_dir = paths["images"] / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        np.save(image_dir / "image.npy", image.astype(np.float32))
        np.save(image_dir / "clean_image.npy", degraded_clean.astype(np.float32))
        np.save(image_dir / "truth_clean_pre_degradation.npy", clean.astype(np.float32))
        np.save(image_dir / "noise.npy", noise.astype(np.float32))
        truth = {
            "image_id": image_id,
            "nx": cfg.nx,
            "ny": cfg.ny,
            "dataset_mode": cfg.dataset_mode,
            "model": "SMBH_model_image",
            "artifact_model": "gaussian asymmetric ring",
            "background_model": "rotated gaussian jet/blob",
            "true_n": n_true,
            "true_artifacts": rings,
            "true_env": env,
            "partials": partials,
            "generation_config": {
                "seed": cfg.seed + image_index,
                "noise_sigma": cfg.noise_sigma,
                "noise_model": cfg.noise_model,
                "beam_fwhm_px": cfg.beam_fwhm_px,
                "corr_noise_length_px": cfg.corr_noise_length_px,
                "reconstruction_method": cfg.reconstruction_method,
            },
            "reconstruction_metadata": reconstruction_metadata,
            "paths": {
                "image": str(image_dir / "image.npy"),
                "clean_image": str(image_dir / "clean_image.npy"),
                "noise": str(image_dir / "noise.npy"),
            },
        }
        dump_json(image_dir / "truth.json", truth)
        dump_json(image_dir / "generation_config.json", truth["generation_config"])
        metric.update({"n_true": n_true, "image_shape": [cfg.ny, cfg.nx]})
    write_worker_metric(cfg.out, metric)
    return {"image_id": image_id, "status": "ok", "true_n": metric.get("n_true")}


def write_manifest(cfg: PipelineConfig, items: List[Dict[str, Any]]) -> None:
    paths = run_paths(cfg.out)
    dump_json(paths["config"], cfg.to_dict())
    dump_json(paths["manifest"], {"items": items})
