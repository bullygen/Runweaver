from __future__ import annotations

"""Reproducible, leakage-safe synthetic data for the article experiments.

This module deliberately lives outside D1--D6.  It changes the distribution of
the inputs, not the mathematical hypotheses or implementations of the detector.
The generated directory layout is compatible with the existing pipeline stages:
``images/<image_id>/image.npy`` and ``truth.json`` are the only required inputs.
"""

import hashlib
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import DEFAULT_BOUNDS
from .io_utils import dump_json, read_json
from .models import background_model, meshgrid_xy


DATASET_FORMAT = "smbh-cv-article-dataset-v1"


def _stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _balanced(rng: np.random.Generator, values: Sequence[Any], n: int) -> List[Any]:
    """Return shuffled cycles, so every discrete level is represented evenly."""
    if not values:
        raise ValueError("A factor must have at least one value")
    out: List[Any] = []
    while len(out) < n:
        block = list(values)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def _split_size(spec: Mapping[str, Any], scale: str) -> int:
    key = "n_images" if scale == "full" else "pilot_images"
    if key not in spec:
        raise ValueError(f"Split does not define {key}")
    return int(spec[key])


def _id_factor_rows(split_name: str, spec: Mapping[str, Any], n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    f = spec["factors"]
    positive_fraction = float(spec.get("positive_fraction", 0.75))
    n_positive = int(round(n * positive_fraction))
    is_positive = np.asarray([True] * n_positive + [False] * (n - n_positive), dtype=bool)
    rng.shuffle(is_positive)
    columns = {
        "target_snr": _balanced(rng, f["target_snr"], n),
        "noise_model": _balanced(rng, f["noise_model"], n),
        "background_value": _balanced(rng, f["background_value"], n),
        "background_gradient": _balanced(rng, f["background_gradient"], n),
        "background_class": _balanced(rng, f["background_class"], n),
        "beam_fwhm_px": _balanced(rng, f["beam_fwhm_px"], n),
        "uv_coverage": _balanced(rng, f["uv_coverage"], n),
        "variability_fraction": _balanced(rng, f["variability_fraction"], n),
        "ring_profile": _balanced(rng, f["ring_profile"], n),
        "parameter_domain": _balanced(rng, f["parameter_domain"], n),
        "overlap_mode": _balanced(rng, f["overlap_mode"], n),
    }
    null_levels = _balanced(rng, f["null_type"], n - n_positive)
    null_index = 0
    rows: List[Dict[str, Any]] = []
    n_folds = int(spec.get("n_folds", 5))
    folds = _balanced(rng, list(range(n_folds)), n)
    for i in range(n):
        row = {k: v[i] for k, v in columns.items()}
        row["is_positive"] = bool(is_positive[i])
        if is_positive[i]:
            row["null_type"] = "none"
        else:
            row["null_type"] = null_levels[null_index]
            null_index += 1
        row["fold"] = int(folds[i])
        row["scenario"] = "in_distribution"
        rows.append(row)
    return rows


def _verification_rows(spec: Mapping[str, Any], n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    scenarios = spec["scenarios"]
    names = list(scenarios)
    scenario_names = _balanced(rng, names, n)
    rows: List[Dict[str, Any]] = []
    folds = _balanced(rng, list(range(int(spec.get("n_folds", 5)))), n)
    for i, name in enumerate(scenario_names):
        sc = scenarios[name]
        positive_fraction = float(sc.get("positive_fraction", 1.0))
        # This deterministic rule keeps the requested fraction within each scenario.
        local_index = sum(1 for prev in scenario_names[:i] if prev == name)
        period = 100
        is_positive = (local_index % period) < int(round(period * positive_fraction))
        choose = lambda key, default: sc.get(key, default)[int(rng.integers(0, len(sc.get(key, default))))]
        row = {
            "target_snr": choose("target_snr", [5.0]),
            "noise_model": choose("noise_model", ["gaussian"]),
            "background_value": choose("background_value", [0.0]),
            "background_gradient": choose("background_gradient", [0.0]),
            "background_class": choose("background_class", ["jet"]),
            "beam_fwhm_px": choose("beam_fwhm_px", [4.0]),
            "uv_coverage": choose("uv_coverage", ["filled"]),
            "variability_fraction": choose("variability_fraction", [0.0]),
            "ring_profile": choose("ring_profile", ["gaussian"]),
            "parameter_domain": choose("parameter_domain", ["id_core"]),
            "overlap_mode": choose("overlap_mode", ["random"]),
            "null_type": "none" if is_positive else choose("null_type", ["turbulent"]),
            "is_positive": bool(is_positive),
            "fold": int(folds[i]),
            "scenario": name,
        }
        rows.append(row)
    return rows


def build_design(plan: Mapping[str, Any], split_name: str, scale: str = "full") -> Dict[str, Any]:
    if scale not in ("full", "pilot"):
        raise ValueError("scale must be 'full' or 'pilot'")
    split_spec = plan["splits"][split_name]
    n = _split_size(split_spec, scale)
    seed = int(split_spec["seed"])
    rng = np.random.default_rng(seed)
    if "scenarios" in split_spec:
        factor_rows = _verification_rows(split_spec, n, rng)
    else:
        factor_rows = _id_factor_rows(split_name, split_spec, n, rng)
    nx, ny = int(plan["image_shape"][0]), int(plan["image_shape"][1])
    items: List[Dict[str, Any]] = []
    for i, factors in enumerate(factor_rows):
        sample_seed = seed + 1_000_003 * (i + 1)
        family_id = f"{split_name}-{sample_seed:012d}"
        items.append({
            "image_id": f"image_{i:06d}",
            "split": split_name,
            "fold": factors["fold"],
            "family_id": family_id,
            "sample_seed": sample_seed,
            "nx": nx,
            "ny": ny,
            "factors": factors,
        })
    return {
        "format": DATASET_FORMAT,
        "plan_version": plan["version"],
        "plan_hash": _stable_hash(plan),
        "split": split_name,
        "scale": scale,
        "n_images": n,
        "items": items,
    }


def _domain_bounds(domain: str, nx: int, ny: int) -> Dict[str, List[float]]:
    b = {k: list(v) for k, v in DEFAULT_BOUNDS.items()}
    cx, cy = nx / 2.0, ny / 2.0
    if domain == "id_core":
        b.update({
            "A": [0.015, 0.065], "R": [52.0, 88.0], "sigma": [5.5, 9.5],
            "B": [0.05, 0.90], "x0": [cx - 26.0, cx + 26.0], "y0": [cy - 26.0, cy + 26.0],
        })
    elif domain == "id_boundary":
        b.update({
            "A": [0.010, 0.070], "R": [50.0, 91.0], "sigma": [5.0, 10.0],
            "B": [0.0, 1.0], "x0": [cx - 30.0, cx + 30.0], "y0": [cy - 30.0, cy + 30.0],
        })
    elif domain == "ood_extended":
        b.update({
            "A": [0.007, 0.085], "R": [42.0, 105.0], "sigma": [3.0, 14.0],
            "B": [0.0, 1.0], "x0": [cx - 42.0, cx + 42.0], "y0": [cy - 42.0, cy + 42.0],
            "j_A": [0.025, 0.13], "j_sx": [1.0, 14.0], "j_sy": [1.0, 14.0],
            "j_x0": [cx - 45.0, cx + 45.0], "j_y0": [cy - 45.0, cy + 45.0],
        })
    else:
        raise ValueError(f"Unknown parameter domain: {domain}")
    return b


def _random_params(rng: np.random.Generator, bounds: Mapping[str, Sequence[float]], names: Iterable[str]) -> Dict[str, float]:
    return {name: float(rng.uniform(*bounds[name])) for name in names}


def _ring_image(
    X: np.ndarray,
    Y: np.ndarray,
    ring: Mapping[str, float],
    profile: str,
    mismatch: Mapping[str, float],
) -> np.ndarray:
    dx, dy = X - ring["x0"], Y - ring["y0"]
    angle = np.arctan2(dy, dx)
    if profile == "elliptical":
        q = float(mismatch["axis_ratio"])
        pa = float(mismatch["position_angle"])
        c, s = math.cos(pa), math.sin(pa)
        xp, yp = c * dx + s * dy, -s * dx + c * dy
        radius = np.sqrt((xp / q) ** 2 + (yp * q) ** 2)
    else:
        radius = np.sqrt(dx * dx + dy * dy)
    sigma = max(float(ring["sigma"]), 1e-9)
    if profile == "variable_width":
        sigma_map = sigma * (1.0 + float(mismatch["width_modulation"]) * np.cos(2.0 * angle + float(mismatch["width_phase"])))
        radial = np.exp(-((radius - ring["R"]) / np.maximum(sigma_map, 0.25 * sigma)) ** 2)
    elif profile == "lorentzian":
        radial = 1.0 / (1.0 + ((radius - ring["R"]) / sigma) ** 2)
    elif profile == "top_hat":
        z = np.abs(radius - ring["R"]) / sigma
        radial = np.where(z <= 0.75, 1.0, np.where(z < 1.25, 0.5 * (1.0 + np.cos(np.pi * (z - 0.75) / 0.5)), 0.0))
    elif profile in ("gaussian", "elliptical", "partial"):
        radial = np.exp(-((radius - ring["R"]) / sigma) ** 2)
    else:
        raise ValueError(f"Unknown ring profile: {profile}")
    angular = 1.0 - ring["B"] * np.cos(angle - ring["phi"])
    if profile == "partial":
        d = (angle - float(mismatch["arc_center"]) + np.pi) % (2.0 * np.pi) - np.pi
        angular = angular * (np.abs(d) <= float(mismatch["arc_width"]) / 2.0)
    return ring["A"] * angular / (2.0 * np.pi) * radial


def _mismatch_params(rng: np.random.Generator, profile: str) -> Dict[str, float]:
    return {
        "axis_ratio": float(rng.uniform(0.72, 0.90)) if profile == "elliptical" else 1.0,
        "position_angle": float(rng.uniform(-np.pi, np.pi)),
        "width_modulation": float(rng.uniform(0.25, 0.50)) if profile == "variable_width" else 0.0,
        "width_phase": float(rng.uniform(-np.pi, np.pi)),
        "arc_center": float(rng.uniform(-np.pi, np.pi)),
        "arc_width": float(rng.uniform(0.9 * np.pi, 1.6 * np.pi)) if profile == "partial" else 2.0 * np.pi,
    }


def _render_dynamic_ring(
    X: np.ndarray,
    Y: np.ndarray,
    ring: Mapping[str, float],
    profile: str,
    mismatch: Mapping[str, float],
    variability: float,
    rng: np.random.Generator,
    n_frames: int,
) -> np.ndarray:
    if variability <= 0 or n_frames <= 1:
        return _ring_image(X, Y, ring, profile, mismatch)
    frames = []
    for _ in range(n_frames):
        r = dict(ring)
        r["A"] = max(1e-6, r["A"] * (1.0 + rng.normal(0.0, variability)))
        r["R"] = max(1.0, r["R"] * (1.0 + rng.normal(0.0, 0.35 * variability)))
        r["sigma"] = max(0.5, r["sigma"] * (1.0 + rng.normal(0.0, 0.5 * variability)))
        r["phi"] = r["phi"] + rng.normal(0.0, np.pi * variability)
        r["x0"] = r["x0"] + rng.normal(0.0, 2.0 * variability)
        r["y0"] = r["y0"] + rng.normal(0.0, 2.0 * variability)
        frames.append(_ring_image(X, Y, r, profile, mismatch))
    return np.mean(frames, axis=0)


def _null_distractors(X: np.ndarray, Y: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    image = np.zeros_like(X, dtype=float)
    nx, ny = X.shape[1], X.shape[0]
    if kind == "none":
        return image
    if kind in ("gaussian_blobs", "jet_knots"):
        for _ in range(int(rng.integers(2, 7))):
            x0, y0 = rng.uniform(0.25 * nx, 0.75 * nx), rng.uniform(0.25 * ny, 0.75 * ny)
            sx = rng.uniform(2.0, 12.0 if kind == "gaussian_blobs" else 5.0)
            sy = rng.uniform(2.0, 12.0 if kind == "gaussian_blobs" else 22.0)
            phi = rng.uniform(-np.pi, np.pi)
            c, s = np.cos(phi), np.sin(phi)
            xp, yp = c * (X - x0) + s * (Y - y0), -s * (X - x0) + c * (Y - y0)
            image += rng.uniform(0.002, 0.015) * np.exp(-0.5 * ((xp / sx) ** 2 + (yp / sy) ** 2))
    elif kind == "smooth_disk":
        x0, y0 = rng.uniform(0.4 * nx, 0.6 * nx), rng.uniform(0.4 * ny, 0.6 * ny)
        r = np.hypot(X - x0, Y - y0)
        image = rng.uniform(0.005, 0.02) * np.exp(-(r / rng.uniform(25.0, 65.0)) ** 2)
    elif kind == "crescent":
        x0, y0 = rng.uniform(0.43 * nx, 0.57 * nx), rng.uniform(0.43 * ny, 0.57 * ny)
        scale = rng.uniform(35.0, 70.0)
        outer = np.exp(-((np.hypot(X - x0, Y - y0) / scale) ** 4))
        inner = np.exp(-((np.hypot(X - x0 - rng.uniform(8, 20), Y - y0) / (0.72 * scale)) ** 4))
        image = rng.uniform(0.004, 0.015) * np.maximum(outer - inner, 0.0)
    elif kind == "short_arcs":
        ring = {"A": float(rng.uniform(0.02, 0.07)), "R": float(rng.uniform(45, 95)), "sigma": float(rng.uniform(4, 12)),
                "B": float(rng.uniform(0, 0.8)), "phi": float(rng.uniform(-np.pi, np.pi)),
                "x0": float(rng.uniform(0.4 * nx, 0.6 * nx)), "y0": float(rng.uniform(0.4 * ny, 0.6 * ny))}
        mm = _mismatch_params(rng, "partial")
        mm["arc_width"] = float(rng.uniform(0.25 * np.pi, 0.7 * np.pi))
        image = _ring_image(X, Y, ring, "partial", mm)
    elif kind == "turbulent":
        raw = rng.normal(size=X.shape)
        field = gaussian_filter(raw, sigma=rng.uniform(3.0, 12.0), mode="reflect")
        field -= np.min(field)
        image = field / max(float(np.max(field)), 1e-12) * rng.uniform(0.004, 0.015)
    else:
        raise ValueError(f"Unknown null type: {kind}")
    return image


def _extra_background(X: np.ndarray, Y: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "jet":
        return np.zeros_like(X)
    if kind == "two_blob":
        return _null_distractors(X, Y, "gaussian_blobs", rng) * 0.5
    if kind == "turbulent":
        return _null_distractors(X, Y, "turbulent", rng) * 0.5
    if kind == "smooth_disk":
        return _null_distractors(X, Y, "smooth_disk", rng) * 0.5
    raise ValueError(f"Unknown background class: {kind}")


def _uv_mask(shape: Sequence[int], profile: str, rng: np.random.Generator) -> np.ndarray:
    ny, nx = int(shape[0]), int(shape[1])
    yy = (np.arange(ny) - ny // 2) / max(ny, 1)
    xx = (np.arange(nx) - nx // 2) / max(nx, 1)
    U, V = np.meshgrid(xx, yy)
    rho = np.hypot(U, V)
    if profile == "filled":
        return np.ones((ny, nx), dtype=float)
    settings = {
        "eht_sparse": (8, 0.30, 0.010, 0.025),
        "ng_eht_dense": (20, 0.38, 0.014, 0.080),
        "space_vlbi": (13, 0.49, 0.008, 0.035),
        "sparse_adversarial": (5, 0.27, 0.007, 0.010),
    }
    if profile not in settings:
        raise ValueError(f"Unknown uv coverage: {profile}")
    n_tracks, cutoff, width, floor = settings[profile]
    mask = np.zeros_like(rho)
    angles = np.linspace(0.0, np.pi, n_tracks, endpoint=False) + rng.uniform(-0.06, 0.06, size=n_tracks)
    for angle in angles:
        perpendicular = np.abs(-np.sin(angle) * U + np.cos(angle) * V)
        along = np.abs(np.cos(angle) * U + np.sin(angle) * V)
        track = np.exp(-0.5 * (perpendicular / width) ** 2) * (along <= cutoff)
        mask = np.maximum(mask, track)
    core = np.exp(-0.5 * (rho / 0.035) ** 2)
    radial_taper = np.exp(-((rho / max(cutoff, 1e-6)) ** 12))
    mask = np.maximum(mask, core) * radial_taper
    mask = floor + (1.0 - floor) * np.clip(mask, 0.0, 1.0)
    mask[ny // 2, nx // 2] = 1.0
    return mask


def _observe(image: np.ndarray, beam_fwhm_px: float, uv_coverage: str, rng: np.random.Generator) -> tuple[np.ndarray, Dict[str, float]]:
    observed = image.astype(float, copy=True)
    if beam_fwhm_px > 0:
        observed = gaussian_filter(observed, sigma=beam_fwhm_px / 2.354820045, mode="reflect")
    mask = _uv_mask(observed.shape, uv_coverage, rng)
    if uv_coverage != "filled":
        ft = np.fft.fftshift(np.fft.fft2(observed))
        observed = np.fft.ifft2(np.fft.ifftshift(ft * mask)).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask)).real)
    peak = max(float(np.max(np.abs(psf))), 1e-12)
    cy, cx = np.unravel_index(int(np.argmax(np.abs(psf))), psf.shape)
    rr = np.hypot(*np.ogrid[-cy:psf.shape[0]-cy, -cx:psf.shape[1]-cx])
    sidelobe = float(np.max(np.abs(psf)[rr > 4])) / peak if np.any(rr > 4) else 0.0
    return observed, {"uv_mask_fraction_gt_0_5": float(np.mean(mask > 0.5)), "dirty_beam_peak_sidelobe": sidelobe}


def _n_true(rng: np.random.Generator) -> int:
    group = rng.choice(3, p=[0.30, 0.45, 0.25])
    lo_hi = [(1, 3), (4, 7), (8, 12)][int(group)]
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _render_item(item: Mapping[str, Any], split_dir: str, save_components: bool) -> Dict[str, Any]:
    split_path = Path(split_dir)
    image_id = str(item["image_id"])
    factors = dict(item["factors"])
    rng = np.random.default_rng(int(item["sample_seed"]))
    nx, ny = int(item["nx"]), int(item["ny"])
    X, Y = meshgrid_xy(nx, ny)
    bounds = _domain_bounds(str(factors["parameter_domain"]), nx, ny)
    ring_names = ["A", "R", "sigma", "B", "phi", "x0", "y0"]
    env_names = ["j_A", "j_sx", "j_sy", "j_phi", "j_x0", "j_y0"]
    n_true = _n_true(rng) if factors["is_positive"] else 0
    rings = [_random_params(rng, bounds, ring_names) for _ in range(n_true)]
    if factors["overlap_mode"] == "clustered" and rings:
        cx, cy = rng.uniform(0.44 * nx, 0.56 * nx), rng.uniform(0.44 * ny, 0.56 * ny)
        for ring in rings:
            ring["x0"] = float(np.clip(cx + rng.normal(0, 6), 0, nx - 1))
            ring["y0"] = float(np.clip(cy + rng.normal(0, 6), 0, ny - 1))
    env = _random_params(rng, bounds, env_names)
    background = background_model(X, Y, env)
    background += float(factors["background_value"])
    gradient_angle = float(rng.uniform(-np.pi, np.pi))
    gx = (X - (nx - 1) / 2.0) / max(nx - 1, 1)
    gy = (Y - (ny - 1) / 2.0) / max(ny - 1, 1)
    background += float(factors["background_gradient"]) * (np.cos(gradient_angle) * gx + np.sin(gradient_angle) * gy)
    background += _extra_background(X, Y, str(factors["background_class"]), rng)
    distractors = _null_distractors(X, Y, str(factors["null_type"]), rng) if not factors["is_positive"] else np.zeros_like(X)
    mismatch_rows: List[Dict[str, float]] = []
    ring_signal = np.zeros_like(X, dtype=float)
    for ring in rings:
        mismatch = _mismatch_params(rng, str(factors["ring_profile"]))
        mismatch_rows.append(mismatch)
        ring_signal += _render_dynamic_ring(
            X, Y, ring, str(factors["ring_profile"]), mismatch,
            float(factors["variability_fraction"]), rng, n_frames=8,
        )
    sky_clean = background + distractors + ring_signal
    # Reuse the same deterministic transfer operator for the scene and ring-only SNR reference.
    observation_seed = int(item["sample_seed"]) + 71
    observed_clean, resolution_meta = _observe(sky_clean, float(factors["beam_fwhm_px"]), str(factors["uv_coverage"]), np.random.default_rng(observation_seed))
    if rings:
        observed_ring, _ = _observe(ring_signal, float(factors["beam_fwhm_px"]), str(factors["uv_coverage"]), np.random.default_rng(observation_seed))
        snr_reference_kind = "observed_true_ring_rms"
    else:
        phantom = _random_params(rng, _domain_bounds("id_core", nx, ny), ring_names)
        phantom_signal = _ring_image(X, Y, phantom, "gaussian", _mismatch_params(rng, "gaussian"))
        observed_ring, _ = _observe(phantom_signal, float(factors["beam_fwhm_px"]), str(factors["uv_coverage"]), np.random.default_rng(observation_seed))
        snr_reference_kind = "unobserved_phantom_ring_rms"
    reference_rms = max(float(np.sqrt(np.mean(observed_ring ** 2))), 1e-8)
    target_snr = float(factors["target_snr"])
    noise_sigma = reference_rms / max(target_snr, 1e-9)
    raw_noise = rng.normal(0.0, noise_sigma, size=observed_clean.shape)
    if factors["noise_model"] == "correlated":
        raw_noise = gaussian_filter(raw_noise, sigma=2.0, mode="reflect")
        raw_noise *= noise_sigma / max(float(np.std(raw_noise)), 1e-12)
    elif factors["noise_model"] != "gaussian":
        raise ValueError(f"Unknown noise model: {factors['noise_model']}")
    image = observed_clean + raw_noise
    realised_noise_std = float(np.std(raw_noise, ddof=1))
    realised_snr = reference_rms / max(realised_noise_std, 1e-12)
    image_dir = split_path / "images" / image_id
    image_dir.mkdir(parents=True, exist_ok=True)
    np.save(image_dir / "image.npy", image.astype(np.float32))
    if save_components:
        np.save(image_dir / "clean_image.npy", observed_clean.astype(np.float32))
        np.save(image_dir / "truth_clean_pre_degradation.npy", sky_clean.astype(np.float32))
        np.save(image_dir / "noise.npy", raw_noise.astype(np.float32))
    truth = {
        "format": DATASET_FORMAT,
        "image_id": image_id,
        "split": item["split"],
        "fold": int(item["fold"]),
        "family_id": item["family_id"],
        "sample_seed": int(item["sample_seed"]),
        "nx": nx,
        "ny": ny,
        "dataset_mode": "article_factorial",
        "model": "SMBH_model_image",
        "artifact_model": "gaussian asymmetric ring",
        "background_model": "rotated gaussian jet/blob plus declared article factors",
        "true_n": n_true,
        "true_artifacts": rings,
        "true_env": env,
        "partials": mismatch_rows,
        "factors": factors,
        "generation_config": {
            **factors,
            "seed": int(item["sample_seed"]),
            "noise_sigma": noise_sigma,
            "realised_noise_std": realised_noise_std,
            "snr_reference_rms": reference_rms,
            "snr_reference_kind": snr_reference_kind,
            "realised_snr": realised_snr,
            "variability_frames": 8,
            "gradient_angle_rad": gradient_angle,
        },
        "reconstruction_metadata": {
            "method": "image_domain_uv_dirty_beam_proxy",
            "beam_fwhm_px": float(factors["beam_fwhm_px"]),
            "uv_coverage": factors["uv_coverage"],
            **resolution_meta,
            "scientific_limit": "proxy dirty-beam transfer; not calibrated visibility simulation or CLEAN/RML reconstruction",
        },
        "paths": {
            "image": str(image_dir / "image.npy"),
            "clean_image": str(image_dir / "clean_image.npy") if save_components else None,
            "noise": str(image_dir / "noise.npy") if save_components else None,
        },
    }
    dump_json(image_dir / "truth.json", truth)
    digest = hashlib.sha256(np.ascontiguousarray(image.astype(np.float32)).tobytes()).hexdigest()
    return {
        "image_id": image_id,
        "status": "ok",
        "true_n": n_true,
        "family_id": item["family_id"],
        "sample_seed": int(item["sample_seed"]),
        "fold": int(item["fold"]),
        "factors": factors,
        "image_sha256": digest,
    }


def _safe_replace_dir(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty dataset split: {path}")
    if overwrite:
        shutil.rmtree(path)


def generate_split(
    plan: Mapping[str, Any],
    split_name: str,
    output_root: str | Path,
    scale: str = "full",
    workers: int = 1,
    overwrite: bool = False,
    save_components: bool = True,
) -> Dict[str, Any]:
    design = build_design(plan, split_name, scale=scale)
    split_dir = Path(output_root) / split_name
    _safe_replace_dir(split_dir, overwrite=overwrite)
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    dump_json(split_dir / "design.json", design)
    items: List[Dict[str, Any]] = []
    if workers <= 1:
        items = [_render_item(item, str(split_dir), save_components) for item in design["items"]]
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [executor.submit(_render_item, item, str(split_dir), save_components) for item in design["items"]]
            for future in as_completed(futures):
                items.append(future.result())
    items.sort(key=lambda row: row["image_id"])
    manifest = {
        "format": DATASET_FORMAT,
        "plan_version": plan["version"],
        "plan_hash": design["plan_hash"],
        "split": split_name,
        "scale": scale,
        "save_components": bool(save_components),
        "items": items,
    }
    dump_json(split_dir / "manifest.json", manifest)
    return manifest


def validate_dataset(output_root: str | Path, expected_plan: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    root = Path(output_root)
    errors: List[str] = []
    warnings: List[str] = []
    split_rows: List[Dict[str, Any]] = []
    seen_families: set[str] = set()
    seen_hashes: Dict[str, str] = {}
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        errors.append("No split manifests found")
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        split = str(manifest.get("split", manifest_path.parent.name))
        items = manifest.get("items", [])
        factor_counts: Dict[str, Dict[str, int]] = {}
        n_positive = n_null = 0
        for item in items:
            image_id = str(item["image_id"])
            family = str(item["family_id"])
            if family in seen_families:
                errors.append(f"family_id leakage across splits: {family}")
            seen_families.add(family)
            digest = str(item.get("image_sha256", ""))
            if digest and digest in seen_hashes:
                errors.append(f"duplicate image content: {split}/{image_id} equals {seen_hashes[digest]}")
            if digest:
                seen_hashes[digest] = f"{split}/{image_id}"
            image_dir = manifest_path.parent / "images" / image_id
            image_path, truth_path = image_dir / "image.npy", image_dir / "truth.json"
            if not image_path.exists() or not truth_path.exists():
                errors.append(f"missing image or truth for {split}/{image_id}")
                continue
            image = np.load(image_path, mmap_mode="r")
            if image.shape != (int(read_json(truth_path)["ny"]), int(read_json(truth_path)["nx"])):
                errors.append(f"shape mismatch for {split}/{image_id}")
            if not np.all(np.isfinite(image)):
                errors.append(f"non-finite pixels for {split}/{image_id}")
            actual_digest = hashlib.sha256(np.ascontiguousarray(np.asarray(image, dtype=np.float32)).tobytes()).hexdigest()
            declared_digest = str(item.get("image_sha256", ""))
            if declared_digest and actual_digest != declared_digest:
                errors.append(f"image SHA-256 mismatch for {split}/{image_id}")
            truth = read_json(truth_path)
            if int(truth["true_n"]) != len(truth.get("true_artifacts", [])):
                errors.append(f"true_n mismatch for {split}/{image_id}")
            if truth["true_n"]:
                n_positive += 1
            else:
                n_null += 1
            for key, value in item.get("factors", {}).items():
                factor_counts.setdefault(key, {})[str(value)] = factor_counts.setdefault(key, {}).get(str(value), 0) + 1
        if expected_plan is not None:
            expected = _split_size(expected_plan["splits"][split], str(manifest.get("scale", "full")))
            if len(items) != expected:
                errors.append(f"{split}: expected {expected} items, found {len(items)}")
        split_rows.append({
            "split": split,
            "n_images": len(items),
            "n_positive": n_positive,
            "n_null": n_null,
            "factor_counts": factor_counts,
        })
    report = {
        "format": DATASET_FORMAT,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_splits": len(manifests),
        "n_unique_families": len(seen_families),
        "n_unique_image_hashes": len(seen_hashes),
        "splits": split_rows,
    }
    dump_json(root / "validation_report.json", report)
    return report


def load_plan(path: str | Path) -> Dict[str, Any]:
    plan = read_json(Path(path))
    required = {"version", "image_shape", "splits"}
    missing = required - set(plan)
    if missing:
        raise ValueError(f"Dataset plan misses fields: {sorted(missing)}")
    return plan
