from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from .config import PipelineConfig, parse_csv_floats
from .io_utils import dump_json, run_paths, measured, write_worker_metric


def _mask_jet(image: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    masked = image.astype(np.float64).copy()
    meta: Dict[str, Any] = {"method": cfg.d1_mask_method}
    fill = float(np.median(masked))
    if cfg.d1_mask_method == "none":
        return masked, meta
    if cfg.d1_mask_method == "rect_max":
        y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
        h = int(cfg.d1_rect_half_size)
        masked[max(0, y - h): min(masked.shape[0], y + h + 1), max(0, x - h): min(masked.shape[1], x + h + 1)] = fill
        meta.update({"x": int(x), "y": int(y), "half_size": h})
        return masked, meta
    if cfg.d1_mask_method == "percentile_cc":
        thr = float(np.percentile(masked, cfg.d1_jet_percentile))
        lab = measure.label(masked >= thr, connectivity=2)
        props = measure.regionprops(lab, intensity_image=masked)
        if not props:
            return masked, meta | {"threshold": thr, "found": False}
        # skimage >=0.26 renamed intensity_image -> image_intensity.
        def integrated_flux(p: Any) -> float:
            img = getattr(p, "image_intensity", None)
            if img is None:
                img = p.intensity_image
            return float(img[p.image].sum())
        prop = max(props, key=integrated_flux)
        comp = lab == prop.label
        comp = morphology.dilation(comp, morphology.disk(max(1, cfg.d1_rect_half_size)))
        masked[comp] = fill
        cy, cx = prop.centroid
        meta.update({
            "threshold": thr,
            "found": True,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "area_px": int(prop.area),
            "dilated_area_px": int(np.sum(comp)),
        })
        return masked, meta
    raise ValueError(f"Unknown D1 mask method: {cfg.d1_mask_method}")


def _edge_score(masked: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    method = cfg.d1_edge_method
    meta: Dict[str, Any] = {"method": method}
    if method == "legacy_laplace_positive":
        # Original project boundary filter: convolution with the 4-neighbour discrete Laplacian.
        # The following threshold must normally be --d1-threshold-method positive.
        kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
        score = ndi.convolve(masked, kernel, mode="nearest")
        meta["kernel"] = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        meta["signed_response"] = True
    elif method == "log_mad":
        sigmas = parse_csv_floats(cfg.d1_edge_sigmas)
        scores = [np.abs(ndi.gaussian_laplace(masked, sigma=s)) for s in sigmas]
        score = np.max(np.stack(scores, axis=0), axis=0)
        meta["sigmas"] = sigmas
        meta["signed_response"] = False
    elif method == "dog":
        sigmas = parse_csv_floats(cfg.d1_edge_sigmas)
        scores = []
        for s in sigmas:
            scores.append(np.abs(ndi.gaussian_filter(masked, s) - ndi.gaussian_filter(masked, 1.6 * s)))
        score = np.max(np.stack(scores, axis=0), axis=0)
        meta["sigmas"] = sigmas
        meta["signed_response"] = False
    elif method == "sobel":
        score = filters.sobel(masked)
        meta["signed_response"] = False
    elif method == "laplace_sign":
        score = np.maximum(ndi.laplace(masked), 0.0)
        meta["signed_response"] = False
    else:
        raise ValueError(f"Unknown D1 edge method: {method}")
    return score.astype(np.float32), meta


def _threshold(score: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method": cfg.d1_threshold_method}
    if cfg.d1_threshold_method == "positive":
        binary = score > 0.0
        meta.update({"threshold": 0.0})
    elif cfg.d1_threshold_method == "mad":
        med = float(np.median(score))
        mad = float(1.4826 * np.median(np.abs(score - med)))
        thr = med + cfg.d1_tau * (mad if mad > 0 else float(np.std(score)))
        binary = score > thr
        meta.update({"median": med, "mad_sigma": mad, "threshold": float(thr), "tau": cfg.d1_tau})
    elif cfg.d1_threshold_method == "percentile":
        thr = float(np.percentile(score, cfg.d1_percentile))
        binary = score > thr
        meta.update({"threshold": thr, "percentile": cfg.d1_percentile})
    elif cfg.d1_threshold_method == "sauvola":
        thr = filters.threshold_sauvola(score, window_size=31)
        binary = score > thr
        meta.update({"threshold_mean": float(np.mean(thr))})
    else:
        raise ValueError(f"Unknown D1 threshold method: {cfg.d1_threshold_method}")
    return binary, meta


def _legacy_control_open(binary: np.ndarray, image: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Original project stopping rule for repeated morphological opening."""
    y_peak, x_peak = np.unravel_index(int(np.argmax(image)), image.shape)
    ny, nx = image.shape
    cx, cy = (nx - 1.0) / 2.0, (ny - 1.0) / 2.0
    control_radius = float(np.sqrt((x_peak - cx) ** 2 + (y_peak - cy) ** 2 + cfg.d1_rect_half_size ** 2))
    yy, xx = np.ogrid[:ny, :nx]
    control = (xx - cx) ** 2 + (yy - cy) ** 2 <= control_radius ** 2
    control_size = int(np.sum(control))
    structure = np.ones((3, 3), dtype=bool)
    best = binary.astype(bool)
    best_n = 0
    best_ratio = float(np.sum(best[control]) / max(control_size, 1))
    for n in range(1, int(cfg.d1_legacy_max_iter) + 1):
        opened = ndi.binary_opening(binary.astype(bool), structure=structure, iterations=n)
        ratio = float(np.sum(opened[control]) / max(control_size, 1))
        best = opened.astype(bool)
        best_n = n
        best_ratio = ratio
        if ratio <= cfg.d1_legacy_badpix:
            break
    return best, {
        "method": "legacy_control_open",
        "iterations": int(best_n),
        "control_radius_px": control_radius,
        "control_size_px": control_size,
        "control_badpix_fraction": best_ratio,
        "target_badpix_fraction": float(cfg.d1_legacy_badpix),
    }


def _normalize_weights(score: np.ndarray) -> np.ndarray:
    w = np.maximum(score.astype(np.float32), 0.0)
    maxv = float(np.max(w))
    return (w / maxv).astype(np.float32) if maxv > 0 else w.astype(np.float32)


def process_one(cfg_dict: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    cfg = PipelineConfig(**cfg_dict)
    paths = run_paths(cfg.out)
    metric: Dict[str, Any] = {}
    with measured("D1", image_id) as metric:
        image_path = paths["images"] / image_id / "image.npy"
        image = np.load(image_path)
        masked, mask_meta = _mask_jet(image, cfg)
        score, edge_meta = _edge_score(masked, cfg)
        binary, threshold_meta = _threshold(score, cfg)

        before_morph = int(np.sum(binary))
        legacy_meta: Dict[str, Any] | None = None
        if cfg.d1_morphology_method == "legacy_control_open":
            binary, legacy_meta = _legacy_control_open(binary, image, cfg)
        elif cfg.d1_morphology_method == "open_close":
            binary = morphology.remove_small_objects(binary.astype(bool), min_size=cfg.d1_min_area)
            binary = morphology.binary_opening(binary, morphology.disk(1))
            binary = morphology.binary_closing(binary, morphology.disk(1))
        elif cfg.d1_morphology_method == "remove_small":
            binary = morphology.remove_small_objects(binary.astype(bool), min_size=cfg.d1_min_area)
        elif cfg.d1_morphology_method == "none":
            binary = binary.astype(bool)
        else:
            raise ValueError(f"Unknown D1 morphology method: {cfg.d1_morphology_method}")
        after_morph = int(np.sum(binary))
        if cfg.d1_skeletonize and np.any(binary):
            binary = morphology.skeletonize(binary).astype(bool)
        after_skeleton = int(np.sum(binary))

        weights = _normalize_weights(score)
        out_dir = paths["d1"] / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "d1_edges.npz", binary=binary.astype(np.uint8), weights=weights.astype(np.float32), edge_response=score.astype(np.float32))
        meta = {
            "image_id": image_id,
            "mask": mask_meta,
            "edge": edge_meta,
            "threshold": threshold_meta,
            "morphology": {
                "method": cfg.d1_morphology_method,
                "min_area": cfg.d1_min_area,
                "before_morph_nonzero": before_morph,
                "after_morph_nonzero": after_morph,
                "skeletonize": cfg.d1_skeletonize,
                "after_skeleton_nonzero": after_skeleton,
                "legacy_control": legacy_meta,
            },
            "paths": {"d1_npz": str(out_dir / "d1_edges.npz")},
        }
        dump_json(out_dir / "d1_meta.json", meta)
        metric.update({"edge_pixels": after_skeleton, "before_morph_pixels": before_morph})
    write_worker_metric(cfg.out, metric)
    return {"image_id": image_id, "status": "ok", "edge_pixels": metric.get("edge_pixels")}
