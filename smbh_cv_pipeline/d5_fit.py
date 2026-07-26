from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .config import ENV_PARAM_NAMES, PARAM_NAMES, PipelineConfig
from .io_utils import dump_json, read_json, run_paths, measured, write_worker_metric
from .models import bounds_to_vectors, meshgrid_xy, rings_to_vector, smbh_model_image, vector_to_rings


def _weight_map(image: np.ndarray, rings: Sequence[Dict[str, float]], env: Dict[str, float], cfg: PipelineConfig) -> np.ndarray:
    if cfg.d5_residual_weighting == "uniform":
        return np.ones_like(image, dtype=np.float64)
    ny, nx = image.shape
    X, Y = meshgrid_xy(nx, ny)
    if cfg.d5_residual_weighting == "ring_support":
        w = np.full_like(image, 0.25, dtype=np.float64)
        for r in rings:
            rad = np.sqrt((X - r["x0"]) ** 2 + (Y - r["y0"]) ** 2)
            mask = np.abs(rad - r["R"]) <= max(2.5 * r["sigma"], 4.0)
            w[mask] = 1.0
        return w
    if cfg.d5_residual_weighting == "inverse_background":
        bg = np.maximum(smbh_model_image(X, Y, [], env), np.percentile(image, 10))
        return 1.0 / np.sqrt(bg + 1e-6)
    raise ValueError(f"Unknown D5 residual weighting: {cfg.d5_residual_weighting}")


def _residual(vec: np.ndarray, X: np.ndarray, Y: np.ndarray, image: np.ndarray, weights: np.ndarray, n_rings: int, fit_background: bool, env0: Dict[str, float], cfg: PipelineConfig) -> np.ndarray:
    rings, env = vector_to_rings(vec, n_rings, fit_background, env0)
    model = smbh_model_image(X, Y, rings, env)
    noise = cfg.noise_sigma if cfg.noise_sigma > 0 else max(float(np.std(image - np.median(image))), 1e-6)
    return ((model - image) * weights / noise).ravel()


def _fit_statistics(image: np.ndarray, model: np.ndarray, weights: np.ndarray, n_params: int, cfg: PipelineConfig) -> Dict[str, float]:
    resid = (model - image).astype(float)
    weighted = resid * weights
    rss = float(np.sum(weighted ** 2))
    n = int(image.size)
    dof = max(n - n_params, 1)
    sigma = cfg.noise_sigma if cfg.noise_sigma > 0 else max(float(np.std(resid)), 1e-12)
    chisq = float(np.sum((weighted / sigma) ** 2))
    y = image.astype(float).ravel()
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(resid.ravel() ** 2) / tss) if tss > 0 else float("nan")
    aic = float(n * np.log(max(rss / n, 1e-300)) + 2 * n_params)
    bic = float(n * np.log(max(rss / n, 1e-300)) + n_params * np.log(n))
    return {
        "n": n,
        "n_params": int(n_params),
        "dof": dof,
        "rss_weighted": rss,
        "rmse": float(np.sqrt(np.mean(resid.ravel() ** 2))),
        "mae": float(np.mean(np.abs(resid.ravel()))),
        "redchi_uncalibrated": float(chisq / dof),
        "r2": r2,
        "aic_weighted": aic,
        "bic_weighted": bic,
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid, ddof=1)) if n > 1 else 0.0,
    }


def _estimate_lsq_covariance(opt: Any) -> tuple[bool, List[float | None]]:
    """Return linearized standard errors for the fitted vector.

    The estimate is an approximation based on the Jacobian returned by
    scipy.optimize.least_squares. It is most meaningful for loss='linear'.
    For robust losses it is still useful as an experimental pruning score,
    but it should be reported as approximate.
    """
    try:
        jac = np.asarray(opt.jac, dtype=np.float64)
        m, n = jac.shape
        if n == 0 or m <= n:
            return False, [None] * n
        _, singular_values, vt = np.linalg.svd(jac, full_matrices=False)
        if singular_values.size == 0 or not np.isfinite(singular_values[0]) or singular_values[0] <= 0:
            return False, [None] * n
        tol = np.finfo(float).eps * max(jac.shape) * singular_values[0]
        rank = int(np.sum(singular_values > tol))
        if rank < n:
            return False, [None] * n
        s = singular_values[:rank]
        vt = vt[:rank]
        cov = (vt.T / (s * s)) @ vt
        rss = 2.0 * float(opt.cost)
        sigma2 = rss / max(m - n, 1)
        cov *= sigma2
        stderr = np.sqrt(np.diag(cov))
        if not np.all(np.isfinite(stderr)) or np.any(stderr <= 0):
            return False, [None] * n
        return True, [float(x) for x in stderr]
    except Exception:
        return False, []


def _split_stderr(stderr: Sequence[float | None], n_rings: int, fit_background: bool) -> tuple[List[Dict[str, float | None]], Dict[str, float | None]]:
    out: List[Dict[str, float | None]] = []
    env_err: Dict[str, float | None] = {}
    k = 0
    for _ in range(n_rings):
        er: Dict[str, float | None] = {}
        for name in PARAM_NAMES:
            er[name] = stderr[k] if k < len(stderr) else None
            k += 1
        out.append(er)
    if fit_background:
        for name in ENV_PARAM_NAMES:
            env_err[name] = stderr[k] if k < len(stderr) else None
            k += 1
    return out, env_err


def process_one(cfg_dict: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    cfg = PipelineConfig(**cfg_dict)
    if cfg.d5_engine != "scipy":
        raise ValueError("This implementation supports d5_engine='scipy'.")
    paths = run_paths(cfg.out)
    metric: Dict[str, Any] = {}
    with measured("D5", image_id) as metric:
        image = np.load(paths["images"] / image_id / "image.npy").astype(np.float64)
        init = read_json(paths["d4"] / image_id / "d4_initial_params.json")
        rings0 = [{p: float(r[p]) for p in PARAM_NAMES} for r in init.get("initial_artifacts", [])]
        env0 = {p: float(init["initial_env"][p]) for p in ENV_PARAM_NAMES}
        n_rings = len(rings0)
        ny, nx = image.shape
        X_full, Y_full = meshgrid_xy(nx, ny)
        weights_full = _weight_map(image, rings0, env0, cfg)
        stride = max(1, int(cfg.d5_fit_stride))
        image_fit = image[::stride, ::stride]
        X = X_full[::stride, ::stride]
        Y = Y_full[::stride, ::stride]
        weights = weights_full[::stride, ::stride]
        if n_rings == 0 and not cfg.d5_fit_background:
            model = np.zeros_like(image_fit)
            stats = _fit_statistics(image_fit, model, weights, 0, cfg)
            result = {"success": True, "message": "no parameters", "nfev": 0, "artifacts": [], "env": env0, "fit_statistics": stats}
        else:
            x0 = rings_to_vector(rings0, env0, cfg.d5_fit_background)
            lo, hi = bounds_to_vectors(n_rings, cfg.fit_bounds, cfg.d5_fit_background)
            x0 = np.clip(x0, lo + 1e-10, hi - 1e-10)
            opt = least_squares(
                _residual,
                x0=x0,
                bounds=(lo, hi),
                args=(X, Y, image_fit, weights, n_rings, cfg.d5_fit_background, env0, cfg),
                method="trf",
                loss=cfg.d5_loss,
                f_scale=max(float(cfg.d5_f_scale), 1e-9),
                x_scale="jac",
                max_nfev=cfg.d5_max_nfev,
            )
            rings, env = vector_to_rings(opt.x, n_rings, cfg.d5_fit_background, env0)
            model_fit = smbh_model_image(X, Y, rings, env)
            stats = _fit_statistics(image_fit, model_fit, weights, len(opt.x), cfg)
            covar_available, stderr_vec = _estimate_lsq_covariance(opt)
            stderr_artifacts, stderr_env = _split_stderr(stderr_vec, n_rings, cfg.d5_fit_background)
            result = {
                "success": bool(opt.success),
                "message": str(opt.message),
                "nfev": int(opt.nfev),
                "cost": float(opt.cost),
                "artifacts": rings,
                "env": env,
                "fit_statistics": stats,
                "stderr_artifacts": stderr_artifacts,
                "stderr_env": stderr_env,
                "covar_available": bool(covar_available),
                "stderr_note": "linearized least-squares covariance; approximate for robust losses",
                "active_initial_n": n_rings,
                "loss": cfg.d5_loss,
                "residual_weighting": cfg.d5_residual_weighting,
                "fit_stride": stride,
            }
        out_dir = paths["d5"] / image_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.save_arrays:
            model = smbh_model_image(X_full, Y_full, result.get("artifacts", []), result.get("env", env0))
            np.savez_compressed(out_dir / "d5_fit_arrays.npz", model=model.astype(np.float32), residual=(model - image).astype(np.float32), weights=weights_full.astype(np.float32))
            result["arrays_path"] = str(out_dir / "d5_fit_arrays.npz")
        dump_json(out_dir / "d5_fit.json", {"image_id": image_id, **result})
        metric.update({"fit_n": len(result.get("artifacts", [])), "nfev": int(result.get("nfev", 0)), "r2": result.get("fit_statistics", {}).get("r2")})
    write_worker_metric(cfg.out, metric)
    return {"image_id": image_id, "status": "ok", "fit_n": metric.get("fit_n")}
