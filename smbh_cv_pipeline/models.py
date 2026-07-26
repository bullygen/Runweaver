from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ENV_PARAM_NAMES, PARAM_NAMES


@dataclass
class RingParams:
    A: float
    R: float
    sigma: float
    B: float
    phi: float
    x0: float
    y0: float

    def asdict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in PARAM_NAMES}


@dataclass
class JetParams:
    j_A: float
    j_sx: float
    j_sy: float
    j_phi: float
    j_x0: float
    j_y0: float
    noise_sigma: float = 0.0

    def asdict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in ENV_PARAM_NAMES} | {"noise_sigma": float(self.noise_sigma)}


def meshgrid_xy(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:ny, 0:nx]
    return x.astype(np.float64), y.astype(np.float64)


def angle_diff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def ring_model(X: np.ndarray, Y: np.ndarray, theta: Dict[str, float]) -> np.ndarray:
    A = float(theta["A"])
    R = float(theta["R"])
    sigma = max(float(theta["sigma"]), 1e-9)
    B = float(theta["B"])
    phi = float(theta["phi"])
    x0 = float(theta["x0"])
    y0 = float(theta["y0"])
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    return A * (1.0 - B * np.cos(angle - phi)) / (2.0 * np.pi) * np.exp(-((r - R) ** 2) / (sigma ** 2))


def background_model(X: np.ndarray, Y: np.ndarray, theta_g: Optional[Dict[str, float]]) -> np.ndarray:
    if theta_g is None:
        return np.zeros_like(X, dtype=np.float64)
    A = float(theta_g.get("j_A", 0.0))
    sx = max(float(theta_g.get("j_sx", 1.0)), 1e-9)
    sy = max(float(theta_g.get("j_sy", 1.0)), 1e-9)
    phi = float(theta_g.get("j_phi", 0.0))
    x0 = float(theta_g.get("j_x0", X.shape[1] / 2.0))
    y0 = float(theta_g.get("j_y0", X.shape[0] / 2.0))
    c = np.cos(phi)
    s = np.sin(phi)
    dx = X - x0
    dy = Y - y0
    xp = c * dx + s * dy
    yp = -s * dx + c * dy
    return A * np.exp(-0.5 * ((xp / sx) ** 2 + (yp / sy) ** 2))


def smbh_model_image(X: np.ndarray, Y: np.ndarray, rings: Sequence[Dict[str, float]], env: Optional[Dict[str, float]] = None) -> np.ndarray:
    image = background_model(X, Y, env)
    for ring in rings:
        image = image + ring_model(X, Y, ring)
    return image


def random_ring(rng: np.random.Generator, bounds: Dict[str, Sequence[float]]) -> Dict[str, float]:
    return {p: float(rng.uniform(*bounds[p])) for p in PARAM_NAMES}


def random_env(rng: np.random.Generator, bounds: Dict[str, Sequence[float]], noise_sigma: float) -> Dict[str, float]:
    d = {p: float(rng.uniform(*bounds[p])) for p in ENV_PARAM_NAMES}
    d["noise_sigma"] = float(noise_sigma)
    return d


def clamp_params(theta: Dict[str, float], bounds: Dict[str, Sequence[float]], names: Sequence[str]) -> Dict[str, float]:
    out = {}
    for p in names:
        lo, hi = bounds[p]
        out[p] = float(np.clip(float(theta.get(p, (lo + hi) / 2.0)), lo, hi))
    return out


def rings_to_vector(rings: Sequence[Dict[str, float]], env: Optional[Dict[str, float]], fit_background: bool) -> np.ndarray:
    vals: List[float] = []
    for ring in rings:
        vals.extend(float(ring[p]) for p in PARAM_NAMES)
    if fit_background and env is not None:
        vals.extend(float(env[p]) for p in ENV_PARAM_NAMES)
    return np.asarray(vals, dtype=np.float64)


def vector_to_rings(vec: np.ndarray, n_rings: int, fit_background: bool, env_default: Optional[Dict[str, float]] = None) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rings: List[Dict[str, float]] = []
    k = 0
    for _ in range(n_rings):
        ring = {}
        for p in PARAM_NAMES:
            ring[p] = float(vec[k])
            k += 1
        rings.append(ring)
    env = dict(env_default or {})
    if fit_background:
        for p in ENV_PARAM_NAMES:
            env[p] = float(vec[k])
            k += 1
    return rings, env


def bounds_to_vectors(n_rings: int, bounds: Dict[str, Sequence[float]], fit_background: bool) -> Tuple[np.ndarray, np.ndarray]:
    lo: List[float] = []
    hi: List[float] = []
    for _ in range(n_rings):
        for p in PARAM_NAMES:
            lo.append(float(bounds[p][0]))
            hi.append(float(bounds[p][1]))
    if fit_background:
        for p in ENV_PARAM_NAMES:
            lo.append(float(bounds[p][0]))
            hi.append(float(bounds[p][1]))
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
