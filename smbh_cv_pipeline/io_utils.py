from __future__ import annotations

import json
import os
import time
import resource
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import psutil


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.integer):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return to_builtin(asdict(obj))
    return obj


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_builtin(data), f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")


def run_paths(out: str | Path) -> Dict[str, Path]:
    root = Path(out)
    paths = {
        "root": root,
        "manifest": root / "manifest.json",
        "config": root / "config_used.json",
        "images": root / "images",
        "d1": root / "D1_preprocess",
        "d2": root / "D2_hough",
        "d3": root / "D3_candidates",
        "d4": root / "D4_initialization",
        "d5": root / "D5_fit",
        "d6": root / "D6_prune",
        "stats": root / "statistics",
        "viz": root / "visualization",
        "runtime": root / "runtime",
    }
    for p in paths.values():
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    return paths


def image_ids_from_manifest(out: str | Path) -> List[str]:
    manifest = read_json(Path(out) / "manifest.json")
    return [item["image_id"] for item in manifest["items"]]


def process_memory_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))


def peak_rss_mb() -> float:
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB; macOS reports bytes. This environment is Linux.
    return float(val / 1024.0)


@contextmanager
def measured(stage: str, image_id: str, extra: Optional[Dict[str, Any]] = None):
    t0 = time.perf_counter()
    rss0 = process_memory_mb()
    peak0 = peak_rss_mb()
    status = "ok"
    err = None
    payload: Dict[str, Any] = {}
    try:
        yield payload
    except Exception as exc:
        status = "failed"
        err = repr(exc)
        raise
    finally:
        t1 = time.perf_counter()
        rss1 = process_memory_mb()
        peak1 = peak_rss_mb()
        payload.update({
            "stage": stage,
            "image_id": image_id,
            "status": status,
            "error": err,
            "wall_time_s": t1 - t0,
            "rss_start_mb": rss0,
            "rss_end_mb": rss1,
            "rss_delta_mb": rss1 - rss0,
            "peak_rss_mb": peak1,
            "peak_rss_delta_mb": max(0.0, peak1 - peak0),
            "pid": os.getpid(),
        })
        if extra:
            payload.update(extra)


def write_worker_metric(out: str | Path, row: Dict[str, Any]) -> None:
    append_jsonl(Path(out) / "runtime" / "per_image_runtime_memory.jsonl", row)


def clean_run_dir(out: str | Path) -> None:
    root = Path(out)
    if not root.exists():
        return
    import shutil
    for name in ["manifest.json", "config_used.json", "images", "D1_preprocess", "D2_hough", "D3_candidates", "D4_initialization", "D5_fit", "D6_prune", "statistics", "visualization", "runtime"]:
        p = root / name
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
