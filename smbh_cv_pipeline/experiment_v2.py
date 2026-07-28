from __future__ import annotations

"""Поэтапный план article_v2 с повторным использованием выходов D3 и D5."""

import argparse
import csv
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc

from .cli import STAGE_FUNCS, _pool_map, setup_logging
from .config import PipelineConfig
from .experiment_runner import (
    _STAGE_OUTPUT,
    _fingerprint,
    _load_config,
    _missing_stage_image_ids,
    _prepare_run,
    _select_dataset_items,
)
from .io_utils import dump_json, read_json, run_paths
from .statistics import run_statistics


STAGES = ("d1", "d2", "d3", "d4", "d5", "d6")


def _finite(values: Iterable[Any]) -> List[float]:
    result: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _sample_dimension(definition: Mapping[str, Any], unit: float) -> Any:
    kind = str(definition["type"])
    unit = float(np.clip(unit, 0.0, np.nextafter(1.0, 0.0)))
    if kind == "float":
        return float(definition["low"] + unit * (definition["high"] - definition["low"]))
    if kind == "log_float":
        low = math.log(float(definition["low"]))
        high = math.log(float(definition["high"]))
        return float(math.exp(low + unit * (high - low)))
    if kind == "int":
        low, high = int(definition["low"]), int(definition["high"])
        return int(min(high, low + math.floor(unit * (high - low + 1))))
    if kind == "log_int":
        low = math.log(float(definition["low"]))
        high = math.log(float(definition["high"]))
        return int(np.clip(round(math.exp(low + unit * (high - low))), int(definition["low"]), int(definition["high"])))
    if kind == "choice":
        values = list(definition["values"])
        return values[min(len(values) - 1, int(unit * len(values)))]
    raise ValueError(f"Неизвестный тип области поиска: {kind}")


def _to_unit(definition: Mapping[str, Any], value: Any) -> float:
    kind = str(definition["type"])
    if kind in ("float", "int"):
        low, high = float(definition["low"]), float(definition["high"])
        return float(np.clip((float(value) - low) / max(high - low, 1e-12), 0.0, 1.0))
    if kind in ("log_float", "log_int"):
        low, high = math.log(float(definition["low"])), math.log(float(definition["high"]))
        return float(np.clip((math.log(max(float(value), 1e-300)) - low) / max(high - low, 1e-12), 0.0, 1.0))
    if kind == "choice":
        values = list(definition["values"])
        try:
            index = values.index(value)
        except ValueError:
            index = 0
        return float((index + 0.5) / len(values))
    raise ValueError(f"Неизвестный тип области поиска: {kind}")


def _active(definition: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    conditions = definition.get("active_when", {})
    for name, allowed in conditions.items():
        values = allowed if isinstance(allowed, list) else [allowed]
        if params.get(name) not in values:
            return False
    return True


def _apply_relations(params: Dict[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(params)
    for target, source in spec.get("derived_equal", {}).items():
        if source in result:
            result[target] = result[source]
    if "d3_max_candidates" in result:
        result["d4_max_init_candidates"] = int(result["d3_max_candidates"])
    return result


def _sobol_units(count: int, dimensions: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, dimensions), dtype=float)
    if dimensions <= 0:
        return np.zeros((count, 0), dtype=float)
    exponent = int(math.ceil(math.log2(count)))
    return qmc.Sobol(d=dimensions, scramble=True, seed=seed).random_base2(exponent)[:count]


def _candidate(
    prefix: str,
    params: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    eligible: bool = True,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    identity = {"params": params, "metadata": metadata, "prefix": prefix}
    return {
        "candidate_id": f"{prefix}_{_fingerprint(identity)[:10]}",
        "params": dict(params),
        "metadata": metadata,
        "eligible_for_selection": bool(eligible),
    }


def _read_ranking(spec: Mapping[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    path = Path(spec["output_root"]) / phase_name / "ranking.json"
    if not path.is_file():
        raise FileNotFoundError(f"Сначала завершите и обобщите этап {phase_name}: {path}")
    return list(read_json(path).get("ranking", []))


def _selectable(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for row in rows:
        try:
            score_is_finite = math.isfinite(float(row.get("score")))
        except (TypeError, ValueError):
            score_is_finite = False
        if (
            bool(row.get("complete"))
            and bool(row.get("eligible_for_selection", True))
            and score_is_finite
        ):
            result.append(dict(row))
    return result


def _require_source_count(rows: Sequence[Mapping[str, Any]], minimum: int, source: str) -> None:
    if len(rows) < minimum:
        raise RuntimeError(
            f"Для этапа {source} готово {len(rows)} допустимых конфигураций, требуется не менее {minimum}"
        )


def _base_params_from_frozen(phase: Mapping[str, Any]) -> Dict[str, Any]:
    path = phase.get("base_frozen")
    if not path:
        return {}
    frozen_path = Path(path)
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Сначала создайте замороженную конфигурацию: {frozen_path}")
    return dict(read_json(frozen_path).get("params", {}))


def _sobol_candidates(spec: Mapping[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    phase = spec["phases"][phase_name]
    space = spec["search_spaces"][phase["search_space"]]
    names = list(space)
    count = int(phase["candidate_count"])
    units = _sobol_units(count, len(names), int(phase.get("candidate_seed", spec["search_seed"])))
    base_params = _base_params_from_frozen(phase)
    result: List[Dict[str, Any]] = []
    if phase.get("include_reference", False):
        result.append(_candidate(
            f"{phase_name}_reference",
            _apply_relations(base_params, spec),
            metadata={"kind": "reference"},
            eligible=False,
        ))
    for index, row in enumerate(units):
        params = dict(base_params)
        for column, name in enumerate(names):
            definition = space[name]
            if not _active(definition, params):
                continue
            params[name] = _sample_dimension(definition, float(row[column]))
        params = _apply_relations(params, spec)
        result.append(_candidate(
            f"{phase_name}_{index + 1:03d}",
            params,
            metadata={"kind": "sobol", "index": index},
        ))
    return result


def _adaptive_candidates(spec: Mapping[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    phase = spec["phases"][phase_name]
    source_phase = str(phase["source_phase"])
    source = _selectable(_read_ranking(spec, source_phase))
    _require_source_count(source, int(phase.get("minimum_source_candidates", 1)), source_phase)
    elite = source[: int(phase["elite_count"])]
    space = spec["search_spaces"][phase["search_space"]]
    names = list(space)
    count = int(phase["candidate_count"])
    seed = int(phase.get("candidate_seed", spec["search_seed"]))
    rng = np.random.default_rng(seed)
    exploratory = _sobol_units(count, len(names), seed + 7919)
    base_params = _base_params_from_frozen(phase)
    result: List[Dict[str, Any]] = []
    for index in range(count):
        anchor = elite[int(rng.integers(0, len(elite)))]
        params = dict(base_params)
        for column, name in enumerate(names):
            definition = space[name]
            if not _active(definition, params):
                continue
            if definition["type"] == "choice":
                values = list(definition["values"])
                counts = np.ones(len(values), dtype=float)
                for row in elite:
                    if row["params"].get(name) in values:
                        counts[values.index(row["params"][name])] += 1.0
                probs = counts / counts.sum()
                params[name] = values[int(rng.choice(len(values), p=probs))]
                continue
            elite_units = np.asarray([
                _to_unit(definition, row["params"].get(name, _sample_dimension(definition, 0.5)))
                for row in elite
            ])
            center = _to_unit(
                definition,
                anchor["params"].get(name, _sample_dimension(definition, 0.5)),
            )
            spread = max(float(np.std(elite_units, ddof=1)) if len(elite_units) > 1 else 0.0, 0.06)
            if rng.random() < float(phase.get("exploration_fraction", 0.20)):
                unit = float(exploratory[index, column])
            else:
                unit = float(np.clip(rng.normal(center, spread), 0.0, np.nextafter(1.0, 0.0)))
            params[name] = _sample_dimension(definition, unit)
        params = _apply_relations(params, spec)
        result.append(_candidate(
            f"{phase_name}_{index + 1:03d}",
            params,
            metadata={
                "kind": "adaptive_distribution",
                "source_phase": source_phase,
                "anchor_id": anchor["candidate_id"],
            },
        ))
    return result


def _top_union_candidates(spec: Mapping[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    phase = spec["phases"][phase_name]
    rows: List[Dict[str, Any]] = []
    for source_phase in phase["source_phases"]:
        for row in _selectable(_read_ranking(spec, str(source_phase))):
            rows.append(dict(row) | {"source_phase": str(source_phase)})
    rows.sort(
        key=lambda row: (
            bool(row.get("feasible")),
            -int(row.get("pareto_rank", 10**9)),
            float(row.get("score", float("-inf"))),
        ),
        reverse=True,
    )
    seen: set[str] = set()
    result: List[Dict[str, Any]] = []
    for row in rows:
        fingerprint = _fingerprint(row["params"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(_candidate(
            f"{phase_name}_{len(result) + 1:03d}",
            row["params"],
            metadata={
                "kind": "promoted",
                "source_phase": row["source_phase"],
                "source_candidate_id": row["candidate_id"],
            },
        ))
        if len(result) >= int(phase["top_k"]):
            break
    _require_source_count(result, int(phase["top_k"]), phase_name)
    return result


def _top_candidates(spec: Mapping[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    phase = spec["phases"][phase_name]
    source_phase = str(phase["source_phase"])
    rows = _selectable(_read_ranking(spec, source_phase))
    _require_source_count(rows, int(phase["top_k"]), source_phase)
    return [
        _candidate(
            f"{phase_name}_{index + 1:03d}",
            row["params"],
            metadata={
                "kind": "promoted",
                "source_phase": source_phase,
                "source_candidate_id": row["candidate_id"],
            },
        )
        for index, row in enumerate(rows[: int(phase["top_k"])])
    ]


def _local_candidates(spec: Mapping[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    phase = spec["phases"][phase_name]
    source_phase = str(phase["source_phase"])
    anchors = _selectable(_read_ranking(spec, source_phase))
    _require_source_count(anchors, int(phase["top_k"]), source_phase)
    space = spec["search_spaces"][phase["search_space"]]
    result: List[Dict[str, Any]] = []
    for anchor_index, anchor in enumerate(anchors[: int(phase["top_k"])]):
        anchor_key = f"a{anchor_index + 1:02d}"
        result.append(_candidate(
            f"{phase_name}_{anchor_key}_center",
            anchor["params"],
            metadata={
                "kind": "local_center",
                "anchor_key": anchor_key,
                "source_candidate_id": anchor["candidate_id"],
            },
        ))
        for name in phase["dimensions"]:
            definition = space[name]
            if definition["type"] == "choice" or not _active(definition, anchor["params"]):
                continue
            center = _to_unit(
                definition,
                anchor["params"].get(name, _sample_dimension(definition, 0.5)),
            )
            for step in phase["steps"]:
                for direction in (-1, 1):
                    params = dict(anchor["params"])
                    params[name] = _sample_dimension(
                        definition,
                        float(np.clip(center + direction * float(step), 0.0, np.nextafter(1.0, 0.0))),
                    )
                    params = _apply_relations(params, spec)
                    result.append(_candidate(
                        f"{phase_name}_{anchor_key}_{name}",
                        params,
                        metadata={
                            "kind": "local_offset",
                            "anchor_key": anchor_key,
                            "source_candidate_id": anchor["candidate_id"],
                            "dimension": name,
                            "step": float(step),
                            "direction": int(direction),
                        },
                    ))
    return result


def _fixed_candidates(spec: Mapping[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    phase = spec["phases"][phase_name]
    return [_candidate(
        f"{phase_name}_reference",
        _apply_relations(_base_params_from_frozen(phase), spec),
        metadata={"kind": "fixed"},
    )]


def _gate_is_open(spec: Mapping[str, Any], phase_name: str) -> None:
    requirement = spec["phases"][phase_name].get("requires_gate")
    if not requirement:
        return
    source_phase = str(requirement["phase"])
    rows = _selectable(_read_ranking(spec, source_phase))
    feasible_count = sum(bool(row.get("feasible")) for row in rows)
    minimum = int(requirement.get("minimum_feasible", 1))
    if feasible_count < minimum:
        raise RuntimeError(
            f"Переход к {phase_name} запрещен: на этапе {source_phase} "
            f"пороги прошли {feasible_count} конфигураций, требуется {minimum}"
        )


def _generate_candidates(spec: Mapping[str, Any], phase_name: str) -> List[Dict[str, Any]]:
    if phase_name not in spec["phases"]:
        raise KeyError(f"Неизвестный этап: {phase_name}")
    _gate_is_open(spec, phase_name)
    strategy = str(spec["phases"][phase_name]["candidate_strategy"])
    if strategy == "sobol":
        return _sobol_candidates(spec, phase_name)
    if strategy == "adaptive":
        return _adaptive_candidates(spec, phase_name)
    if strategy == "top_union":
        return _top_union_candidates(spec, phase_name)
    if strategy == "top":
        return _top_candidates(spec, phase_name)
    if strategy == "local":
        return _local_candidates(spec, phase_name)
    if strategy == "fixed":
        return _fixed_candidates(spec, phase_name)
    if strategy == "frozen":
        return []
    raise ValueError(f"Неизвестный способ построения конфигураций: {strategy}")


def candidate_plan_path(spec: Mapping[str, Any], phase_name: str) -> Path:
    return Path(spec["output_root"]) / phase_name / "candidate_plan.json"


def plan_phase(
    spec: Mapping[str, Any],
    phase_name: str,
    *,
    force: bool = False,
) -> List[Dict[str, Any]]:
    path = candidate_plan_path(spec, phase_name)
    if path.is_file() and not force:
        existing = read_json(path)
        if existing.get("spec_fingerprint") != _fingerprint(spec):
            raise RuntimeError(
                f"План {path} создан по другой версии файла параметров; "
                "до начала расчетов пересоздайте его командой plan --force"
            )
        return list(existing["candidates"])
    candidates = _generate_candidates(spec, phase_name)
    payload = {
        "version": "article-v2-candidate-plan-v1",
        "phase": phase_name,
        "spec_fingerprint": _fingerprint(spec),
        "phase_fingerprint": _fingerprint(spec["phases"][phase_name]),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    dump_json(path, payload)
    return candidates


def _frozen_candidate(path: str | Path) -> Dict[str, Any]:
    data = read_json(Path(path))
    return {
        "candidate_id": str(data["candidate_id"]),
        "params": dict(data["params"]),
        "metadata": {"kind": "frozen", "source_phase": data.get("source_phase")},
        "eligible_for_selection": True,
    }


def _source_run_from_frozen(
    spec: Mapping[str, Any],
    frozen_path: str | Path,
    algorithm_seed: int,
) -> Path:
    frozen = read_json(Path(frozen_path))
    source_phase = str(frozen["source_phase"])
    candidate_id = str(frozen["candidate_id"])
    return Path(spec["output_root"]) / source_phase / candidate_id / f"algorithm_seed_{algorithm_seed}"


def _link_stage_prefix(
    spec: Mapping[str, Any],
    phase: Mapping[str, Any],
    run_dir: Path,
    image_ids: Sequence[str],
    algorithm_seed: int,
) -> Dict[str, Any] | None:
    reuse = phase.get("reuse_prefix")
    if not reuse:
        return None
    source_run = _source_run_from_frozen(spec, reuse["frozen"], algorithm_seed)
    if not (source_run / "run_signature.json").is_file():
        raise FileNotFoundError(f"Нет исходного прогона для повторного использования: {source_run}")
    source_manifest = read_json(source_run / "manifest.json")
    source_ids = [str(row["image_id"]) for row in source_manifest["items"]]
    missing = sorted(set(image_ids) - set(source_ids))
    if missing:
        raise RuntimeError(
            f"Исходный прогон {source_run} не содержит {len(missing)} требуемых изображений"
        )
    source_paths = run_paths(source_run)
    target_paths = run_paths(run_dir)
    through = str(reuse["through"])
    last_index = STAGES.index(through)
    linked: List[str] = []
    for stage in STAGES[: last_index + 1]:
        source_stage = source_paths[stage]
        target_stage = target_paths[stage]
        output_key, output_name = _STAGE_OUTPUT[stage]
        if output_key != stage:
            raise RuntimeError(f"Нарушено соответствие каталога для {stage}")
        absent_outputs = [
            image_id
            for image_id in image_ids
            if not (source_stage / image_id / output_name).is_file()
        ]
        if absent_outputs:
            raise FileNotFoundError(
                f"В исходном прогоне нет {len(absent_outputs)} выходов {stage.upper()}: {source_run}"
            )
        if source_ids == list(image_ids):
            if target_stage.is_symlink():
                linked.append(stage)
                continue
            if target_stage.is_dir() and not any(target_stage.iterdir()):
                target_stage.rmdir()
                target_stage.symlink_to(source_stage.resolve(), target_is_directory=True)
                linked.append(stage)
                continue
        for image_id in image_ids:
            source = source_stage / image_id
            target = target_stage / image_id
            if not source.is_dir():
                raise FileNotFoundError(f"Нет сохраненного выхода {stage.upper()}: {source}")
            if target.is_symlink():
                continue
            if target.exists():
                raise RuntimeError(f"Целевой каталог уже занят и не является ссылкой: {target}")
            target.symlink_to(source.resolve(), target_is_directory=True)
        linked.append(stage)
    return {
        "source_run": str(source_run),
        "through": through,
        "linked_stages": linked,
        "truth_files_used_by_linking": False,
    }


def _d3_match_counts(
    candidates: Sequence[Mapping[str, Any]],
    truth: Sequence[Mapping[str, Any]],
    cfg: PipelineConfig,
) -> tuple[int, int, int]:
    if not truth:
        return 0, len(candidates), 0
    if not candidates:
        return 0, 0, len(truth)
    cost = np.zeros((len(truth), len(candidates)), dtype=float)
    for i, ring in enumerate(truth):
        for j, candidate in enumerate(candidates):
            center = np.hypot(
                float(candidate["x0"]) - float(ring["x0"]),
                float(candidate["y0"]) - float(ring["y0"]),
            )
            radius = abs(float(candidate["R"]) - float(ring["R"]))
            cost[i, j] = np.hypot(
                center / max(cfg.max_center_distance_px, 1e-9),
                radius / max(cfg.max_center_distance_px, 1e-9),
            )
    rows, columns = linear_sum_assignment(cost)
    true_positive = 0
    for i, j in zip(rows, columns):
        ring = truth[int(i)]
        candidate = candidates[int(j)]
        center = np.hypot(
            float(candidate["x0"]) - float(ring["x0"]),
            float(candidate["y0"]) - float(ring["y0"]),
        )
        radius = abs(float(candidate["R"]) - float(ring["R"]))
        true_positive += int(
            center <= cfg.max_center_distance_px
            and radius <= cfg.max_center_distance_px
        )
    return true_positive, len(candidates) - true_positive, len(truth) - true_positive


def _d3_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeats: int = 1000,
    seed: int = 38191,
) -> Dict[str, List[float]]:
    if not rows:
        return {"recall": [float("nan"), float("nan")]}
    counts = np.asarray([[row["tp"], row["fn"]] for row in rows], dtype=int)
    rng = np.random.default_rng(seed)
    recall: List[float] = []
    for _ in range(repeats):
        total = counts[rng.integers(0, len(counts), size=len(counts))].sum(axis=0)
        denominator = int(total[0] + total[1])
        if denominator:
            recall.append(float(total[0] / denominator))
    if not recall:
        return {"recall": [float("nan"), float("nan")]}
    return {"recall": [float(np.quantile(recall, 0.025)), float(np.quantile(recall, 0.975))]}


def _runtime_per_image(run_dir: Path, image_ids: Sequence[str], stages: Sequence[str]) -> float:
    path = run_dir / "runtime" / "per_image_runtime_memory.jsonl"
    if not path.is_file():
        return float("nan")
    latest: Dict[tuple[str, str], float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                key = (str(row.get("stage", "")).lower(), str(row.get("image_id", "")))
                value = float(row.get("wall_time_s"))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if key[0] in stages and key[1] in image_ids and math.isfinite(value):
                latest[key] = value
    if not latest:
        return float("nan")
    return float(sum(latest.values()) / max(len(image_ids), 1))


def run_d3_statistics(cfg: PipelineConfig) -> Dict[str, Any]:
    paths = run_paths(cfg.out)
    manifest = read_json(paths["manifest"])
    image_ids = [str(row["image_id"]) for row in manifest["items"]]
    per_image: List[Dict[str, Any]] = []
    for image_id in image_ids:
        truth_payload = read_json(paths["images"] / image_id / "truth.json")
        truth = list(truth_payload.get("true_artifacts", []))
        candidates = list(read_json(paths["d3"] / image_id / "d3_candidates.json").get("candidates", []))
        tp, fp, fn = _d3_match_counts(candidates, truth, cfg)
        per_image.append({
            "image_id": image_id,
            "true_n": len(truth),
            "candidate_n": len(candidates),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
    tp = int(sum(row["tp"] for row in per_image))
    fp = int(sum(row["fp"] for row in per_image))
    fn = int(sum(row["fn"] for row in per_image))
    positive = [row for row in per_image if row["true_n"] > 0]
    null = [row for row in per_image if row["true_n"] == 0]
    denominator = tp + fn
    ci = _d3_bootstrap(per_image)
    summary = {
        "metric_scope": "D1-D3 proposals",
        "truth_access_boundary": "Истинные параметры читаются только здесь, после завершения D1-D3",
        "n_images_total": len(per_image),
        "n_positive_images": len(positive),
        "n_null_images": len(null),
        "d3_tp": tp,
        "d3_fp": fp,
        "d3_fn": fn,
        "d3_recall": float(tp / denominator) if denominator else float("nan"),
        "d3_recall_ci95": ci["recall"],
        "positive_images_without_candidates_fraction": (
            float(sum(row["candidate_n"] == 0 for row in positive) / len(positive))
            if positive
            else float("nan")
        ),
        "mean_candidate_n_positive": (
            float(np.mean([row["candidate_n"] for row in positive]))
            if positive
            else float("nan")
        ),
        "null_candidates_per_image": (
            float(sum(row["candidate_n"] for row in null) / len(null))
            if null
            else float("nan")
        ),
        "d1_d3_runtime_s_per_image": _runtime_per_image(Path(cfg.out), image_ids, ("d1", "d2", "d3")),
    }
    stats_dir = paths["stats"]
    with open(stats_dir / "per_image_d3_metrics.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]) if per_image else ["image_id"])
        writer.writeheader()
        writer.writerows(per_image)
    dump_json(stats_dir / "summary.json", summary)
    return summary


def _run_one(
    spec: Mapping[str, Any],
    phase_name: str,
    candidate: Mapping[str, Any],
    algorithm_seed: int,
    *,
    restart: bool = False,
) -> Dict[str, Any]:
    phase = spec["phases"][phase_name]
    dataset_split = Path(spec["dataset_root"]) / phase["split"]
    items = _select_dataset_items(dataset_split, phase)
    image_ids = [str(item["image_id"]) for item in items]
    candidate_id = str(candidate["candidate_id"])
    run_dir = Path(spec["output_root"]) / phase_name / candidate_id / f"algorithm_seed_{algorithm_seed}"
    cfg = _load_config(spec["base_config"])
    for key, value in candidate.get("params", {}).items():
        if not hasattr(cfg, key):
            raise ValueError(f"Конфигурация содержит неизвестный параметр: {key}")
        setattr(cfg, key, value)
    for key, value in phase.get("config_overrides", {}).items():
        if not hasattr(cfg, key):
            raise ValueError(f"Этап содержит неизвестный параметр: {key}")
        setattr(cfg, key, value)
    cfg.out = str(run_dir)
    cfg.seed = int(algorithm_seed)
    cfg.n_images = len(items)
    cfg.make_plots = bool(phase.get("make_plots", False))
    cfg.save_arrays = bool(phase.get("save_arrays", False))
    dataset_manifest = read_json(dataset_split / "manifest.json")
    signature_payload = {
        "version": spec["version"],
        "phase": phase_name,
        "candidate_id": candidate_id,
        "candidate_params": candidate.get("params", {}),
        "algorithm_seed": int(algorithm_seed),
        "dataset_plan_hash": dataset_manifest.get("plan_hash"),
        "selected_images": [[item["image_id"], item.get("image_sha256")] for item in items],
        "pipeline_config": cfg.to_dict(),
        "phase_config": phase,
    }
    signature = {"fingerprint": _fingerprint(signature_payload), "payload": signature_payload}
    _prepare_run(dataset_split, run_dir, items, cfg, restart=restart, signature=signature)
    reuse_metadata = _link_stage_prefix(spec, phase, run_dir, image_ids, algorithm_seed)
    workers = spec.get("workers", {})
    for stage in phase["run_stages"]:
        missing = _missing_stage_image_ids(cfg, stage, image_ids)
        if not missing:
            continue
        logging.info(
            "Этап %s, конфигурация=%s, зерно=%d: осталось %d из %d изображений",
            stage.upper(),
            candidate_id,
            algorithm_seed,
            len(missing),
            len(image_ids),
        )
        _pool_map(
            stage,
            STAGE_FUNCS[stage],
            cfg,
            missing,
            int(workers.get(stage, 1)),
            total_items=len(image_ids),
        )
    summary_path = run_dir / "statistics" / "summary.json"
    if not summary_path.is_file():
        if phase["metric_mode"] == "d3":
            run_d3_statistics(cfg)
        elif phase["metric_mode"] == "final":
            run_statistics(cfg)
        else:
            raise ValueError(f"Неизвестный способ расчета показателей: {phase['metric_mode']}")
    summary = read_json(summary_path)
    dump_json(run_dir / "experiment_metadata.json", {
        "phase": phase_name,
        "candidate": candidate,
        "algorithm_seed": int(algorithm_seed),
        "dataset_split": str(dataset_split),
        "n_images": len(items),
        "run_stages": list(phase["run_stages"]),
        "reused_prefix": reuse_metadata,
        "truth_access": (
            "После D1-D3, только при расчете показателей"
            if phase["metric_mode"] == "d3"
            else "После D6, только при расчете показателей"
        ),
        "config_fingerprint": _fingerprint(cfg.to_dict()),
    })
    return summary


def _d3_objective(
    summaries: Sequence[Mapping[str, Any]],
    objective: Mapping[str, Any],
) -> Dict[str, Any]:
    recall = _finite(row.get("d3_recall") for row in summaries)
    recall_low = _finite((row.get("d3_recall_ci95") or [None])[0] for row in summaries)
    null_count = _finite(row.get("null_candidates_per_image") for row in summaries)
    missing = _finite(row.get("positive_images_without_candidates_fraction") for row in summaries)
    runtime = _finite(row.get("d1_d3_runtime_s_per_image") for row in summaries)
    if not recall or not recall_low:
        return {"score": float("-inf"), "feasible": False}
    worst_recall = min(recall)
    worst_low = min(recall_low)
    max_null = max(null_count) if null_count else float("inf")
    max_missing = max(missing) if missing else float("inf")
    max_runtime = max(runtime) if runtime else 0.0
    feasible = (
        worst_recall >= float(objective["recall_floor"])
        and max_null <= float(objective["null_candidates_ceiling"])
        and max_missing <= float(objective["positive_missing_ceiling"])
    )
    return {
        "score": float(
            worst_low
            - float(objective["null_candidate_penalty"]) * max_null
            - float(objective["missing_positive_penalty"]) * max_missing
            - float(objective["runtime_penalty"]) * max_runtime
        ),
        "feasible": bool(feasible),
        "min_d3_recall": float(worst_recall),
        "worst_d3_recall_ci95_low": float(worst_low),
        "max_null_candidates_per_image": float(max_null),
        "max_positive_images_without_candidates_fraction": float(max_missing),
        "max_d1_d3_runtime_s_per_image": float(max_runtime),
    }


def _final_objective(
    summaries: Sequence[Mapping[str, Any]],
    objective: Mapping[str, Any],
) -> Dict[str, Any]:
    f1: List[float] = []
    f1_low: List[float] = []
    precision: List[float] = []
    for row in summaries:
        no_true_positive_with_truth = (
            int(row.get("article_tp", 0)) == 0
            and int(row.get("article_fn", 0)) > 0
        )
        no_detections_with_truth = (
            no_true_positive_with_truth
            and int(row.get("article_fp", 0)) == 0
        )
        f1_values = _finite([row.get("article_f1")])
        f1.extend(f1_values if f1_values else ([0.0] if no_true_positive_with_truth else []))
        interval = row.get("article_f1_ci95") or [None]
        low_values = _finite([interval[0] if interval else None])
        f1_low.extend(low_values if low_values else ([0.0] if no_true_positive_with_truth else []))
        precision_values = _finite([row.get("article_precision")])
        precision.extend(
            precision_values
            if precision_values
            else ([0.0] if no_detections_with_truth else [])
        )
    recall = _finite(row.get("article_recall") for row in summaries)
    null_fppi = _finite(row.get("null_fppi") for row in summaries)
    if not f1 or not f1_low or not precision or not recall:
        return {"score": float("-inf"), "feasible": False}
    mean_f1 = float(np.mean(f1))
    std_f1 = float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0
    worst_low = float(min(f1_low))
    min_precision = float(min(precision))
    min_recall = float(min(recall))
    max_null = float(max(null_fppi)) if null_fppi else float("inf")
    precision_shortfall = max(0.0, float(objective["precision_floor"]) - min_precision)
    recall_shortfall = max(0.0, float(objective["recall_floor"]) - min_recall)
    null_excess = max(0.0, max_null - float(objective["null_fppi_ceiling"]))
    feasible = precision_shortfall == 0.0 and recall_shortfall == 0.0 and null_excess == 0.0
    score = (
        worst_low
        - float(objective["seed_std_penalty"]) * std_f1
        - float(objective["null_fppi_penalty"]) * max_null
        - float(objective["precision_shortfall_penalty"]) * precision_shortfall
        - float(objective["recall_shortfall_penalty"]) * recall_shortfall
        - float(objective["null_excess_penalty"]) * null_excess
    )
    return {
        "score": float(score),
        "feasible": bool(feasible),
        "mean_f1": mean_f1,
        "std_f1_across_algorithm_seeds": std_f1,
        "worst_f1_ci95_low": worst_low,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "max_null_fppi": max_null,
        "precision_shortfall": precision_shortfall,
        "recall_shortfall": recall_shortfall,
        "null_fppi_excess": null_excess,
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any], mode: str) -> bool:
    if mode == "d3":
        fields = [
            ("worst_d3_recall_ci95_low", 1),
            ("max_null_candidates_per_image", -1),
            ("max_positive_images_without_candidates_fraction", -1),
            ("max_d1_d3_runtime_s_per_image", -1),
        ]
    else:
        fields = [
            ("worst_f1_ci95_low", 1),
            ("min_precision", 1),
            ("min_recall", 1),
            ("max_null_fppi", -1),
            ("std_f1_across_algorithm_seeds", -1),
        ]
    left_values: List[float] = []
    right_values: List[float] = []
    for field, direction in fields:
        try:
            lv, rv = float(left[field]), float(right[field])
        except (KeyError, TypeError, ValueError):
            return False
        if not (math.isfinite(lv) and math.isfinite(rv)):
            return False
        left_values.append(direction * lv)
        right_values.append(direction * rv)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _assign_pareto_ranks(rows: List[Dict[str, Any]], mode: str) -> None:
    remaining = {
        index
        for index, row in enumerate(rows)
        if bool(row.get("objective_valid"))
    }
    for index, row in enumerate(rows):
        if index not in remaining:
            row["pareto_rank"] = 10**9
    rank = 1
    while remaining:
        front = [
            index
            for index in remaining
            if not any(
                _dominates(rows[other], rows[index], mode)
                for other in remaining
                if other != index
            )
        ]
        if not front:
            front = list(remaining)
        for index in front:
            rows[index]["pareto_rank"] = rank
        remaining.difference_update(front)
        rank += 1


def _write_ranking_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = ["rank"]
    for row in rows:
        for key in row:
            if key not in fields and key not in ("params", "metadata"):
                fields.append(key)
    fields.extend(["params", "metadata"])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({
                **row,
                "rank": rank,
                "params": json.dumps(row.get("params", {}), sort_keys=True),
                "metadata": json.dumps(row.get("metadata", {}), sort_keys=True),
            })


def _write_local_sensitivity(
    root: Path,
    candidates: Sequence[Mapping[str, Any]],
    ranking: Sequence[Mapping[str, Any]],
) -> None:
    score_by_id = {str(row["candidate_id"]): float(row["score"]) for row in ranking}
    groups: Dict[tuple[str, str, float], Dict[int, float]] = {}
    centers: Dict[str, float] = {}
    for candidate in candidates:
        metadata = candidate.get("metadata", {})
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in score_by_id:
            continue
        if metadata.get("kind") == "local_center":
            centers[str(metadata["anchor_key"])] = score_by_id[candidate_id]
        elif metadata.get("kind") == "local_offset":
            key = (
                str(metadata["anchor_key"]),
                str(metadata["dimension"]),
                float(metadata["step"]),
            )
            groups.setdefault(key, {})[int(metadata["direction"])] = score_by_id[candidate_id]
    rows: List[Dict[str, Any]] = []
    for (anchor, dimension, step), values in sorted(groups.items()):
        if anchor not in centers or -1 not in values or 1 not in values:
            continue
        q_minus, q_zero, q_plus = values[-1], centers[anchor], values[1]
        rows.append({
            "anchor_key": anchor,
            "dimension": dimension,
            "h_normalized": step,
            "q_minus": q_minus,
            "q_zero": q_zero,
            "q_plus": q_plus,
            "first_difference": abs(q_plus - q_minus) / (2.0 * step),
            "second_difference": abs(q_plus - 2.0 * q_zero + q_minus) / (step * step),
        })
    dump_json(root / "local_sensitivity.json", {
        "formula_first": "abs(Q(x+h)-Q(x-h))/(2h)",
        "formula_second": "abs(Q(x+h)-2Q(x)+Q(x-h))/h^2",
        "rows": rows,
    })
    if rows:
        with open(root / "local_sensitivity.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize_phase(
    spec: Mapping[str, Any],
    phase_name: str,
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    phase = spec["phases"][phase_name]
    root = Path(spec["output_root"]) / phase_name
    objective_name = str(phase["objective"])
    objective = spec["objectives"][objective_name]
    expected_runs = len(phase["algorithm_seeds"])
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_dir = root / str(candidate["candidate_id"])
        summaries = [
            read_json(path)
            for path in sorted(candidate_dir.glob("algorithm_seed_*/statistics/summary.json"))
        ]
        if not summaries:
            continue
        values = (
            _d3_objective(summaries, objective)
            if phase["metric_mode"] == "d3"
            else _final_objective(summaries, objective)
        )
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "params": candidate.get("params", {}),
            "metadata": candidate.get("metadata", {}),
            "eligible_for_selection": bool(candidate.get("eligible_for_selection", True)),
            "n_runs": len(summaries),
            "expected_runs": expected_runs,
            "complete": len(summaries) == expected_runs,
            **values,
        })
        try:
            rows[-1]["objective_valid"] = math.isfinite(float(rows[-1]["score"]))
        except (TypeError, ValueError):
            rows[-1]["objective_valid"] = False
    _assign_pareto_ranks(rows, phase["metric_mode"])
    rows.sort(
        key=lambda row: (
            bool(row["complete"]),
            bool(row["eligible_for_selection"]),
            bool(row["objective_valid"]),
            bool(row["feasible"]),
            -int(row["pareto_rank"]),
            float(row["score"]),
        ),
        reverse=True,
    )
    result = {
        "phase": phase_name,
        "objective_name": objective_name,
        "objective": objective,
        "selection_order": [
            "полный прогон",
            "допуск к отбору",
            "определенность показателя",
            "прохождение порогов",
            "слой Парето",
            "непрерывная оценка",
        ],
        "ranking": rows,
    }
    dump_json(root / "ranking.json", result)
    _write_ranking_csv(root / "ranking.csv", rows)
    if phase["candidate_strategy"] == "local":
        _write_local_sensitivity(root, candidates, rows)
    return result


def freeze_best(
    spec: Mapping[str, Any],
    phase_name: str,
    output: str | Path,
    *,
    allow_gate_failure: bool = False,
) -> Dict[str, Any]:
    rows = _selectable(_read_ranking(spec, phase_name))
    if not rows:
        raise ValueError(f"На этапе {phase_name} нет полной допустимой конфигурации")
    best = rows[0]
    if spec["phases"][phase_name].get("freeze_requires_feasible", False) and not best.get("feasible"):
        if not allow_gate_failure:
            raise RuntimeError(
                f"Лучшая конфигурация этапа {phase_name} не прошла пороги; "
                "для диагностического продолжения укажите --allow-gate-failure"
            )
    base = _load_config(spec["base_config"])
    for key, value in best["params"].items():
        setattr(base, key, value)
    result = {
        "format": "smbh-cv-frozen-hyperparameters-v2",
        "source_phase": phase_name,
        "candidate_id": best["candidate_id"],
        "params": best["params"],
        "selection_metrics": {
            key: value
            for key, value in best.items()
            if key not in ("params", "metadata")
        },
        "gate_overridden": bool(allow_gate_failure and not best.get("feasible")),
        "full_pipeline_config": base.to_dict(),
        "config_fingerprint": _fingerprint(base.to_dict()),
        "warning": "После открытия проверочной выборки этот файл не изменять",
    }
    dump_json(Path(output), result)
    return result


def load_spec(path: str | Path) -> Dict[str, Any]:
    spec = read_json(Path(path))
    if not str(spec.get("version", "")).startswith("article-search-v2"):
        raise ValueError("Ожидался план article-search-v2")
    _load_config(spec["base_config"])
    for phase_name, phase in spec["phases"].items():
        stages = list(phase["run_stages"])
        if stages != sorted(stages, key=STAGES.index):
            raise ValueError(f"Ступени этапа {phase_name} заданы не по порядку")
        if phase["metric_mode"] == "d3" and "d3" not in stages:
            raise ValueError(f"Этапу {phase_name} для показателей D3 нужна ступень D3")
        if phase["metric_mode"] == "final" and "d6" not in stages:
            raise ValueError(f"Этапу {phase_name} для итоговых показателей нужна ступень D6")
    return spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Поэтапный воспроизводимый план экспериментов article_v2")
    parser.add_argument("--spec", default="experiments/article_v2/search_plan.json")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--phase", required=True)
    plan.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--phase", required=True)
    run.add_argument("--max-candidates", type=int)
    run.add_argument("--frozen")
    run.add_argument("--restart", action="store_true")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--phase", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--phase", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--allow-gate-failure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    spec = load_spec(args.spec)
    phase = spec["phases"][args.phase]
    if args.command == "plan":
        candidates = plan_phase(spec, args.phase, force=args.force)
        print(json.dumps({
            "phase": args.phase,
            "candidate_count": len(candidates),
            "path": str(candidate_plan_path(spec, args.phase)),
        }, ensure_ascii=False))
        return 0
    if args.command == "freeze":
        result = freeze_best(
            spec,
            args.phase,
            args.out,
            allow_gate_failure=args.allow_gate_failure,
        )
        print(json.dumps({"candidate_id": result["candidate_id"], "out": args.out}, ensure_ascii=False))
        return 0
    if phase["candidate_strategy"] == "frozen":
        if args.command == "summarize":
            path = candidate_plan_path(spec, args.phase)
            if not path.is_file():
                raise FileNotFoundError(f"Нет плана замороженной конфигурации: {path}")
            candidates = list(read_json(path)["candidates"])
        else:
            if not args.frozen:
                raise ValueError(f"Этап {args.phase} требует --frozen")
            candidates = [_frozen_candidate(args.frozen)]
            dump_json(candidate_plan_path(spec, args.phase), {
                "version": "article-v2-candidate-plan-v1",
                "phase": args.phase,
                "candidate_count": 1,
                "candidates": candidates,
            })
    else:
        candidates = plan_phase(spec, args.phase)
    if args.command == "summarize":
        result = summarize_phase(spec, args.phase, candidates)
        print(json.dumps({
            "phase": args.phase,
            "completed_candidates": sum(bool(row.get("complete")) for row in result["ranking"]),
        }, ensure_ascii=False))
        return 0
    selected = candidates[: args.max_candidates] if args.max_candidates is not None else candidates
    for candidate_index, candidate in enumerate(selected, 1):
        for seed in [int(value) for value in phase["algorithm_seeds"]]:
            logging.info(
                "Конфигурация %d/%d, идентификатор=%s, зерно=%d",
                candidate_index,
                len(selected),
                candidate["candidate_id"],
                seed,
            )
            _run_one(
                spec,
                args.phase,
                candidate,
                seed,
                restart=args.restart,
            )
    result = summarize_phase(spec, args.phase, candidates)
    print(json.dumps({
        "phase": args.phase,
        "completed_candidates": sum(bool(row.get("complete")) for row in result["ranking"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
