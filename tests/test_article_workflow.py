from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from smbh_cv_pipeline.article_data import build_design, generate_split, load_plan, validate_dataset
from smbh_cv_pipeline.cli import _pool_map
from smbh_cv_pipeline.config import PipelineConfig
from smbh_cv_pipeline.experiment_runner import (
    _STAGE_OUTPUT,
    _missing_stage_image_ids,
    _run_one,
    create_candidates,
    load_spec,
)
from smbh_cv_pipeline.io_utils import dump_json, run_paths
from smbh_cv_pipeline.quality_gate import evaluate_quality
from smbh_cv_pipeline.statistics import _candidate_stage_recall, _match_article


ROOT = Path(__file__).resolve().parents[1]


class DatasetDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_plan(ROOT / "experiments/article_v1/dataset_plan.json")

    def test_split_sizes_and_no_seed_leakage(self) -> None:
        designs = {name: build_design(self.plan, name, "pilot") for name in self.plan["splits"]}
        families = [item["family_id"] for design in designs.values() for item in design["items"]]
        seeds = [item["sample_seed"] for design in designs.values() for item in design["items"]]
        self.assertEqual(len(families), len(set(families)))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(len(designs["development"]["items"]), 40)
        self.assertEqual(len(designs["validation"]["items"]), 20)
        self.assertEqual(len(designs["test"]["items"]), 30)
        self.assertEqual(len(designs["verification"]["items"]), 30)

    def test_small_generation_is_finite_and_valid(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["splits"] = {"development": plan["splits"]["development"]}
        plan["splits"]["development"]["pilot_images"] = 4
        with tempfile.TemporaryDirectory() as tmp:
            manifest = generate_split(plan, "development", tmp, scale="pilot", workers=1, save_components=True)
            self.assertEqual(len(manifest["items"]), 4)
            for item in manifest["items"]:
                image = np.load(Path(tmp) / "development" / "images" / item["image_id"] / "image.npy")
                self.assertEqual(image.shape, (256, 256))
                self.assertTrue(np.all(np.isfinite(image)))
            report = validate_dataset(tmp, expected_plan=plan)
            self.assertTrue(report["valid"], report["errors"])


class SearchTests(unittest.TestCase):
    def test_candidates_are_deterministic_and_methods_locked(self) -> None:
        spec = load_spec(ROOT / "experiments/article_v1/search_plan.json")
        first = create_candidates(spec)
        second = create_candidates(spec)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 25)
        locked = set(spec["locked_method_hypotheses"])
        for candidate in first:
            self.assertFalse(locked.intersection(candidate["params"]))
            if "d6_merge_center_px" in candidate["params"]:
                self.assertEqual(candidate["params"]["d6_merge_center_px"], candidate["params"]["d6_merge_radius_px"])

    def test_missing_stage_images_are_detected_for_d1_through_d6(self) -> None:
        image_ids = ["image_000000", "image_000001", "image_000002"]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(out=tmp)
            paths = run_paths(tmp)
            for stage, (path_key, filename) in _STAGE_OUTPUT.items():
                for image_id in (image_ids[0], image_ids[2]):
                    output = paths[path_key] / image_id / filename
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"complete")
                with self.subTest(stage=stage):
                    self.assertEqual(_missing_stage_image_ids(cfg, stage, image_ids), [image_ids[1]])

    def test_run_resumes_each_partial_stage_with_only_missing_images(self) -> None:
        image_ids = ["image_000000", "image_000001", "image_000002"]
        stages = list(_STAGE_OUTPUT)
        for interrupted_index, interrupted_stage in enumerate(stages):
            with self.subTest(stage=interrupted_stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset_split = root / "dataset" / "development"
                dataset_items = []
                for image_id in image_ids:
                    image_dir = dataset_split / "images" / image_id
                    image_dir.mkdir(parents=True, exist_ok=True)
                    dataset_items.append({"image_id": image_id, "image_sha256": image_id})
                dump_json(dataset_split / "manifest.json", {"plan_hash": "test-plan", "items": dataset_items})

                base_config = root / "base_config.json"
                dump_json(base_config, PipelineConfig().to_dict())
                output_root = root / "runs"
                run_dir = output_root / "screening" / "candidate" / "algorithm_seed_239"
                paths = run_paths(run_dir)

                for completed_stage in stages[:interrupted_index]:
                    path_key, filename = _STAGE_OUTPUT[completed_stage]
                    for image_id in image_ids:
                        output = paths[path_key] / image_id / filename
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(b"complete")

                path_key, filename = _STAGE_OUTPUT[interrupted_stage]
                for image_id in (image_ids[0], image_ids[2]):
                    output = paths[path_key] / image_id / filename
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"complete")

                calls = []

                def fake_pool(stage, func, cfg, pending_ids, workers, total_items=None):
                    pending_ids = list(pending_ids)
                    calls.append((stage, pending_ids, total_items))
                    output_key, output_filename = _STAGE_OUTPUT[stage]
                    stage_paths = run_paths(cfg.out)
                    for image_id in pending_ids:
                        output = stage_paths[output_key] / image_id / output_filename
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(b"complete")
                    return [{"image_id": image_id, "status": "ok"} for image_id in pending_ids]

                def fake_statistics(cfg):
                    summary = {"article_f1": 0.0}
                    dump_json(Path(cfg.out) / "statistics" / "summary.json", summary)
                    return summary

                spec = {
                    "dataset_root": str(root / "dataset"),
                    "base_config": str(base_config),
                    "output_root": str(output_root),
                    "workers": {stage: 1 for stage in stages},
                    "phases": {"screening": {"split": "development", "algorithm_seeds": [239]}},
                }
                candidate = {"candidate_id": "candidate", "params": {}}

                with (
                    patch("smbh_cv_pipeline.experiment_runner._prepare_run"),
                    patch("smbh_cv_pipeline.experiment_runner._pool_map", side_effect=fake_pool),
                    patch("smbh_cv_pipeline.experiment_runner.run_statistics", side_effect=fake_statistics),
                ):
                    _run_one(spec, "screening", candidate, 239)

                expected = [
                    (stage, [image_ids[1]] if index == interrupted_index else image_ids, len(image_ids))
                    for index, stage in enumerate(stages[interrupted_index:], interrupted_index)
                ]
                self.assertEqual(calls, expected)

    def test_resumed_stage_summary_separates_pending_and_preexisting_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(out=tmp)

            def process_one(cfg_dict, image_id):
                return {"image_id": image_id, "status": "ok"}

            _pool_map("d5", process_one, cfg, ["image_000002"], workers=1, total_items=3)
            summary = json.loads((Path(tmp) / "runtime" / "d5_stage_summary.json").read_text())
            self.assertEqual(summary["n_items"], 1)
            self.assertEqual(summary["n_items_total"], 3)
            self.assertEqual(summary["n_items_preexisting"], 2)
            self.assertEqual(summary["wall_time_scope"], "current_invocation")


class MetricTests(unittest.TestCase):
    def test_json_artifacts_replace_non_finite_values_with_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strict.json"
            dump_json(path, {"nan": float("nan"), "inf": float("inf"), "value": 1.0})
            payload = json.loads(path.read_text())
            self.assertIsNone(payload["nan"])
            self.assertIsNone(payload["inf"])
            self.assertEqual(payload["value"], 1.0)

    def test_candidate_recall_uses_only_d3_geometry(self) -> None:
        truth = [{"A": 0.01, "R": 60.0, "sigma": 10.0, "B": 1.0, "phi": 2.0, "x0": 100.0, "y0": 110.0}]
        candidates = [{"R": 61.0, "x0": 101.0, "y0": 109.0, "score": 1.0}]
        self.assertEqual(_candidate_stage_recall(candidates, truth, PipelineConfig(), 256, 256), 1.0)

    def test_article_match_is_geometric_under_profile_parameter_mismatch(self) -> None:
        truth = [{"A": 0.01, "R": 60.0, "sigma": 5.0, "B": 0.0, "phi": 0.0, "x0": 100.0, "y0": 110.0}]
        fitted = [{"A": 0.07, "R": 61.0, "sigma": 7.0, "B": 1.0, "phi": 2.8, "x0": 101.0, "y0": 109.0}]
        matches, tp, fp, fn = _match_article(truth, fitted, PipelineConfig(), 256, 256)
        self.assertEqual((tp, fp, fn), (1, 0, 0))
        self.assertEqual(matches[0]["match_protocol"], "article_geometric_iou_v1")

    def test_quality_gate_has_no_hidden_score(self) -> None:
        thresholds = json.loads((ROOT / "experiments/article_v1/quality_gate.json").read_text())
        test = {
            "n_images_total": 500, "n_null_images": 125,
            "article_precision": 0.95, "article_recall": 0.88, "article_f1": 0.91,
            "article_precision_ci95": [0.92, 0.97], "article_recall_ci95": [0.84, 0.91], "article_f1_ci95": [0.87, 0.93],
            "null_fppi": 0.04, "null_fppi_ci95": [0.01, 0.08], "worst_eligible_factor_f1_ci95_low": 0.73,
        }
        verification = {
            "n_images_total": 500, "n_null_images": 180,
            "article_precision": 0.82, "article_recall": 0.72, "article_f1": 0.77,
            "article_precision_ci95": [0.78, 0.86], "article_recall_ci95": [0.67, 0.76], "article_f1_ci95": [0.72, 0.80],
            "null_fppi": 0.12, "null_fppi_ci95": [0.07, 0.20], "worst_eligible_factor_f1_ci95_low": 0.60,
        }
        decision = evaluate_quality(test, verification, thresholds)
        self.assertEqual(decision["tier"], "article_ready_strong_under_tested_conditions")
        self.assertNotIn("subjective_score", decision)


if __name__ == "__main__":
    unittest.main()
