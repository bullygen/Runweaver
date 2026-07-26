from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from smbh_cv_pipeline.article_data import build_design, generate_split, load_plan, validate_dataset
from smbh_cv_pipeline.config import PipelineConfig
from smbh_cv_pipeline.experiment_runner import create_candidates, load_spec
from smbh_cv_pipeline.io_utils import dump_json
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
