from __future__ import annotations

import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np

from smbh_cv_pipeline import d4_initialization, d5_fit
from smbh_cv_pipeline.config import PipelineConfig
from smbh_cv_pipeline.experiment_v2 import (
    _adaptive_candidates,
    _d3_match_counts,
    _final_objective,
    _link_stage_prefix,
    _local_candidates,
    _sobol_candidates,
    load_spec,
    run_d3_statistics,
)
from smbh_cv_pipeline.io_utils import dump_json, run_paths


ROOT = Path(__file__).resolve().parents[1]


class ArticleV2PlanTests(unittest.TestCase):
    def test_global_design_has_96_points_and_separate_reference(self) -> None:
        spec = load_spec(ROOT / "experiments/article_v2/search_plan.json")
        candidates = _sobol_candidates(spec, "d1d3_global")
        self.assertEqual(len(candidates), 97)
        self.assertFalse(candidates[0]["eligible_for_selection"])
        searched = candidates[1:]
        self.assertTrue(all("d2_votes_fraction" not in row["params"] for row in searched))
        self.assertTrue(all(40000 <= row["params"]["d2_max_votes"] <= 300000 for row in searched))
        self.assertGreater(len({row["params"]["d1_edge_method"] for row in searched}), 1)
        self.assertGreater(len({row["params"]["d3_method"] for row in searched}), 1)
        for row in searched:
            params = row["params"]
            if params["d1_threshold_method"] != "mad":
                self.assertNotIn("d1_tau", params)
            if params["d3_threshold_method"] != "relative":
                self.assertNotIn("d3_relative_threshold", params)

    def test_d3_matching_requires_both_coordinate_conditions(self) -> None:
        cfg = PipelineConfig(max_center_distance_px=8.0)
        truth = [{"x0": 100.0, "y0": 100.0, "R": 60.0}]
        too_far = [{"x0": 109.0, "y0": 100.0, "R": 60.0}]
        self.assertEqual(_d3_match_counts(too_far, truth, cfg), (0, 1, 1))

    def test_final_objective_has_continuous_shortfall(self) -> None:
        objective = {
            "precision_floor": 0.80,
            "recall_floor": 0.70,
            "null_fppi_ceiling": 0.50,
            "seed_std_penalty": 0.25,
            "null_fppi_penalty": 0.05,
            "precision_shortfall_penalty": 1.0,
            "recall_shortfall_penalty": 1.0,
            "null_excess_penalty": 0.10,
        }
        base = {
            "article_f1": 0.75,
            "article_f1_ci95": [0.70, 0.80],
            "article_recall": 0.75,
            "null_fppi": 0.20,
        }
        low = _final_objective([base | {"article_precision": 0.79}], objective)
        high = _final_objective([base | {"article_precision": 0.81}], objective)
        self.assertAlmostEqual(high["score"] - low["score"], 0.01)
        self.assertFalse(low["feasible"])
        self.assertTrue(high["feasible"])

        empty = _final_objective([{
            "article_tp": 0,
            "article_fp": 0,
            "article_fn": 4,
            "article_precision": None,
            "article_recall": 0.0,
            "article_f1": None,
            "article_f1_ci95": [None, None],
            "null_fppi": 0.0,
        }], objective)
        self.assertTrue(np.isfinite(empty["score"]))
        self.assertEqual(empty["mean_f1"], 0.0)

    def test_adaptive_and_local_designs_use_finished_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = copy.deepcopy(load_spec(ROOT / "experiments/article_v2/search_plan.json"))
            spec["output_root"] = str(Path(tmp) / "runs")
            global_candidates = _sobol_candidates(spec, "d1d3_global")[1:]
            ranking = [
                {
                    "candidate_id": row["candidate_id"],
                    "params": row["params"],
                    "complete": True,
                    "eligible_for_selection": True,
                    "feasible": index < 8,
                    "pareto_rank": 1 + index // 8,
                    "score": 1.0 - index / 1000.0,
                }
                for index, row in enumerate(global_candidates)
            ]
            dump_json(
                Path(spec["output_root"]) / "d1d3_global" / "ranking.json",
                {"ranking": ranking},
            )

            first = _adaptive_candidates(spec, "d1d3_adaptive")
            second = _adaptive_candidates(spec, "d1d3_adaptive")
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)

            dump_json(
                Path(spec["output_root"]) / "d1d3_promotion" / "ranking.json",
                {"ranking": ranking[:12]},
            )
            local = _local_candidates(spec, "d1d3_local")
            centers = [row for row in local if row["metadata"]["kind"] == "local_center"]
            offsets = [row for row in local if row["metadata"]["kind"] == "local_offset"]
            self.assertEqual(len(centers), 3)
            self.assertTrue(offsets)
            self.assertEqual({row["metadata"]["step"] for row in offsets}, {0.02, 0.05, 0.10})


class ArticleV2TruthBoundaryTests(unittest.TestCase):
    def test_d4_and_d5_run_without_truth_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(
                out=tmp,
                d5_fit_background=False,
                save_arrays=False,
            )
            paths = run_paths(tmp)
            image_id = "image_000000"
            image_dir = paths["images"] / image_id
            image_dir.mkdir(parents=True, exist_ok=True)
            yy, xx = np.mgrid[:64, :64]
            image = np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / 50.0)
            np.save(image_dir / "image.npy", image.astype(np.float32))
            dump_json(paths["d3"] / image_id / "d3_candidates.json", {
                "image_id": image_id,
                "candidates": [],
            })

            d4_initialization.process_one(cfg.to_dict(), image_id)
            d5_fit.process_one(cfg.to_dict(), image_id)

            self.assertFalse((image_dir / "truth.json").exists())
            self.assertTrue((paths["d4"] / image_id / "d4_initial_params.json").is_file())
            self.assertTrue((paths["d5"] / image_id / "d5_fit.json").is_file())

    def test_saved_d3_prefix_is_linked_without_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "runs" / "d1d3_confirm" / "candidate" / "algorithm_seed_239"
            source_paths = run_paths(source)
            image_id = "image_000000"
            dump_json(source / "run_signature.json", {"fingerprint": "source"})
            dump_json(source / "manifest.json", {"items": [{"image_id": image_id}]})
            outputs = {
                "d1": "d1_edges.npz",
                "d2": "d2_sparse_accumulator.npz",
                "d3": "d3_candidates.json",
            }
            for stage, filename in outputs.items():
                output = source_paths[stage] / image_id / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"complete")
            frozen = root / "frozen.json"
            dump_json(frozen, {
                "source_phase": "d1d3_confirm",
                "candidate_id": "candidate",
                "params": {},
            })
            target = root / "runs" / "d4d5_global" / "next" / "algorithm_seed_239"
            run_paths(target)
            spec = {"output_root": str(root / "runs")}
            phase = {"reuse_prefix": {"frozen": str(frozen), "through": "d3"}}

            metadata = _link_stage_prefix(spec, phase, target, [image_id], 239)

            target_paths = run_paths(target)
            self.assertTrue(target_paths["d1"].is_symlink())
            self.assertTrue(target_paths["d3"].is_symlink())
            self.assertFalse(target_paths["d4"].is_symlink())
            self.assertFalse((source_paths["images"] / image_id / "truth.json").exists())
            self.assertEqual(metadata["through"], "d3")

    def test_d3_statistics_are_computed_after_d3_outputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(out=tmp)
            paths = run_paths(tmp)
            positive_id = "image_000000"
            null_id = "image_000001"
            dump_json(paths["manifest"], {
                "items": [{"image_id": positive_id}, {"image_id": null_id}],
            })
            for image_id in (positive_id, null_id):
                (paths["images"] / image_id).mkdir(parents=True, exist_ok=True)
                (paths["d3"] / image_id).mkdir(parents=True, exist_ok=True)
            dump_json(paths["images"] / positive_id / "truth.json", {
                "true_artifacts": [{"x0": 100.0, "y0": 110.0, "R": 60.0}],
            })
            dump_json(paths["images"] / null_id / "truth.json", {"true_artifacts": []})
            dump_json(paths["d3"] / positive_id / "d3_candidates.json", {
                "candidates": [{"x0": 101.0, "y0": 109.0, "R": 61.0}],
            })
            dump_json(paths["d3"] / null_id / "d3_candidates.json", {
                "candidates": [
                    {"x0": 80.0, "y0": 80.0, "R": 50.0},
                    {"x0": 120.0, "y0": 120.0, "R": 70.0},
                ],
            })

            summary = run_d3_statistics(cfg)

            self.assertEqual(summary["d3_recall"], 1.0)
            self.assertEqual(summary["null_candidates_per_image"], 2.0)
            self.assertIn("после завершения D1-D3", summary["truth_access_boundary"])


if __name__ == "__main__":
    unittest.main()
