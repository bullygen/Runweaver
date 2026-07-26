from __future__ import annotations

"""A deterministic article-quality decision gate using only measured variables."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .io_utils import dump_json, read_json


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _get(summary: Mapping[str, Any], key: str, endpoint: int | None = None) -> float | None:
    value = summary.get(key)
    if endpoint is not None:
        if not isinstance(value, list) or len(value) <= endpoint:
            return None
        value = value[endpoint]
    return float(value) if _finite(value) else None


def _criterion(name: str, value: float | None, operator: str, threshold: float) -> Dict[str, Any]:
    if value is None:
        passed = False
        reason = "missing_or_non_finite"
    else:
        passed = value >= threshold if operator == ">=" else value <= threshold
        reason = "measured"
    return {"name": name, "value": value, "operator": operator, "threshold": threshold, "passed": bool(passed), "source": reason}


def evaluate_quality(
    test: Mapping[str, Any],
    verification: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    validation: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a quality tier without latent scores or expert-only variables.

    Every decision input is copied into ``criteria`` and is traceable to a
    statistics/summary.json value.  Confidence interval endpoints are preferred
    for the strong tier; point estimates are used only for the limited tier.
    """
    strong_t = thresholds["strong"]
    limited_t = thresholds["promising_limited"]
    strong = [
        _criterion("test_precision_ci95_low", _get(test, "article_precision_ci95", 0), ">=", strong_t["test_precision_ci95_low"]),
        _criterion("test_recall_ci95_low", _get(test, "article_recall_ci95", 0), ">=", strong_t["test_recall_ci95_low"]),
        _criterion("test_f1_ci95_low", _get(test, "article_f1_ci95", 0), ">=", strong_t["test_f1_ci95_low"]),
        _criterion("test_null_fppi_ci95_high", _get(test, "null_fppi_ci95", 1), "<=", strong_t["test_null_fppi_ci95_high"]),
        _criterion("test_worst_factor_f1_ci95_low", _get(test, "worst_eligible_factor_f1_ci95_low"), ">=", strong_t["test_worst_factor_f1_ci95_low"]),
        _criterion("verification_precision_ci95_low", _get(verification, "article_precision_ci95", 0), ">=", strong_t["verification_precision_ci95_low"]),
        _criterion("verification_recall_ci95_low", _get(verification, "article_recall_ci95", 0), ">=", strong_t["verification_recall_ci95_low"]),
        _criterion("verification_f1_ci95_low", _get(verification, "article_f1_ci95", 0), ">=", strong_t["verification_f1_ci95_low"]),
        _criterion("verification_null_fppi_ci95_high", _get(verification, "null_fppi_ci95", 1), "<=", strong_t["verification_null_fppi_ci95_high"]),
    ]
    limited = [
        _criterion("test_precision", _get(test, "article_precision"), ">=", limited_t["test_precision"]),
        _criterion("test_recall", _get(test, "article_recall"), ">=", limited_t["test_recall"]),
        _criterion("test_f1", _get(test, "article_f1"), ">=", limited_t["test_f1"]),
        _criterion("test_null_fppi", _get(test, "null_fppi"), "<=", limited_t["test_null_fppi"]),
        _criterion("verification_f1", _get(verification, "article_f1"), ">=", limited_t["verification_f1"]),
        _criterion("verification_null_fppi", _get(verification, "null_fppi"), "<=", limited_t["verification_null_fppi"]),
    ]
    required_counts = [
        _criterion("test_n_images", _get(test, "n_images_total"), ">=", float(thresholds["minimum_sample_sizes"]["test"])),
        _criterion("test_n_null_images", _get(test, "n_null_images"), ">=", float(thresholds["minimum_sample_sizes"]["test_null"])),
        _criterion("verification_n_images", _get(verification, "n_images_total"), ">=", float(thresholds["minimum_sample_sizes"]["verification"])),
        _criterion("verification_n_null_images", _get(verification, "n_null_images"), ">=", float(thresholds["minimum_sample_sizes"]["verification_null"])),
    ]
    enough_data = all(row["passed"] for row in required_counts)
    if not enough_data:
        tier = "insufficient_evidence"
    elif all(row["passed"] for row in strong):
        tier = "article_ready_strong_under_tested_conditions"
    elif all(row["passed"] for row in limited):
        tier = "promising_but_limited"
    else:
        tier = "experimental_not_yet_robust"
    validation_gap = None
    if validation is not None and _get(validation, "article_f1") is not None and _get(test, "article_f1") is not None:
        validation_gap = float(_get(validation, "article_f1") - _get(test, "article_f1"))
    return {
        "decision_version": thresholds["version"],
        "metric_protocol": "article_geometric_iou_v1",
        "tier": tier,
        "all_strong_criteria_pass": all(row["passed"] for row in strong),
        "all_limited_criteria_pass": all(row["passed"] for row in limited),
        "minimum_sample_sizes_pass": enough_data,
        "validation_minus_test_f1": validation_gap,
        "criteria": {"minimum_sample_sizes": required_counts, "strong": strong, "promising_limited": limited},
        "interpretation_limit": "The tier describes measured annular-morphology detection under the declared synthetic conditions; it does not identify a physical photon ring.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the predeclared SMBH article quality gate.")
    parser.add_argument("--test", required=True, help="Test statistics/summary.json")
    parser.add_argument("--verification", required=True, help="Verification statistics/summary.json")
    parser.add_argument("--validation", help="Optional validation statistics/summary.json")
    parser.add_argument("--thresholds", default="experiments/article_v1/quality_gate.json")
    parser.add_argument("--out", default="article_quality_decision.json")
    args = parser.parse_args(argv)
    result = evaluate_quality(
        read_json(Path(args.test)), read_json(Path(args.verification)), read_json(Path(args.thresholds)),
        read_json(Path(args.validation)) if args.validation else None,
    )
    dump_json(Path(args.out), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["tier"] != "insufficient_evidence" else 2


if __name__ == "__main__":
    raise SystemExit(main())
