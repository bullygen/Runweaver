from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .article_data import build_design, generate_split, load_plan, validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and validate leakage-safe SMBH article datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "generate"):
        p = sub.add_parser(command)
        p.add_argument("--plan", default="experiments/article_v1/dataset_plan.json")
        p.add_argument("--split", choices=["development", "validation", "test", "verification", "all"], default="all")
        p.add_argument("--scale", choices=["pilot", "full"], default="full")
        if command == "generate":
            p.add_argument("--out", default="datasets/article_v1")
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--overwrite", action="store_true")
            p.add_argument("--no-components", action="store_true", help="Save only image.npy and truth.json to reduce disk usage.")
    val = sub.add_parser("validate")
    val.add_argument("--plan", default="experiments/article_v1/dataset_plan.json")
    val.add_argument("--out", default="datasets/article_v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_plan(args.plan)
    if args.command == "validate":
        report = validate_dataset(args.out, expected_plan=plan)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["valid"] else 2
    split_names = list(plan["splits"]) if args.split == "all" else [args.split]
    if args.command == "plan":
        summaries = []
        for split in split_names:
            design = build_design(plan, split, args.scale)
            summaries.append({"split": split, "n_images": design["n_images"], "plan_hash": design["plan_hash"]})
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return 0
    for split in split_names:
        manifest = generate_split(
            plan, split, Path(args.out), scale=args.scale, workers=args.workers,
            overwrite=args.overwrite, save_components=not args.no_components,
        )
        print(json.dumps({"split": split, "n_images": len(manifest["items"])}, ensure_ascii=False))
    report = validate_dataset(args.out, expected_plan=plan)
    print(json.dumps({"valid": report["valid"], "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
