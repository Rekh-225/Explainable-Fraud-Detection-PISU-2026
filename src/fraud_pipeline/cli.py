"""Command-line entry points.

    fraud-pipeline run --dataset PATH --source kaggle|openml|synthetic [options]
    fraud-pipeline reproduce-historical [--dataset PATH]
    fraud-pipeline synthetic --output PATH [--rows N --seed S --fraud-rate R --duplicates D]
    fraud-pipeline verify-run RUN_DIR

``python -m fraud_pipeline ...`` is equivalent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import json_ready, verify_run
from .modeling import ModelConfig
from .pipeline import DEFAULT_OUTPUT_ROOT, HISTORICAL_KAGGLE_CSV, RunConfig, run_pipeline
from .synthetic import write_synthetic_csv


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None, help="Defaults to <utc-stamp>_<source>_<sha8>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-recall", type=float, default=0.80)
    parser.add_argument("--duplicate-policy", choices=["drop_exact", "keep"], default="drop_exact")
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--no-figures", action="store_true")


def _run_config(args: argparse.Namespace, dataset: Path, source: str) -> RunConfig:
    return RunConfig(
        dataset_path=dataset,
        source=source,
        output_root=args.output_root,
        run_id=args.run_id,
        seed=args.seed,
        minimum_recall=args.minimum_recall,
        duplicate_policy=args.duplicate_policy,
        model=ModelConfig(seed=args.seed, rf_n_estimators=args.rf_estimators),
        figures=not args.no_figures,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-pipeline", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the experiment on an explicit dataset")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--source", choices=["kaggle", "openml", "synthetic"], required=True)
    _add_run_options(run)

    hist = sub.add_parser(
        "reproduce-historical",
        help="Re-run the original experiment (Kaggle CSV, seed 42, 64/16/20, 200 trees)",
    )
    hist.add_argument("--dataset", type=Path, default=HISTORICAL_KAGGLE_CSV)
    _add_run_options(hist)

    syn = sub.add_parser("synthetic", help="Write a deterministic synthetic fixture CSV")
    syn.add_argument("--output", type=Path, required=True)
    syn.add_argument("--rows", type=int, default=4000)
    syn.add_argument("--seed", type=int, default=0)
    syn.add_argument("--fraud-rate", type=float, default=0.02)
    syn.add_argument("--duplicates", type=int, default=0)
    syn.add_argument("--no-time", action="store_true", help="Mimic the OpenML layout without Time")

    verify = sub.add_parser("verify-run", help="Re-hash artifacts and compare with the manifest")
    verify.add_argument("run_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "synthetic":
        path = write_synthetic_csv(
            args.output,
            n_rows=args.rows,
            fraud_rate=args.fraud_rate,
            seed=args.seed,
            include_time=not args.no_time,
            n_duplicates=args.duplicates,
        )
        print(path)
        return 0

    if args.command == "verify-run":
        report = verify_run(args.run_dir)
        print(json.dumps(report, indent=2))
        return 0 if not any(report.values()) else 1

    if args.command == "run":
        config = _run_config(args, args.dataset, args.source)
    else:
        if not args.dataset.is_file():
            print(
                f"Historical dataset not found at {args.dataset}. Download creditcard.csv from "
                "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (login required) and "
                "place it there, or pass --dataset. The OpenML mirror is NOT substituted "
                "automatically because it omits the Time feature.",
                file=sys.stderr,
            )
            return 2
        config = _run_config(args, args.dataset, "kaggle")

    result = run_pipeline(config)
    summary = {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "selected_model": result.selected_model,
        "selected_threshold": result.thresholds[result.selected_model],
        "test_metrics": result.results[
            (result.results["model"] == result.selected_model)
            & (result.results["policy"] == "validation_operating_point")
        ].iloc[0].to_dict(),
    }
    print(json.dumps(json_ready(summary), indent=2))
    return 0
