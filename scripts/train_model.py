#!/usr/bin/env python
"""
Train fraud detection models and write artefacts the API can serve.

    # Against the real ULB dataset (not redistributable - see --help)
    python scripts/train_model.py --data data/creditcard.csv --output models/

    # Against calibrated synthetic data, to exercise the pipeline
    python scripts/train_model.py --output models/

Artefacts land in --output, which is what MODEL_PATH points at, so the API
picks them up on its next start and /api/v1/health flips to "trained": true.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import TrainingConfig, load_dataset, save_artifacts, train
from src.training.dataset import ULB_ROWS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The ULB Credit Card Fraud Detection dataset (284,807 real\n"
            "transactions, 492 fraudulent) is not redistributable and is not\n"
            "included here. Download creditcard.csv from\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and pass\n"
            "it with --data.\n\n"
            "Without --data the run uses synthetic data calibrated to that\n"
            "dataset's published statistics. The pipeline is then genuinely\n"
            "exercised, but the resulting scores describe a generated problem\n"
            "and are NOT evidence of real-world accuracy. Every report records\n"
            "which source it used."
        ),
    )
    parser.add_argument("--data", type=Path, help="path to creditcard.csv")
    parser.add_argument(
        "--output", type=Path, default=Path("models"), help="artefact directory (default: models/)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=ULB_ROWS,
        help="rows to generate when --data is absent (default: matches ULB)",
    )
    parser.add_argument(
        "--review-cost",
        type=float,
        default=3.0,
        help="cost of reviewing one alert, in transaction-amount units (default: 3.0)",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset = load_dataset(args.data, rows=args.rows)

    if not dataset.source.is_real:
        print(
            "\n  NOTE: no dataset supplied, so these results come from synthetic\n"
            "  data calibrated to the ULB dataset's published statistics.\n"
            "  They demonstrate the pipeline. They are not measured accuracy.\n",
            file=sys.stderr,
        )

    config = TrainingConfig(review_cost=args.review_cost)
    result, models, scaler = train(dataset=dataset, config=config)
    report_path = save_artifacts(result, models, scaler, args.output)

    summary = result.to_dict()
    best = summary["models"][summary["best_model"]]

    print(json.dumps(summary, indent=2))
    print(
        f"\nBest model: {summary['best_model']}"
        f"\n  Average precision : {best['average_precision']:.4f}"
        f"  (random baseline {summary['baseline_average_precision']:.6f})"
        f"\n  Recall            : {best['at_threshold']['recall']:.4f}"
        f"\n  Precision         : {best['at_threshold']['precision']:.4f}"
        f"\n  Net savings       : {best['cost']['net_savings']:,.2f}"
        f"\n  Data source       : {summary['data']['source']}"
        f"\n\nArtefacts + metrics: {report_path.parent}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
