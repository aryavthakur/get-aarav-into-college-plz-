"""CLI entrypoint for CatalystLens historical backtests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.validation.backtest import load_historical_examples, run_backtest
from training.validation.backtest_report import generate_backtest_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CatalystLens backtest on a historical catalyst dataset.")
    parser.add_argument("--dataset", required=True, help="Path to historical dataset CSV.")
    parser.add_argument("--target", default="financing_before_catalyst", help="Binary target to score.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for JSON and Markdown outputs.")
    args = parser.parse_args()

    dataset = load_historical_examples(args.dataset)
    result = run_backtest(dataset, target_name=args.target)
    report = generate_backtest_report(result)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "backtest_result.json"
    md_path = output_dir / "backtest_report.md"
    json_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    md_path.write_text(report)

    print("CatalystLens backtest complete")
    print(f"dataset_id: {result.dataset_id}")
    print(f"target: {result.target_name}")
    print(f"n_examples: {result.n_examples}")
    print(f"calibration_status: {result.calibration_status}")
    print(f"brier_score: {result.metric_summary.brier_score:.4f}")
    print(f"expected_calibration_error: {result.metric_summary.expected_calibration_error:.4f}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")


if __name__ == "__main__":
    main()
