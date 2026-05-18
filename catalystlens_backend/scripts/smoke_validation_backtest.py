"""Smoke test for the full synthetic validation backtest."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from the catalystlens_backend directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schemas import SimulationConfig
from training.validation.backtest import load_historical_examples, run_backtest


DATASET_PATH = Path("training/datasets/example_historical_biotech_panel.csv")


def main() -> None:
    dataset = load_historical_examples(DATASET_PATH)
    result = run_backtest(
        dataset,
        SimulationConfig(n_simulations=100, random_seed=42, monthly_horizon=36),
    )

    print(f"n_examples: {result.n_examples}")
    print(f"brier_score: {result.metric_summary.brier_score:.6f}")
    print(
        "expected_calibration_error: "
        f"{result.metric_summary.expected_calibration_error:.6f}"
    )
    print(f"calibration_status: {result.calibration_status}")

    assert result.n_examples >= 30
    assert result.calibration_status == "synthetic_test_only"
    assert result.metric_summary.brier_score >= 0
    assert result.metric_summary.expected_calibration_error >= 0


if __name__ == "__main__":
    main()
