"""Backtest result schemas for CatalystLens validation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

CalibrationStatus = Literal[
    "synthetic_test_only",
    "preliminary_backtest",
    "insufficient_data",
    "externally_validated",
]


class CalibrationBucket(BaseModel):
    bucket_start: float
    bucket_end: float
    n_examples: int
    mean_predicted_probability: Optional[float] = None
    observed_event_rate: Optional[float] = None


class BacktestMetricSummary(BaseModel):
    n_examples: int
    brier_score: float
    log_loss: float
    roc_auc: Optional[float] = None
    expected_calibration_error: float
    calibration_buckets: list[CalibrationBucket]
    confusion_matrix: dict[str, int]
    event_rate: float
    mean_predicted_probability: float


class PerExampleBacktestResult(BaseModel):
    example_id: str
    ticker: str
    as_of_date: str
    predicted_cashout_risk: float
    predicted_reaches_catalyst_before_cashout: float
    predicted_program_discontinuation: float
    posterior_mean_pos: float
    actual_financing_before_catalyst: bool
    actual_reached_catalyst_before_financing_pressure: bool
    actual_program_discontinued_before_catalyst: bool
    actual_clinical_success: Optional[bool] = None


class BacktestResult(BaseModel):
    dataset_id: str
    synthetic: bool
    n_examples: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_name: str
    metric_summary: BacktestMetricSummary
    per_example_results: list[PerExampleBacktestResult]
    warnings: list[str] = Field(default_factory=list)
    calibration_status: CalibrationStatus
