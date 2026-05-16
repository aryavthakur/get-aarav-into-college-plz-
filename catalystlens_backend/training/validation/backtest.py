"""Run CatalystLens against point-in-time historical catalyst examples."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from app.engines.monte_carlo import run_full_audit
from app.models.schemas import (
    AuditRequest,
    ClinicalCatalystInput,
    CompanyFinancialInput,
    DisclosureInput,
    SimulationConfig,
    SuccessProbabilityInput,
    ValuationInput,
)
from training.datasets.historical_schema import HistoricalCompanyCatalystExample, HistoricalDataset
from training.validation.metrics import (
    brier_score,
    calibration_by_bucket,
    confusion_matrix_at_threshold,
    expected_calibration_error,
    log_loss_binary,
    roc_auc,
)
from training.validation.schemas import (
    BacktestMetricSummary,
    BacktestResult,
    PerExampleBacktestResult,
)


def _parse_bool(value: str | bool | None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _clean_row(row: dict[str, str]) -> dict:
    out: dict = {}
    for key, value in row.items():
        if key in {"dataset_id", "synthetic", "source_description"}:
            continue
        out[key] = None if value == "" else value
    for key in (
        "cash",
        "marketable_securities",
        "quarterly_operating_cash_burn",
        "debt",
        "market_cap",
        "shares_outstanding",
        "estimated_dilution",
    ):
        if out.get(key) is not None:
            out[key] = float(out[key])
    for key in ("enrollment_target", "enrollment_completed"):
        if out.get(key) is not None:
            out[key] = int(float(out[key]))
    for key in ("financing_before_catalyst", "program_discontinued_before_catalyst", "delayed_readout"):
        parsed = _parse_bool(out.get(key))
        out[key] = parsed
    if out.get("delayed_readout") is None:
        out.pop("delayed_readout", None)
    return out


def load_historical_examples(path: str | Path) -> HistoricalDataset:
    """Load a CSV historical catalyst panel into validated schemas."""
    csv_path = Path(path)
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("historical dataset CSV contains no rows")

    dataset_id = rows[0].get("dataset_id") or csv_path.stem
    synthetic = bool(_parse_bool(rows[0].get("synthetic")))
    source_description = rows[0].get("source_description") or "Historical CatalystLens validation dataset."
    examples = [HistoricalCompanyCatalystExample(**_clean_row(row)) for row in rows]
    return HistoricalDataset(
        dataset_id=dataset_id,
        examples=examples,
        synthetic=synthetic,
        source_description=source_description,
    )


def _months_between(start, end) -> float:
    return max((end - start).days / 30.4375, 1.0)


def build_audit_request_from_historical_example(
    example: HistoricalCompanyCatalystExample,
    simulation_config: SimulationConfig | None = None,
) -> AuditRequest:
    """Construct a point-in-time AuditRequest using only fields known as of as_of_date."""
    enrollment_target = example.enrollment_target or 100
    enrollment_completed = min(example.enrollment_completed or 0, enrollment_target)
    stated_months = _months_between(example.as_of_date, example.stated_catalyst_date)

    financial = CompanyFinancialInput(
        company_name=example.company_name,
        ticker=example.ticker,
        cash_on_hand=example.cash,
        marketable_securities=example.marketable_securities,
        quarterly_operating_cash_burn=max(example.quarterly_operating_cash_burn, 1.0),
        market_cap=max(example.market_cap, 1.0),
        debt=example.debt,
    )
    clinical = ClinicalCatalystInput(
        asset_name=f"{example.ticker}-lead-program",
        indication=example.disease_area or "Unknown",
        trial_phase=example.trial_phase,
        trial_status=example.trial_status,
        stated_months_to_catalyst=stated_months,
        enrollment_target=enrollment_target,
        enrollment_completed=enrollment_completed,
        enrollment_rate_per_month=max(enrollment_target / max(stated_months, 1.0), 1.0),
        number_of_sites=10,
        public_readout_months=stated_months,
    )
    success = SuccessProbabilityInput(
        trial_phase=example.trial_phase,
        disease_area=example.disease_area,
        modality=example.modality,
        endpoint_family=example.endpoint_family,
    )
    valuation = ValuationInput(
        asset_value_success=max(example.market_cap * 2.5, 50_000_000.0),
        expected_dilution_if_refinanced=example.estimated_dilution or 0.25,
    )
    disclosure = DisclosureInput(
        company_narrative_distribution={
            "runway_strength": 0.6,
            "clinical_timeline_confidence": 0.6,
            "dilution_risk": 0.4,
            "trial_maturity": 0.5,
            "endpoint_strength": 0.5,
            "pipeline_diversification": 0.4,
        },
        structured_audit_distribution={
            "runway_strength": 0.5,
            "clinical_timeline_confidence": 0.5,
            "dilution_risk": 0.5,
            "trial_maturity": 0.5,
            "endpoint_strength": 0.5,
            "pipeline_diversification": 0.4,
        },
    )
    sim = simulation_config or SimulationConfig(n_simulations=300, random_seed=42, monthly_horizon=48)
    return AuditRequest(
        financial=financial,
        clinical=clinical,
        success_probability=success,
        valuation=valuation,
        disclosure=disclosure,
        simulation=sim,
    )


def _target_values(results: Iterable[PerExampleBacktestResult], target_name: str) -> tuple[list[int], list[float]]:
    y_true: list[int] = []
    y_prob: list[float] = []
    for row in results:
        if target_name == "financing_before_catalyst":
            y_true.append(int(row.actual_financing_before_catalyst))
            y_prob.append(row.predicted_cashout_risk)
        elif target_name == "program_discontinued_before_catalyst":
            y_true.append(int(row.actual_program_discontinued_before_catalyst))
            y_prob.append(row.predicted_program_discontinuation)
        elif target_name == "reached_catalyst_before_financing_pressure":
            y_true.append(int(row.actual_reached_catalyst_before_financing_pressure))
            y_prob.append(row.predicted_reaches_catalyst_before_cashout)
        else:
            raise ValueError(f"Unsupported backtest target: {target_name}")
    return y_true, y_prob


def run_backtest(
    dataset: HistoricalDataset,
    simulation_config: SimulationConfig | None = None,
    target_name: str = "financing_before_catalyst",
) -> BacktestResult:
    """Run the current CatalystLens audit engine on every point-in-time example."""
    sim = simulation_config or SimulationConfig(n_simulations=300, random_seed=42, monthly_horizon=48)
    per_example: list[PerExampleBacktestResult] = []
    warnings: list[str] = []
    if dataset.synthetic:
        warnings.append("Synthetic test data only; this does not validate model performance.")

    for i, example in enumerate(dataset.examples):
        req = build_audit_request_from_historical_example(
            example,
            sim.model_copy(update={"random_seed": sim.random_seed + i}),
        )
        audit = run_full_audit(req)
        clinical_success = None
        if example.clinical_outcome in {"positive", "mixed"}:
            clinical_success = True
        elif example.clinical_outcome == "negative":
            clinical_success = False
        reached = not (
            example.financing_before_catalyst
            or example.program_discontinued_before_catalyst
        )
        per_example.append(PerExampleBacktestResult(
            example_id=example.example_id,
            ticker=example.ticker,
            as_of_date=example.as_of_date.isoformat(),
            predicted_cashout_risk=audit.capital_to_catalyst.probability_cashout_before_catalyst,
            predicted_reaches_catalyst_before_cashout=audit.capital_to_catalyst.probability_reaches_catalyst,
            predicted_program_discontinuation=audit.valuation.p_program_discontinuation,
            posterior_mean_pos=audit.success_probability.posterior_mean,
            actual_financing_before_catalyst=example.financing_before_catalyst,
            actual_reached_catalyst_before_financing_pressure=reached,
            actual_program_discontinued_before_catalyst=example.program_discontinued_before_catalyst,
            actual_clinical_success=clinical_success,
        ))

    y_true, y_prob = _target_values(per_example, target_name)
    buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    summary = BacktestMetricSummary(
        n_examples=len(per_example),
        brier_score=brier_score(y_true, y_prob),
        log_loss=log_loss_binary(y_true, y_prob),
        roc_auc=roc_auc(y_true, y_prob),
        expected_calibration_error=expected_calibration_error(y_true, y_prob, buckets),
        calibration_buckets=calibration_by_bucket(y_true, y_prob, buckets),
        confusion_matrix=confusion_matrix_at_threshold(y_true, y_prob, threshold=0.5),
        event_rate=float(np.mean(y_true)),
        mean_predicted_probability=float(np.mean(y_prob)),
    )
    status = "synthetic_test_only" if dataset.synthetic else (
        "preliminary_backtest" if len(per_example) >= 25 else "insufficient_data"
    )
    return BacktestResult(
        dataset_id=dataset.dataset_id,
        synthetic=dataset.synthetic,
        n_examples=len(per_example),
        target_name=target_name,
        metric_summary=summary,
        per_example_results=per_example,
        warnings=warnings,
        calibration_status=status,
    )
