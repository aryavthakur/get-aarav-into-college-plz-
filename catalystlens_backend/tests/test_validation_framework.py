"""Tests for historical validation and backtesting scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models.schemas import SimulationConfig
from app.registry.model_registry import ModelArtifactCard, ModelRegistry
from training.datasets.historical_schema import (
    HistoricalCompanyCatalystExample,
    HistoricalDataset,
)
from training.validation.backtest import load_historical_examples, run_backtest
from training.validation.backtest_report import generate_backtest_report
from training.validation.run_backtest import write_prediction_error_table
from training.validation.metrics import (
    brier_score,
    calibration_diagnostics,
    calibration_by_bucket,
    confusion_matrix_at_threshold,
    expected_calibration_error,
    log_loss_binary,
)
from training.validation.schemas import PerExampleBacktestResult
from training.validation.target_mapping import (
    TARGET_DEFINITIONS,
    extract_actual_label,
    probability_for_target,
)


DATASET_PATH = Path("training/datasets/example_historical_biotech_panel.csv")


@pytest.fixture(scope="module")
def synthetic_dataset():
    return load_historical_examples(DATASET_PATH)


@pytest.fixture(scope="module")
def synthetic_backtest_result(synthetic_dataset):
    return run_backtest(
        synthetic_dataset,
        SimulationConfig(n_simulations=100, random_seed=42, monthly_horizon=48),
    )


def _example(**overrides) -> HistoricalCompanyCatalystExample:
    data = {
        "example_id": "ex-001",
        "ticker": "SYN1",
        "company_name": "Synthetic Bio 1",
        "as_of_date": "2022-03-31",
        "cash": 100_000_000,
        "marketable_securities": 5_000_000,
        "quarterly_operating_cash_burn": 30_000_000,
        "debt": 0,
        "market_cap": 300_000_000,
        "trial_phase": "phase_2",
        "trial_status": "recruiting",
        "stated_catalyst_date": "2023-03-31",
        "actual_readout_date": "2023-04-15",
        "catalyst_type": "primary_readout",
        "financing_before_catalyst": False,
        "financing_type": "none",
        "program_discontinued_before_catalyst": False,
        "clinical_outcome": "positive",
    }
    data.update(overrides)
    return HistoricalCompanyCatalystExample(**data)


class _MockCapitalToCatalyst:
    probability_cashout_before_catalyst = 0.2
    probability_reaches_catalyst = 0.7


class _MockValuation:
    p_refinancing_success = 0.3
    p_distressed_financing = 0.2
    p_program_discontinuation = 0.1


class _MockAudit:
    capital_to_catalyst = _MockCapitalToCatalyst()
    valuation = _MockValuation()


class _ExactValuation:
    p_refinancing_success = 0.9
    p_distressed_financing = 0.0
    p_program_discontinuation = 0.0
    p_any_financing_event_before_catalyst = 0.25
    p_financing_pressure_before_catalyst = 0.10
    p_program_discontinuation_before_catalyst = 0.05
    p_clean_refinancing_before_catalyst = 0.15
    p_distressed_refinancing_before_catalyst = 0.05
    p_partnership_before_catalyst = 0.05
    p_debt_or_royalty_before_catalyst = 0.0
    p_cash_exhaustion_before_catalyst = 0.0
    p_nondilutive_financing_before_catalyst = 0.05
    p_dilutive_financing_before_catalyst = 0.20


class _ExactAudit:
    capital_to_catalyst = _MockCapitalToCatalyst()
    valuation = _ExactValuation()


class TestHistoricalSchemas:
    def test_valid_example_accepts_point_in_time_case(self):
        ex = _example()
        assert ex.as_of_date.isoformat() == "2022-03-31"
        assert ex.financing_type == "none"

    def test_invalid_date_ordering_fails(self):
        with pytest.raises(ValidationError):
            _example(actual_readout_date="2022-01-01")

    def test_invalid_dilution_fails(self):
        with pytest.raises(ValidationError):
            _example(estimated_dilution=1.2)

    def test_financing_true_cannot_use_none_type(self):
        with pytest.raises(ValidationError):
            _example(financing_before_catalyst=True, financing_type="none")

    def test_dataset_schema_wraps_examples(self):
        ds = HistoricalDataset(
            dataset_id="synthetic-test",
            examples=[_example()],
            synthetic=True,
            source_description="Synthetic unit-test data.",
        )
        assert ds.n_examples == 1


class TestValidationMetrics:
    def test_binary_metrics_known_values(self):
        y_true = [0, 1, 1, 0]
        y_prob = [0.1, 0.8, 0.6, 0.3]

        assert brier_score(y_true, y_prob) == pytest.approx(0.075)
        assert log_loss_binary(y_true, y_prob) > 0

    def test_calibration_buckets_and_ece(self):
        table = calibration_by_bucket([0, 1, 1, 0], [0.1, 0.3, 0.7, 0.9], buckets=[0, 0.5, 1])

        assert len(table) == 2
        assert table[0].n_examples == 2
        assert table[1].n_examples == 2
        assert expected_calibration_error([0, 1, 1, 0], [0.1, 0.3, 0.7, 0.9], [0, 0.5, 1]) >= 0

    def test_confusion_matrix_at_threshold(self):
        cm = confusion_matrix_at_threshold([0, 1, 1, 0], [0.1, 0.8, 0.4, 0.7], threshold=0.5)
        assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}

    def test_calibration_diagnostics_direction(self):
        over = calibration_diagnostics(mean_predicted_probability=0.37, observed_event_rate=0.11)
        under = calibration_diagnostics(mean_predicted_probability=0.10, observed_event_rate=0.30)
        near = calibration_diagnostics(mean_predicted_probability=0.31, observed_event_rate=0.30)

        assert over["calibration_direction"] == "overpredicting"
        assert over["overprediction_gap"] == pytest.approx(0.26)
        assert under["calibration_direction"] == "underpredicting"
        assert under["underprediction_gap"] == pytest.approx(0.20)
        assert near["calibration_direction"] == "approximately_calibrated"


class TestTargetProbabilityMapping:
    def test_financing_before_catalyst_uses_financing_states_not_raw_cashout(self):
        probability, note = probability_for_target(_MockAudit(), "financing_before_catalyst")

        assert probability == pytest.approx(0.6)
        assert probability > _MockAudit.capital_to_catalyst.probability_cashout_before_catalyst
        assert "financing-state" in note

    def test_target_mapping_clamps_probabilities(self):
        audit = _MockAudit()
        audit.valuation.p_refinancing_success = 0.8
        audit.valuation.p_distressed_financing = 0.7
        audit.valuation.p_program_discontinuation = 0.4

        probability, _ = probability_for_target(audit, "financing_before_catalyst")

        assert probability == pytest.approx(1.0)

    def test_reached_label_requires_actual_readout_date(self):
        missing = _example(actual_readout_date=None, financing_before_catalyst=False, program_discontinued_before_catalyst=False)
        clean = _example(actual_readout_date="2023-04-15", financing_before_catalyst=False, program_discontinued_before_catalyst=False)
        financed = _example(actual_readout_date="2023-04-15", financing_before_catalyst=True, financing_type="clean_refinancing")
        discontinued = _example(actual_readout_date="2023-04-15", program_discontinued_before_catalyst=True)

        assert extract_actual_label(missing, "reached_catalyst_before_financing_pressure") is False
        assert extract_actual_label(clean, "reached_catalyst_before_financing_pressure") is True
        assert extract_actual_label(financed, "reached_catalyst_before_financing_pressure") is False
        assert extract_actual_label(discontinued, "reached_catalyst_before_financing_pressure") is False

    def test_cash_distress_blocks_reached_label(self):
        distressed = _example(
            actual_readout_date="2023-04-15",
            financing_before_catalyst=False,
            program_discontinued_before_catalyst=False,
            cash_distress_or_going_concern_before_catalyst=True,
        )

        assert extract_actual_label(distressed, "reached_catalyst_before_financing_pressure") is False

    def test_financing_target_uses_exact_field_when_available(self):
        probability, note = probability_for_target(_ExactAudit(), "financing_before_catalyst")

        assert probability == pytest.approx(0.25)
        assert "exact financing-state fields used" in note.lower()

    def test_reached_target_subtracts_pressure_not_any_financing(self):
        probability, _ = probability_for_target(_ExactAudit(), "reached_catalyst_before_financing_pressure")

        assert probability == pytest.approx(0.60)

    def test_split_validation_targets_are_registered(self):
        for target in (
            "reached_public_readout",
            "reached_without_any_financing_event",
            "reached_without_dilutive_financing",
            "reached_without_distress",
            "failed_before_readout_due_to_science",
            "failed_before_readout_due_to_finance",
        ):
            assert target in TARGET_DEFINITIONS
            probability, note = probability_for_target(_ExactAudit(), target)
            assert 0.0 <= probability <= 1.0
            assert note


class TestSyntheticBacktest:
    def test_synthetic_dataset_loads(self, synthetic_dataset):
        assert synthetic_dataset.synthetic is True
        assert synthetic_dataset.n_examples >= 30
        assert "synthetic" in synthetic_dataset.source_description.lower()

    def test_backtest_runs_end_to_end_on_synthetic_data(self, synthetic_dataset, synthetic_backtest_result):
        result = synthetic_backtest_result

        assert result.synthetic is True
        assert result.calibration_status == "synthetic_test_only"
        assert result.n_examples == synthetic_dataset.n_examples
        assert result.metric_summary.n_examples == synthetic_dataset.n_examples
        assert len(result.per_example_results) == synthetic_dataset.n_examples
        assert result.metric_summary.brier_score >= 0
        assert result.metric_summary.calibration_direction in {
            "overpredicting",
            "underpredicting",
            "approximately_calibrated",
        }
        assert "financing_before_catalyst" in TARGET_DEFINITIONS
        assert result.per_example_results[0].probability_mapping_note
        assert 0.0 <= result.per_example_results[0].predicted_financing_before_catalyst <= 1.0

    def test_backtest_report_contains_synthetic_warning(self, synthetic_backtest_result):
        report = generate_backtest_report(synthetic_backtest_result)

        assert "CatalystLens Backtest Report" in report
        assert "synthetic test data and does not validate model performance" in report
        assert "Label Definition and Probability Mapping" in report
        assert "broader than cash exhaustion" in report
        assert "p_any_financing_event_before_catalyst" in report
        assert "p_financing_pressure_before_catalyst" in report
        assert "clean/proactive financing" in report

    def test_target_values_use_financing_probability_field(self, synthetic_backtest_result):
        result = synthetic_backtest_result
        assert any(
            row.predicted_financing_before_catalyst != row.predicted_cashout_risk
            for row in result.per_example_results
        )

    def test_prediction_error_table_exports_mapping_note(self, tmp_path, synthetic_backtest_result):
        result = synthetic_backtest_result
        path = tmp_path / "errors.csv"

        write_prediction_error_table(result, path)
        text = path.read_text()

        assert "absolute_error" in text
        assert "probability_mapping_note" in text
        assert result.per_example_results[0].example_id in text


class TestModelRegistryValidationMetadata:
    def test_model_card_can_store_validation_metrics(self, tmp_path):
        registry = ModelRegistry(tmp_path)
        card = ModelArtifactCard(
            artifact_id="solvency_test_v1",
            model_family="solvency",
            training_cutoff_date="2025-12-31",
            feature_schema_version="validation_v1",
            config_hash="abc123",
            training_dataset_id="train-v1",
            validation_dataset_id="synthetic-v1",
            n_training_examples=100,
            n_validation_examples=40,
            validation_metrics={"brier_score": 0.2, "expected_calibration_error": 0.05},
            calibration_status="synthetic_test_only",
            validation_report_path="outputs/backtest_report.md",
        )

        registry.save(card)
        loaded = registry.load("solvency_test_v1")

        assert loaded.validation_metrics["brier_score"] == pytest.approx(0.2)
        assert loaded.n_validation_examples == 40

    def test_synthetic_validation_cannot_be_externally_validated(self):
        with pytest.raises(ValidationError):
            ModelArtifactCard(
                artifact_id="bad",
                model_family="solvency",
                training_cutoff_date="2025-12-31",
                feature_schema_version="validation_v1",
                config_hash="abc123",
                validation_dataset_id="synthetic-v1",
                calibration_status="externally_validated",
            )

    def test_missing_metrics_keeps_status_insufficient_or_synthetic(self):
        card = ModelArtifactCard(
            artifact_id="empty",
            model_family="solvency",
            training_cutoff_date="2025-12-31",
            feature_schema_version="validation_v1",
            config_hash="abc123",
        )

        assert card.calibration_status == "insufficient_data"


def test_validation_status_endpoint_reports_research_mode():
    with TestClient(app) as client:
        response = client.get("/validation/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine_mode"] == "research_mode"
    assert payload["calibration_status"] in {"insufficient_data", "synthetic_test_only"}
    assert payload["synthetic_metrics_are_validation"] is False
