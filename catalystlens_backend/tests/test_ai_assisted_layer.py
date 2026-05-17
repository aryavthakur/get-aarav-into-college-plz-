import csv

import pytest

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
from app.ai.claim_extraction import (
    extract_catalyst_guidance,
    extract_financing_event,
    extract_program_discontinuation,
    extract_runway_guidance,
)
from app.ai.error_diagnosis import diagnose_prediction_error
from app.ai.feature_enrichment import enrich_company_features
from app.ai.schemas import AIBacktestErrorDiagnosis, AIExtractionResult
from training.validation.backtest import load_historical_examples, run_backtest
from training.validation.backtest_report import generate_backtest_report
from training.validation.run_backtest import write_prediction_error_table


DATASET_PATH = "training/datasets/example_historical_biotech_panel.csv"


def _audit_request(use_ai: bool = False) -> AuditRequest:
    return AuditRequest(
        financial=CompanyFinancialInput(
            company_name="AIBio",
            ticker="AIB",
            cash_on_hand=120_000_000,
            marketable_securities=10_000_000,
            quarterly_operating_cash_burn=18_000_000,
            market_cap=650_000_000,
            debt=5_000_000,
            going_concern_flag=False,
            biotech_market_condition_score=7.0,
        ),
        clinical=ClinicalCatalystInput(
            asset_name="AIB-101",
            indication="Oncology",
            trial_phase="phase_2",
            trial_status="recruiting",
            stated_months_to_catalyst=14,
            public_readout_months=14,
            enrollment_target=120,
            enrollment_completed=80,
            enrollment_rate_per_month=8,
            number_of_sites=20,
            catalyst_type="primary_readout",
        ),
        success_probability=SuccessProbabilityInput(
            trial_phase="phase_2",
            disease_area="oncology",
            modality="cell therapy",
            positive_signals=["validated_biomarker"],
            negative_signals=["small_sample_size"],
        ),
        valuation=ValuationInput(asset_value_success=700_000_000),
        disclosure=DisclosureInput(
            company_narrative_distribution={
                "runway_strength": 0.7,
                "clinical_timeline_confidence": 0.7,
                "dilution_risk": 0.3,
                "trial_maturity": 0.6,
                "endpoint_strength": 0.6,
                "pipeline_diversification": 0.4,
            },
            structured_audit_distribution={
                "runway_strength": 0.6,
                "clinical_timeline_confidence": 0.6,
                "dilution_risk": 0.4,
                "trial_maturity": 0.5,
                "endpoint_strength": 0.5,
                "pipeline_diversification": 0.4,
            },
        ),
        simulation=SimulationConfig(
            n_simulations=300,
            random_seed=123,
            monthly_horizon=48,
            use_ai_feature_enrichment=use_ai,
        ),
    )


class TestAISchemas:
    def test_ai_schemas_validate_method_status(self):
        extraction = AIExtractionResult(
            raw_text="cash runway into 2027",
            extracted_value="into 2027",
            normalized_value="2027",
            extraction_type="runway_guidance",
            confidence=0.8,
            requires_human_review=True,
        )
        diagnosis = AIBacktestErrorDiagnosis(
            example_id="ex-1",
            company_name="Example Bio",
            ticker="EXB",
            target="financing_before_catalyst",
            y_true=1,
            y_prob=0.2,
            absolute_error=0.8,
            error_type="false_negative",
            diagnosed_failure_mode="partnership_underpredicted",
            likely_missing_features=["partnerability_score"],
            suggested_model_patch="Add partnerability features.",
            confidence=0.75,
        )

        assert extraction.confidence == pytest.approx(0.8)
        assert diagnosis.method_status == "heuristic_ai_assisted"


class TestAIErrorDiagnosis:
    def test_identifies_partnership_underpredicted(self):
        result = diagnose_prediction_error({
            "example_id": "ex-1",
            "company_name": "Partner Bio",
            "ticker": "PRTN",
            "target": "financing_before_catalyst",
            "y_true": 1,
            "y_prob": 0.2,
            "financing_type": "partnership",
        })

        assert result.error_type == "false_negative"
        assert result.diagnosed_failure_mode == "partnership_underpredicted"
        assert "partnerability_score" in result.likely_missing_features

    def test_identifies_proactive_clean_financing_underpredicted(self):
        result = diagnose_prediction_error({
            "example_id": "ex-2",
            "company_name": "Clean Bio",
            "ticker": "CLN",
            "target": "financing_before_catalyst",
            "y_true": 1,
            "y_prob": 0.2,
            "financing_type": "clean_refinancing",
        })

        assert result.diagnosed_failure_mode == "proactive_financing_underpredicted"
        assert "market_window_strength" in result.likely_missing_features

    def test_identifies_scientific_discontinuation_underpredicted(self):
        result = diagnose_prediction_error({
            "example_id": "ex-3",
            "company_name": "Safety Bio",
            "ticker": "SFTY",
            "target": "program_discontinued_before_catalyst",
            "y_true": 1,
            "y_prob": 0.1,
        })

        assert result.diagnosed_failure_mode == "scientific_discontinuation_underpredicted"
        assert "modality_safety_prior" in result.likely_missing_features


class TestAIFeatureEnrichment:
    def test_feature_enrichment_scores_are_bounded(self):
        result = enrich_company_features({
            "disease_area": "oncology",
            "modality": "gene therapy platform",
            "trial_phase": "phase_2",
            "market_cap": 800_000_000,
            "simple_runway_months": 18,
            "posterior_pos": 0.42,
            "trial_status": "recruiting",
        })

        scores = [
            result.partnerability_score,
            result.proactive_financing_likelihood,
            result.scientific_discontinuation_risk_score,
            result.safety_sensitive_modality_score,
            result.management_narrative_optimism_score,
            result.source_grounding_quality,
        ]
        assert all(score is None or 0.0 <= score <= 1.0 for score in scores)
        assert result.requires_human_review is True


class TestClaimExtraction:
    def test_extracts_common_guidance_phrases(self):
        runway = extract_runway_guidance("Management said cash runway into 2027.")
        catalyst = extract_catalyst_guidance("Topline data expected in H1 2025.")
        financing = extract_financing_event("The company announced $150 million public offering.")
        discontinued = extract_program_discontinuation("The company will discontinue development of ABC-101.")

        assert runway.normalized_value == "2027"
        assert catalyst.extraction_type == "catalyst_guidance"
        assert financing.normalized_value == "$150 million public offering"
        assert discontinued.normalized_value == "program_discontinuation"


class TestAIBacktestIntegration:
    def test_diagnose_errors_adds_csv_columns_and_report_section(self, tmp_path):
        dataset = load_historical_examples(DATASET_PATH)
        result = run_backtest(dataset, target_name="financing_before_catalyst", diagnose_errors=True)
        csv_path = tmp_path / "errors.csv"

        write_prediction_error_table(result, csv_path, diagnose_errors=True)
        rows = list(csv.DictReader(csv_path.open()))
        report = generate_backtest_report(result)

        assert "diagnosed_failure_mode" in rows[0]
        assert "ai_method_status" in rows[0]
        assert "AI-Assisted Error Diagnosis" in report
        assert "heuristic AI-assisted diagnosis" in report


class TestAIAuditIntegration:
    def test_default_audit_does_not_include_ai_enrichment(self):
        result = run_full_audit(_audit_request(use_ai=False))

        assert result.ai_feature_enrichment is None

    def test_ai_enrichment_is_optional_and_does_not_overwrite_core_probabilities(self):
        base = run_full_audit(_audit_request(use_ai=False))
        enriched = run_full_audit(_audit_request(use_ai=True))

        assert enriched.ai_feature_enrichment is not None
        assert 0.0 <= enriched.ai_feature_enrichment.partnerability_score <= 1.0
        assert enriched.capital_to_catalyst.probability_cashout_before_catalyst == pytest.approx(
            base.capital_to_catalyst.probability_cashout_before_catalyst
        )
        assert "AI-Assisted Feature Enrichment" in enriched.markdown_report
