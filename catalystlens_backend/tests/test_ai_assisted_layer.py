import csv
from datetime import datetime, timezone

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
from app.engines.financing_strategy import estimate_financing_strategy
from app.engines.program_discontinuation import estimate_program_discontinuation
from training.validation.backtest_report import generate_backtest_report
from training.validation.run_backtest import write_prediction_error_table
from training.validation.schemas import (
    BacktestMetricSummary,
    BacktestResult,
    CalibrationBucket,
    PerExampleBacktestResult,
)


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

    def test_identifies_moderate_clean_financing_false_negative(self):
        result = diagnose_prediction_error({
            "example_id": "ex-4",
            "company_name": "Clean Moderate Bio",
            "ticker": "CMB",
            "target": "financing_before_catalyst",
            "y_true": 1,
            "y_prob": 0.42,
            "financing_type": "clean_refinancing",
        })

        assert result.diagnosed_failure_mode == "proactive_financing_underpredicted"

    def test_identifies_moderate_partnership_false_negative(self):
        result = diagnose_prediction_error({
            "example_id": "ex-5",
            "company_name": "Partner Moderate Bio",
            "ticker": "PMB",
            "target": "financing_before_catalyst",
            "y_true": 1,
            "y_prob": 0.45,
            "financing_type": "partnership",
        })

        assert result.diagnosed_failure_mode == "partnership_underpredicted"

    def test_identifies_moderate_program_discontinuation_false_negative(self):
        result = diagnose_prediction_error({
            "example_id": "ex-6",
            "company_name": "Disco Moderate Bio",
            "ticker": "DMB",
            "target": "program_discontinued_before_catalyst",
            "y_true": 1,
            "y_prob": 0.40,
        })

        assert result.diagnosed_failure_mode == "scientific_discontinuation_underpredicted"


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

    @pytest.mark.parametrize("phrase", [
        "The company expects cash to be sufficient to fund operations through Q2 2026.",
        "Cash will fund operating expenses and capital expenditure requirements through at least the second quarter of 2026.",
        "The cash runway extends into the first half of 2027.",
        "The company is expected to fund operations into 2027.",
    ])
    def test_extracts_expanded_runway_guidance(self, phrase):
        result = extract_runway_guidance(phrase)

        assert result.normalized_value is not None
        assert result.confidence >= 0.6

    @pytest.mark.parametrize("phrase", [
        "Topline results expected in Q3 2025.",
        "Data readout anticipated in H1 2026.",
        "Primary completion expected in 2024.",
        "The PDUFA date of January 5, 2024 was assigned.",
    ])
    def test_extracts_expanded_catalyst_guidance(self, phrase):
        result = extract_catalyst_guidance(phrase)

        assert result.normalized_value is not None
        assert result.confidence >= 0.6

    @pytest.mark.parametrize("phrase", [
        "The financing generated gross proceeds of approximately $150 million.",
        "The company entered into a private placement with institutional investors.",
        "The company established an ATM facility.",
        "The company completed debt financing.",
        "The company received an upfront payment under collaboration agreement.",
    ])
    def test_extracts_expanded_financing_events(self, phrase):
        result = extract_financing_event(phrase)

        assert result.normalized_value is not None
        assert result.confidence >= 0.6

    @pytest.mark.parametrize("phrase", [
        "The company will discontinue development of ABC-101.",
        "The board approved a strategic restructuring.",
        "The company announced a strategic restructuring to discontinue development of ABC-101.",
        "The company announced a strategic restructuring and pipeline prioritization.",
        "The company announced a strategic restructuring including pause of enrollment.",
        "The company announced a strategic restructuring including termination of the study.",
        "The sponsor will pause enrollment in the study.",
        "The sponsor paused enrollment for safety review.",
        "The company will terminate the study after review.",
        "The company terminated the study due to futility.",
        "The company will terminate development of ABC-101.",
        "The company terminated development of ABC-101.",
        "The company announced a strategic restructuring to discontinue development of ABC-101.",
        "The company announced a strategic restructuring and pipeline prioritization.",
        "The company announced a strategic restructuring including pause of enrollment.",
        "The company announced a strategic restructuring including termination of the study.",
    ])
    def test_extracts_program_discontinuation_variants(self, phrase):
        result = extract_program_discontinuation(phrase)

        assert result.normalized_value == "program_discontinuation"
        assert result.confidence >= 0.6

    @pytest.mark.parametrize("phrase", [
        "Management paused before answering the question.",
        "The lease may terminate at the end of the year.",
        "The company completed a strategic restructuring of its finance department.",
        "The company restructured its lease obligations.",
        "The board approved a strategic restructuring.",
        "The company announced a strategic restructuring of administrative operations.",
    ])
    def test_program_discontinuation_requires_trial_or_program_context(self, phrase):
        result = extract_program_discontinuation(phrase)

        assert result.normalized_value is None


class TestHeuristicTaxonomyEngines:
    def test_financing_strategy_outputs_bounded_probabilities(self):
        result = estimate_financing_strategy(
            months_to_catalyst=12,
            simple_runway_months=9,
            market_cap=600_000_000,
            market_condition_score=7,
            trial_phase="phase_2",
            posterior_pos=0.45,
            catalyst_type="primary_readout",
            recent_positive_signal=True,
            partnerability_score=0.6,
        )

        probs = [
            result.p_proactive_clean_refinancing,
            result.p_partnership_or_nondilutive,
            result.p_debt_or_royalty,
            result.p_distressed_financing,
            result.p_cash_exhaustion,
            result.p_dilutive_financing,
            result.p_nondilutive_financing,
        ]
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert (
            result.p_proactive_clean_refinancing
            + result.p_partnership_or_nondilutive
            + result.p_debt_or_royalty
            + result.p_distressed_financing
            + result.p_cash_exhaustion
            <= 1.0
        )
        assert result.p_nondilutive_financing == pytest.approx(
            min(1.0, result.p_partnership_or_nondilutive + result.p_debt_or_royalty),
            abs=1e-4,
        )
        assert result.method_status == "heuristic"

    def test_program_discontinuation_separates_scientific_and_financial_risk(self):
        result = estimate_program_discontinuation(
            modality="cell therapy",
            disease_area="oncology",
            trial_phase="phase_1",
            trial_status="suspended",
            endpoint_family="safety",
            safety_sensitive_modality_score=0.8,
            prior_human_signal=False,
            open_label_design=True,
            small_sample_size=True,
            single_asset_dependency=0.9,
            clinical_hold_or_safety_pause=True,
            cash_runway_months=5,
            posterior_pos=0.2,
        )

        assert result.p_scientific_discontinuation > result.p_financial_discontinuation
        assert result.p_total_program_discontinuation >= result.p_scientific_discontinuation
        assert result.method_status == "heuristic"


class TestAIBacktestIntegration:
    def test_diagnose_errors_adds_csv_columns_and_report_section(self, tmp_path):
        result = BacktestResult(
            dataset_id="tiny_ai_test",
            synthetic=True,
            n_examples=2,
            generated_at=datetime.now(timezone.utc),
            target_name="financing_before_catalyst",
            metric_summary=BacktestMetricSummary(
                n_examples=2,
                brier_score=0.25,
                log_loss=0.7,
                roc_auc=None,
                expected_calibration_error=0.1,
                calibration_buckets=[
                    CalibrationBucket(bucket_start=0.0, bucket_end=0.5, n_examples=1, mean_predicted_probability=0.25, observed_event_rate=1.0),
                    CalibrationBucket(bucket_start=0.5, bucket_end=1.0, n_examples=1, mean_predicted_probability=0.75, observed_event_rate=0.0),
                ],
                confusion_matrix={"tp": 0, "fp": 1, "tn": 0, "fn": 1},
                event_rate=0.5,
                mean_predicted_probability=0.5,
                overprediction_gap=0.0,
                underprediction_gap=0.0,
                calibration_direction="approximately_calibrated",
            ),
            per_example_results=[
                PerExampleBacktestResult(
                    example_id="ex-1",
                    company_name="Partner Bio",
                    ticker="PRTN",
                    as_of_date="2024-01-01",
                    predicted_cashout_risk=0.1,
                    predicted_financing_before_catalyst=0.25,
                    predicted_distressed_or_cashout_before_catalyst=0.1,
                    predicted_clean_or_nondilutive_financing_before_catalyst=0.2,
                    predicted_program_discontinuation=0.1,
                    predicted_reaches_catalyst_before_financing_pressure=0.7,
                    predicted_reaches_catalyst_before_cashout=0.8,
                    posterior_mean_pos=0.45,
                    actual_financing_before_catalyst=True,
                    actual_distressed_financing_or_cashout=False,
                    actual_reached_catalyst_before_financing_pressure=False,
                    actual_program_discontinued_before_catalyst=False,
                    probability_mapping_note="manual test row",
                    error_type="false_negative",
                    diagnosed_failure_mode="partnership_underpredicted",
                    likely_missing_features=["partnerability_score"],
                    suggested_model_patch="Add partnerability features.",
                    ai_diagnosis_confidence=0.75,
                    ai_method_status="heuristic_ai_assisted",
                    partnerability_score=0.8,
                    proactive_financing_likelihood=0.6,
                    scientific_discontinuation_risk_score=0.2,
                    safety_sensitive_modality_score=0.7,
                    false_negative_financing_event=True,
                    partnership_not_captured=True,
                ),
                PerExampleBacktestResult(
                    example_id="ex-2",
                    company_name="Clean Bio",
                    ticker="CLN",
                    as_of_date="2024-01-01",
                    predicted_cashout_risk=0.2,
                    predicted_financing_before_catalyst=0.75,
                    predicted_distressed_or_cashout_before_catalyst=0.2,
                    predicted_clean_or_nondilutive_financing_before_catalyst=0.7,
                    predicted_program_discontinuation=0.0,
                    predicted_reaches_catalyst_before_financing_pressure=0.2,
                    predicted_reaches_catalyst_before_cashout=0.4,
                    posterior_mean_pos=0.35,
                    actual_financing_before_catalyst=False,
                    actual_distressed_financing_or_cashout=False,
                    actual_reached_catalyst_before_financing_pressure=True,
                    actual_program_discontinued_before_catalyst=False,
                    probability_mapping_note="manual test row",
                    error_type="false_positive",
                    diagnosed_failure_mode="cash_distress_overpredicted",
                    likely_missing_features=["proactive_financing_likelihood"],
                    suggested_model_patch="Check proactive financing.",
                    ai_diagnosis_confidence=0.55,
                    ai_method_status="heuristic_ai_assisted",
                    partnerability_score=0.3,
                    proactive_financing_likelihood=0.5,
                    scientific_discontinuation_risk_score=0.1,
                    safety_sensitive_modality_score=0.2,
                ),
            ],
            warnings=["Synthetic test data only."],
            calibration_status="synthetic_test_only",
        )
        csv_path = tmp_path / "errors.csv"

        write_prediction_error_table(result, csv_path, diagnose_errors=True)
        rows = list(csv.DictReader(csv_path.open()))
        report = generate_backtest_report(result)

        assert "diagnosed_failure_mode" in rows[0]
        assert "ai_method_status" in rows[0]
        assert "partnerability_score" in rows[0]
        assert rows[0]["partnerability_score"] == "0.800000"
        assert "false_negative_financing_event" in rows[0]
        assert "partnership_not_captured" in rows[0]
        assert "AI-Assisted Error Diagnosis" in report
        assert "Diagnostic Flag Counts" in report
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
