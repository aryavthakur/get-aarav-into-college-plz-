"""
Regression tests for the requested mathematical/integration patch.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from app.engines.bayesian_success import run_success_probability_analysis
from app.engines.monte_carlo import apply_cash_path_cap, run_full_audit
from app.models.schemas import (
    AuditRequest,
    ClinicalCatalystInput,
    CompanyFinancialInput,
    DisclosureInput,
    FinancingEventInput,
    QuarterlyBurnEntry,
    SimulationConfig,
    SuccessProbabilityInput,
    ValuationInput,
)
from training.datasets.build_company_panel import (
    build_company_quarter_panel,
    derive_quarterly_operating_cash_burn,
)


def _request(
    cash: float = 2_000_000,
    burn: float = 6_000_000,
    burn_history: bool = True,
    disclosure_audit: dict[str, float] | None = None,
    events: list[FinancingEventInput] | None = None,
    n: int = 500,
) -> AuditRequest:
    hist = [
        QuarterlyBurnEntry(quarter="2026-Q1", operating_cash_burn=burn * 0.8),
        QuarterlyBurnEntry(quarter="2026-Q2", operating_cash_burn=burn),
    ] if burn_history else []
    full_disclosure = {
        "runway_strength": 0.4,
        "clinical_timeline_confidence": 0.5,
        "dilution_risk": 0.6,
        "trial_maturity": 0.5,
        "endpoint_strength": 0.5,
        "pipeline_diversification": 0.3,
    }
    financial = CompanyFinancialInput(
        company_name="PatchCo",
        ticker="PCH",
        cash_on_hand=cash,
        marketable_securities=0,
        quarterly_operating_cash_burn=burn,
        quarterly_burn_history=hist,
        market_cap=100_000_000,
        planned_financing_events=events or [],
    )
    clinical = ClinicalCatalystInput(
        asset_name="PCH-001",
        indication="Oncology",
        trial_phase="phase_2",
        trial_status="recruiting",
        stated_months_to_catalyst=18,
        enrollment_target=100,
        enrollment_completed=40,
        enrollment_rate_per_month=5,
        number_of_sites=10,
    )
    return AuditRequest(
        financial=financial,
        clinical=clinical,
        success_probability=SuccessProbabilityInput(trial_phase="phase_2"),
        valuation=ValuationInput(asset_value_success=300_000_000),
        disclosure=DisclosureInput(
            company_narrative_distribution={
                "runway_strength": 0.8,
                "clinical_timeline_confidence": 0.8,
                "dilution_risk": 0.2,
                "trial_maturity": 0.6,
                "endpoint_strength": 0.6,
                "pipeline_diversification": 0.3,
            },
            structured_audit_distribution=full_disclosure if disclosure_audit is None else disclosure_audit,
        ),
        simulation=SimulationConfig(n_simulations=n, random_seed=77, monthly_horizon=24),
    )


class TestCashPathCapIntegration:
    def test_apply_cash_path_cap_caps_financial_samples(self):
        request = _request(cash=2_000_000, burn=6_000_000)
        t_fin = np.array([4.0, 12.0, 36.0])

        capped, cash_path = apply_cash_path_cap(
            t_fin,
            request.financial,
            request.simulation,
            np.random.default_rng(1),
        )

        assert cash_path.cash_exhaustion_month == 1
        assert capped.tolist() == [1.0, 1.0, 1.0]

    def test_base_scenario_and_sensitivity_use_short_cash_runway_cap(self):
        result = run_full_audit(_request(cash=1_000_000, burn=9_000_000, n=1000))

        assert result.cash_path.cash_exhaustion_month == 1
        assert result.capital_to_catalyst.probability_cashout_before_catalyst == pytest.approx(1.0)
        assert all(sc.probability_cashout_before_catalyst == pytest.approx(1.0) for sc in result.final_summary.scenarios)
        assert all(sp.base_cashout_prob == pytest.approx(1.0) for sp in result.final_summary.sensitivity)

    def test_planned_financing_extends_audit_cash_path_and_report_mentions_it(self):
        without = run_full_audit(_request(cash=2_000_000, burn=6_000_000))
        with_event = run_full_audit(
            _request(
                cash=2_000_000,
                burn=6_000_000,
                events=[FinancingEventInput(month=0, kind="clean_refi", gross_proceeds=20_000_000)],
            )
        )

        assert without.cash_path.cash_exhaustion_month == 1
        assert with_event.cash_path.cash_exhaustion_month > without.cash_path.cash_exhaustion_month
        assert "Planned Financing Events" in with_event.markdown_report
        assert "Clean Refi" in with_event.markdown_report


class TestHierarchicalPoS:
    def test_exact_prior_lookup_changes_phase_prior_and_reports_source(self):
        plain = run_success_probability_analysis(SuccessProbabilityInput(trial_phase="phase_2"))
        stratified = run_success_probability_analysis(
            SuccessProbabilityInput(
                trial_phase="phase_2",
                disease_area="oncology",
                modality="small molecule",
                endpoint_family="surrogate",
            )
        )

        assert stratified.alpha_prior != plain.alpha_prior
        assert stratified.prior_fallback_level == "phase_disease_modality_endpoint"
        assert stratified.prior_source == "disease_modality_endpoint_default_hierarchical_v1"
        assert stratified.prior_confidence > plain.prior_confidence

    def test_missing_stratum_falls_back_without_crashing(self):
        result = run_success_probability_analysis(
            SuccessProbabilityInput(
                trial_phase="phase_2",
                disease_area="oncology",
                modality="unknown_modality",
                endpoint_family="unknown_endpoint",
            )
        )

        assert result.prior_fallback_level == "phase_disease"
        assert result.alpha_prior == pytest.approx(2.4)


class TestClinicalTrialsStatusAndBurnDerivation:
    def test_additional_trial_statuses_are_schema_safe(self):
        for status in ("terminated", "enrolling_by_invitation", "unknown"):
            ClinicalCatalystInput(
                asset_name="A",
                indication="I",
                trial_phase="phase_2",
                trial_status=status,
                stated_months_to_catalyst=12,
                enrollment_target=10,
                enrollment_completed=1,
                enrollment_rate_per_month=1,
            )

    def test_build_panel_status_is_accepted_by_clinical_schema(self):
        rows = build_company_quarter_panel(
            cik="1",
            ticker="T",
            nct_id="NCT1",
            sec_companyfacts={
                "facts": {"us-gaap": {
                    "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [{"fy": 2026, "fp": "Q1", "end": "2026-03-31", "val": 1}]}},
                    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [{"fy": 2026, "fp": "Q1", "end": "2026-03-31", "val": -1}]}}
                }}
            },
            ctgov_study={"protocolSection": {"statusModule": {"overallStatus": "ENROLLING_BY_INVITATION"}, "designModule": {"phases": ["PHASE2"]}}},
            fred_observations={},
        )
        row = rows[0]
        ClinicalCatalystInput(
            asset_name="A",
            indication="I",
            trial_phase=row.trial_phase,
            trial_status=row.trial_status,
            stated_months_to_catalyst=12,
            enrollment_target=10,
            enrollment_completed=1,
            enrollment_rate_per_month=1,
        )

    def test_ytd_cash_flow_is_derived_to_quarterly_burn(self):
        derived = derive_quarterly_operating_cash_burn([
            {"fy": 2026, "fp": "Q1", "end": "2026-03-31", "val": -20_000_000},
            {"fy": 2026, "fp": "Q2", "end": "2026-06-30", "val": -45_000_000},
            {"fy": 2026, "fp": "Q3", "end": "2026-09-30", "val": -75_000_000},
        ])

        assert [item.quarterly_burn for item in derived] == [20_000_000, 25_000_000, 30_000_000]

    def test_missing_prior_quarter_returns_none_with_warning(self):
        derived = derive_quarterly_operating_cash_burn([
            {"fy": 2026, "fp": "Q2", "end": "2026-06-30", "val": -45_000_000},
        ])

        assert derived[0].quarterly_burn is None
        assert "Missing prior quarter" in derived[0].warning


class TestDataQualityCapsAndReportSections:
    def test_empty_structured_disclosure_prevents_high_quality(self):
        result = run_full_audit(_request(disclosure_audit={}))

        assert result.data_quality.overall_completeness <= 0.70
        assert result.data_quality.data_quality_score != "high"

    def test_missing_burn_history_caps_financial_score(self):
        result = run_full_audit(_request(cash=50_000_000, burn=3_000_000, burn_history=False))

        assert result.data_quality.financial_data_completeness <= 0.80

    def test_zero_liquidity_caps_overall_at_half(self):
        result = run_full_audit(_request(cash=0, burn=6_000_000))

        assert result.data_quality.overall_completeness <= 0.50

    def test_report_contains_validation_provenance_and_clock_sections(self):
        result = run_full_audit(_request())
        report = result.markdown_report

        assert "Model Validation Status" in report
        assert "Provenance Appendix" in report
        assert "Financial Clock Decomposition" in report
        assert "manual inputs are unverified" in report
