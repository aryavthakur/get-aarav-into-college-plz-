"""
Regression tests for the deep-research remediation recommendations.
"""

import pytest

from app.engines.monte_carlo import run_full_audit, spawn_streams
from app.models.schemas import (
    AuditRequest,
    ClinicalCatalystInput,
    CompanyFinancialInput,
    DisclosureInput,
    QuarterlyBurnEntry,
    SimulationConfig,
    SuccessProbabilityInput,
    ValuationInput,
)


def _request(**overrides) -> AuditRequest:
    clinical = ClinicalCatalystInput(
        asset_name="TST-001",
        indication="Oncology",
        trial_phase="phase_2",
        trial_status="recruiting",
        stated_months_to_catalyst=12,
        enrollment_target=120,
        enrollment_completed=80,
        enrollment_rate_per_month=10,
        number_of_sites=12,
        indication_complexity_score=0.5,
        endpoint_complexity_score=0.5,
        regulatory_complexity_score=0.5,
        catalyst_type="primary_readout",
        followup_months_after_enrollment=2.0,
        data_cleaning_months=1.0,
        analysis_months=1.0,
        disclosure_lag_months=2.0,
    )
    financial = CompanyFinancialInput(
        company_name="TestCo",
        ticker="TST",
        cash_on_hand=30_000_000,
        marketable_securities=0,
        quarterly_operating_cash_burn=15_000_000,
        quarterly_burn_history=[
            QuarterlyBurnEntry(quarter="2025-Q1", operating_cash_burn=12_000_000),
            QuarterlyBurnEntry(quarter="2025-Q2", operating_cash_burn=13_000_000),
            QuarterlyBurnEntry(quarter="2025-Q3", operating_cash_burn=15_000_000),
        ],
        market_cap=200_000_000,
        debt=0,
        months_since_last_raise=9,
    )
    request = AuditRequest(
        financial=financial,
        clinical=clinical,
        success_probability=SuccessProbabilityInput(
            trial_phase="phase_2",
            disease_area="oncology",
            modality="small_molecule",
            endpoint_family="surrogate",
            positive_signals=["validated_biomarker"],
            negative_signals=[],
        ),
        valuation=ValuationInput(
            asset_value_success=500_000_000,
            downside_value=10_000_000,
            expected_dilution_if_refinanced=0.25,
        ),
        disclosure=DisclosureInput(
            company_narrative_distribution={
                "runway_strength": 0.7,
                "clinical_timeline_confidence": 0.7,
                "dilution_risk": 0.2,
                "trial_maturity": 0.7,
                "endpoint_strength": 0.6,
                "pipeline_diversification": 0.3,
            },
            structured_audit_distribution={
                "runway_strength": 0.4,
                "clinical_timeline_confidence": 0.5,
                "dilution_risk": 0.6,
                "trial_maturity": 0.5,
                "endpoint_strength": 0.5,
                "pipeline_diversification": 0.3,
            },
        ),
        simulation=SimulationConfig(n_simulations=1200, random_seed=123),
    )
    return request.model_copy(update=overrides)


class TestRNGStreams:
    def test_spawn_streams_are_exactly_reproducible_and_independent(self):
        s1 = spawn_streams(42)
        s2 = spawn_streams(42)

        assert s1.cash.random() == pytest.approx(s2.cash.random())
        assert s1.financing.random() == pytest.approx(s2.financing.random())
        assert s1.science.random() == pytest.approx(s2.science.random())
        assert s1.valuation.random() == pytest.approx(s2.valuation.random())
        assert s1.cash.random() != s1.financing.random()


class TestTimingDecomposition:
    def test_public_readout_is_separate_from_primary_completion(self):
        result = run_full_audit(_request())
        timing = result.milestone_timing

        assert timing.primary_completion_months < timing.public_readout_months
        assert timing.public_readout_lag_months == pytest.approx(4.0)
        assert timing.enrollment_component_months == pytest.approx(4.0)
        assert "public readout" in result.markdown_report.lower()


class TestCashPathAuditIntegration:
    def test_cash_path_caps_financial_timing_when_mechanical_runway_is_shorter(self):
        request = _request()
        result = run_full_audit(request)

        assert result.cash_path.cash_exhaustion_month == 6
        assert result.capital_to_catalyst.median_financial_failure_time <= 6.0


class TestModelGovernanceContract:
    def test_audit_response_contains_validation_snapshot_and_provenance(self):
        result = run_full_audit(_request())

        assert result.validation_snapshot.solvency_calibration_status == "research_mode"
        assert result.validation_snapshot.pos_ppc_status == "not_available"
        assert result.provenance.provenance_status == "manual_inputs_unverified"
        assert "training_cutoff_date" in result.model_version.model_dump()
