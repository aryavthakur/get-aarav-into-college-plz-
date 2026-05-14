"""
Tests for the Monte Carlo engine and capital-to-catalyst analysis.
"""

import pytest
import numpy as np

from app.engines.capital_to_catalyst import classify_capital_risk, run_capital_to_catalyst_analysis
from app.engines.monte_carlo import run_full_audit
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


def _make_audit_request(
    cash_on_hand: float = 100_000_000,
    quarterly_burn: float = 15_000_000,
    stated_months: float = 18,
    enrollment_completed: int = 60,
    enrollment_target: int = 120,
    enrollment_rate: float = 8,
    n_simulations: int = 3000,
    seed: int = 42,
    positive_signals=None,
    negative_signals=None,
) -> AuditRequest:
    if positive_signals is None:
        positive_signals = ["randomized_controlled_design", "validated_biomarker"]
    if negative_signals is None:
        negative_signals = ["small_sample_size"]

    return AuditRequest(
        financial=CompanyFinancialInput(
            company_name="TestCo",
            ticker="TST",
            cash_on_hand=cash_on_hand,
            marketable_securities=10_000_000,
            quarterly_operating_cash_burn=quarterly_burn,
            quarterly_burn_history=[
                QuarterlyBurnEntry(quarter="2023-Q1", operating_cash_burn=quarterly_burn * 0.85),
                QuarterlyBurnEntry(quarter="2023-Q2", operating_cash_burn=quarterly_burn * 0.95),
                QuarterlyBurnEntry(quarter="2023-Q3", operating_cash_burn=quarterly_burn),
            ],
            market_cap=300_000_000,
            debt=5_000_000,
            going_concern_flag=False,
            recent_financing_flag=False,
            months_since_last_raise=10.0,
            biotech_market_condition_score=5.0,
            pipeline_concentration_score=0.7,
        ),
        clinical=ClinicalCatalystInput(
            asset_name="TST-001",
            indication="Test Indication",
            trial_phase="phase_2",
            trial_status="recruiting",
            stated_months_to_catalyst=stated_months,
            enrollment_target=enrollment_target,
            enrollment_completed=enrollment_completed,
            enrollment_rate_per_month=enrollment_rate,
            number_of_sites=15,
            indication_complexity_score=0.5,
            endpoint_complexity_score=0.5,
            regulatory_complexity_score=0.5,
            catalyst_type="primary_readout",
        ),
        success_probability=SuccessProbabilityInput(
            trial_phase="phase_2",
            positive_signals=positive_signals,
            negative_signals=negative_signals,
        ),
        valuation=ValuationInput(
            asset_value_success=500_000_000,
            downside_value=10_000_000,
            annual_discount_rate=0.12,
            expected_dilution_if_refinanced=0.25,
            financing_penalty_strength=0.6,
        ),
        disclosure=DisclosureInput(
            company_narrative_distribution={
                "runway_strength": 0.7,
                "clinical_timeline_confidence": 0.8,
                "dilution_risk": 0.2,
                "trial_maturity": 0.6,
                "endpoint_strength": 0.7,
                "pipeline_diversification": 0.3,
            },
            structured_audit_distribution={
                "runway_strength": 0.45,
                "clinical_timeline_confidence": 0.50,
                "dilution_risk": 0.55,
                "trial_maturity": 0.40,
                "endpoint_strength": 0.55,
                "pipeline_diversification": 0.30,
            },
        ),
        simulation=SimulationConfig(
            n_simulations=n_simulations,
            random_seed=seed,
            monthly_horizon=48,
        ),
    )


class TestCapitalToCatalystGap:
    def test_probabilities_sum_to_one(self):
        t_sci = np.array([10.0, 20.0, 30.0, 5.0])
        t_fin = np.array([15.0, 10.0, 35.0, 3.0])
        result = run_capital_to_catalyst_analysis(t_sci, t_fin)
        total = result.probability_reaches_catalyst + result.probability_cashout_before_catalyst
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_probabilities_in_0_1_range(self):
        rng = np.random.default_rng(42)
        t_sci = rng.gamma(shape=5.0, scale=4.0, size=1000)
        t_fin = rng.exponential(scale=20.0, size=1000)
        result = run_capital_to_catalyst_analysis(t_sci, t_fin)
        assert 0.0 <= result.probability_reaches_catalyst <= 1.0
        assert 0.0 <= result.probability_cashout_before_catalyst <= 1.0

    def test_all_funded_gives_zero_cashout_risk(self):
        """When T_fin is always larger than T_sci, cashout prob should be ~0."""
        t_sci = np.ones(1000) * 10.0
        t_fin = np.ones(1000) * 100.0
        result = run_capital_to_catalyst_analysis(t_sci, t_fin)
        assert result.probability_cashout_before_catalyst == pytest.approx(0.0)
        assert result.probability_reaches_catalyst == pytest.approx(1.0)

    def test_all_cashout_gives_one_cashout_risk(self):
        """When T_fin is always less than T_sci, cashout prob should be ~1."""
        t_sci = np.ones(1000) * 100.0
        t_fin = np.ones(1000) * 5.0
        result = run_capital_to_catalyst_analysis(t_sci, t_fin)
        assert result.probability_cashout_before_catalyst == pytest.approx(1.0)

    def test_risk_classification_low(self):
        assert classify_capital_risk(0.10) == "Low Risk"

    def test_risk_classification_moderate(self):
        assert classify_capital_risk(0.40) == "Moderate Risk"

    def test_risk_classification_high(self):
        assert classify_capital_risk(0.65) == "High Risk"

    def test_risk_classification_critical(self):
        assert classify_capital_risk(0.80) == "Critical Risk"

    def test_median_gap_positive_when_funded(self):
        """Median gap should be positive when T_fin >> T_sci."""
        t_sci = np.ones(1000) * 10.0
        t_fin = np.ones(1000) * 30.0
        result = run_capital_to_catalyst_analysis(t_sci, t_fin)
        assert result.median_gap_months > 0

    def test_median_gap_negative_when_cashout_dominates(self):
        """Median gap should be negative when T_fin << T_sci."""
        t_sci = np.ones(1000) * 30.0
        t_fin = np.ones(1000) * 5.0
        result = run_capital_to_catalyst_analysis(t_sci, t_fin)
        assert result.median_gap_months < 0


class TestCapitalToCatalystObvious:
    """
    These tests verify the model behaves correctly in obvious extreme cases:
      - Well-funded + near catalyst → low cashout risk
      - Poorly funded + distant catalyst → high cashout risk
    """

    def test_well_funded_near_catalyst_low_cashout_risk(self):
        """
        Lots of cash, low burn, catalyst in 6 months.
        P(cashout before catalyst) should be low.
        """
        request = _make_audit_request(
            cash_on_hand=300_000_000,  # ~33 months of runway
            quarterly_burn=9_000_000,
            stated_months=6,
            enrollment_completed=100,
            enrollment_target=120,
            enrollment_rate=15,
            n_simulations=5000,
        )
        result = run_full_audit(request)
        cashout_prob = result.capital_to_catalyst.probability_cashout_before_catalyst
        assert cashout_prob < 0.50, (
            f"Well-funded near-catalyst company should have low cashout risk, got {cashout_prob:.2%}"
        )

    def test_poorly_funded_distant_catalyst_high_cashout_risk(self):
        """
        Very little cash, high burn, catalyst far away.
        P(cashout before catalyst) should be high.
        """
        request = _make_audit_request(
            cash_on_hand=20_000_000,  # <2 months of runway
            quarterly_burn=30_000_000,
            stated_months=30,
            enrollment_completed=10,
            enrollment_target=200,
            enrollment_rate=3,
            n_simulations=5000,
            positive_signals=[],
            negative_signals=["small_sample_size", "slow_enrollment", "prior_failed_trials"],
        )
        result = run_full_audit(request)
        cashout_prob = result.capital_to_catalyst.probability_cashout_before_catalyst
        assert cashout_prob > 0.50, (
            f"Poorly-funded distant-catalyst company should have high cashout risk, got {cashout_prob:.2%}"
        )


class TestMonteCarloReproducibility:
    def test_same_seed_produces_same_result(self):
        request = _make_audit_request(n_simulations=1000, seed=123)
        result1 = run_full_audit(request)
        result2 = run_full_audit(request)
        assert (
            result1.capital_to_catalyst.probability_cashout_before_catalyst
            == result2.capital_to_catalyst.probability_cashout_before_catalyst
        )
        assert result1.valuation.mean_value == result2.valuation.mean_value

    def test_different_seed_produces_different_but_close_result(self):
        req1 = _make_audit_request(n_simulations=5000, seed=1)
        req2 = _make_audit_request(n_simulations=5000, seed=2)
        r1 = run_full_audit(req1)
        r2 = run_full_audit(req2)
        # Different seeds → different samples, but results should be close
        p1 = r1.capital_to_catalyst.probability_cashout_before_catalyst
        p2 = r2.capital_to_catalyst.probability_cashout_before_catalyst
        assert abs(p1 - p2) < 0.10, f"Results too divergent across seeds: {p1:.3f} vs {p2:.3f}"


class TestAuditResponseStructure:
    def test_all_required_top_level_fields_present(self):
        request = _make_audit_request(n_simulations=1000)
        result = run_full_audit(request)

        assert result.company_name == "TestCo"
        assert result.ticker == "TST"
        assert result.solvency is not None
        assert result.success_probability is not None
        assert result.milestone_timing is not None
        assert result.capital_to_catalyst is not None
        assert result.valuation is not None
        assert result.burn_regime is not None
        assert result.disclosure_consistency is not None
        assert result.final_summary is not None
        assert len(result.warnings) > 0
        assert len(result.assumptions) > 0
        assert len(result.markdown_report) > 100

    def test_probabilities_in_valid_range(self):
        request = _make_audit_request(n_simulations=1000)
        result = run_full_audit(request)
        ctc = result.capital_to_catalyst
        assert 0.0 <= ctc.probability_cashout_before_catalyst <= 1.0
        assert 0.0 <= ctc.probability_reaches_catalyst <= 1.0

    def test_scenarios_generated(self):
        request = _make_audit_request(n_simulations=1000)
        result = run_full_audit(request)
        assert len(result.final_summary.scenarios) == 5
        for sc in result.final_summary.scenarios:
            assert 0.0 <= sc.probability_cashout_before_catalyst <= 1.0

    def test_sensitivity_generated(self):
        request = _make_audit_request(n_simulations=1000)
        result = run_full_audit(request)
        assert len(result.final_summary.sensitivity) > 0

    def test_markdown_report_contains_key_sections(self):
        request = _make_audit_request(n_simulations=1000)
        result = run_full_audit(request)
        report = result.markdown_report
        assert "Executive Investment Summary" in report
        assert "Capital-to-Catalyst" in report
        assert "Bayesian" in report
        assert "Disclaimer" in report or "not investment advice" in report.lower()
