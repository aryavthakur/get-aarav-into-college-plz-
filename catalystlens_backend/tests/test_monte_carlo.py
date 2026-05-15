"""
Tests for the Monte Carlo engine and capital-to-catalyst analysis.
"""

import pytest
import numpy as np
from pydantic import ValidationError

from app.engines.capital_to_catalyst import classify_capital_risk, run_capital_to_catalyst_analysis
from app.engines.disclosure_consistency import run_disclosure_consistency_analysis
from app.engines.monte_carlo import run_full_audit
from app.engines.valuation import run_valuation_simulation
from app.core.config import CatalystLensConfig
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

    def test_mismatched_sample_lengths_raise_value_error(self):
        t_sci = np.array([10.0, 20.0, 30.0])
        t_fin = np.array([15.0, 10.0])
        with pytest.raises(ValueError, match="equal length"):
            run_capital_to_catalyst_analysis(t_sci, t_fin)


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

    def test_model_version_present_and_valid(self):
        request = _make_audit_request(n_simulations=1000)
        result = run_full_audit(request)
        mv = result.model_version
        assert mv.backend_version == "0.1.0"
        assert mv.coefficient_set == "mvp_untrained_v1"
        assert mv.n_simulations == 1000
        assert len(mv.config_hash) > 0

    def test_data_quality_present_and_valid(self):
        request = _make_audit_request(n_simulations=1000)
        result = run_full_audit(request)
        dq = result.data_quality
        assert 0.0 <= dq.financial_data_completeness <= 1.0
        assert 0.0 <= dq.clinical_data_completeness <= 1.0
        assert 0.0 <= dq.disclosure_data_completeness <= 1.0
        assert 0.0 <= dq.overall_completeness <= 1.0
        assert dq.data_quality_score in ("high", "moderate", "low")

    def test_data_quality_penalises_missing_burn_history(self):
        """Without burn history, financial completeness should be lower than with history."""
        req_with = _make_audit_request(n_simulations=500)
        req_without = _make_audit_request(n_simulations=500)
        # Strip burn history
        fin_data = req_without.financial.model_dump()
        fin_data["quarterly_burn_history"] = []
        from app.models.schemas import CompanyFinancialInput
        req_without = req_without.model_copy(update={"financial": CompanyFinancialInput(**fin_data)})
        result_with = run_full_audit(req_with)
        result_without = run_full_audit(req_without)
        assert result_with.data_quality.financial_data_completeness > result_without.data_quality.financial_data_completeness

    def test_report_contains_data_quality_section(self):
        request = _make_audit_request(n_simulations=1000)
        result = run_full_audit(request)
        assert "Data Quality" in result.markdown_report
        assert "UNCALIBRATED" in result.markdown_report

    def test_report_uses_actual_financing_state_probabilities(self):
        request = _make_audit_request(
            cash_on_hand=12_000_000,
            quarterly_burn=24_000_000,
            stated_months=30,
            n_simulations=3000,
        )
        result = run_full_audit(request)
        report = result.markdown_report
        valuation = result.valuation
        for probability in (
            valuation.p_funded_through_catalyst,
            valuation.p_refinancing_success,
            valuation.p_distressed_financing,
            valuation.p_program_discontinuation,
        ):
            assert f"{probability:.1%}" in report


class TestConfigIsolation:
    """Verify that concurrent requests with different seeds don't contaminate each other."""

    def test_different_seeds_produce_independent_results(self):
        req1 = _make_audit_request(n_simulations=2000, seed=10)
        req2 = _make_audit_request(n_simulations=2000, seed=99)
        r1 = run_full_audit(req1)
        r2 = run_full_audit(req2)
        # Results should differ (different seeds)
        assert r1.valuation.mean_value != r2.valuation.mean_value

    def test_config_hash_stable_across_identical_requests(self):
        req = _make_audit_request(n_simulations=500, seed=42)
        r1 = run_full_audit(req)
        r2 = run_full_audit(req)
        assert r1.model_version.config_hash == r2.model_version.config_hash

    def test_config_hash_changes_when_signal_weights_change(self):
        req = _make_audit_request(n_simulations=500, seed=42)
        cfg_base = CatalystLensConfig()
        cfg_changed = CatalystLensConfig()
        cfg_changed.signal_weights.positive["validated_biomarker"] = 4.25
        r1 = run_full_audit(req, config=cfg_base)
        r2 = run_full_audit(req, config=cfg_changed)
        assert r1.model_version.config_hash != r2.model_version.config_hash

    def test_repeated_audits_give_identical_results(self):
        """Ensures no global state mutation between calls."""
        req = _make_audit_request(n_simulations=2000, seed=777)
        r1 = run_full_audit(req)
        r2 = run_full_audit(req)
        assert r1.capital_to_catalyst.probability_cashout_before_catalyst == pytest.approx(
            r2.capital_to_catalyst.probability_cashout_before_catalyst
        )
        assert r1.valuation.mean_value == pytest.approx(r2.valuation.mean_value)


class TestPoSSensitivityActuallyChanges:
    """PoS sensitivity rows must not be identical to the base case."""

    def test_pos_sensitivity_varies_ev(self):
        request = _make_audit_request(n_simulations=2000)
        result = run_full_audit(request)
        pos_sens = next(
            (s for s in result.final_summary.sensitivity if s.variable == "posterior_pos"),
            None,
        )
        assert pos_sens is not None, "posterior_pos sensitivity variable not found"
        assert pos_sens.low_expected_value != pos_sens.high_expected_value, (
            "Low and high PoS sensitivity must yield different EVs — "
            f"both are {pos_sens.low_expected_value}"
        )
        # Low PoS → lower EV, high PoS → higher EV
        assert pos_sens.low_expected_value < pos_sens.high_expected_value, (
            "Lower PoS should produce lower EV"
        )

    def test_pos_sensitivity_varies_cashout_prob(self):
        """PoS sensitivity should not affect cashout probability (PoS ≠ financing risk)."""
        request = _make_audit_request(n_simulations=2000)
        result = run_full_audit(request)
        pos_sens = next(
            s for s in result.final_summary.sensitivity if s.variable == "posterior_pos"
        )
        # Cashout probability should be roughly equal across PoS levels (PoS is technical, not financial)
        # Allow small Monte Carlo noise (< 15%)
        assert abs(pos_sens.low_cashout_prob - pos_sens.high_cashout_prob) < 0.20, (
            "PoS shifts should not drastically change cashout probability"
        )


class TestFourStateFinancing:
    """Validate that the four-state financing model behaves correctly."""

    def test_financing_state_probs_sum_to_one(self):
        request = _make_audit_request(n_simulations=5000)
        result = run_full_audit(request)
        v = result.valuation
        total = (
            v.p_funded_through_catalyst
            + v.p_refinancing_success
            + v.p_distressed_financing
            + v.p_program_discontinuation
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_well_funded_company_mostly_funded_state(self):
        """A company with very long runway should be mostly in the FUNDED state."""
        request = _make_audit_request(
            cash_on_hand=500_000_000,
            quarterly_burn=8_000_000,
            stated_months=12,
            enrollment_completed=100,
            enrollment_target=120,
            n_simulations=5000,
        )
        result = run_full_audit(request)
        assert result.valuation.p_funded_through_catalyst > 0.50

    def test_poorly_funded_company_high_discontinuation(self):
        """A company with very short runway should have significant discontinuation probability."""
        request = _make_audit_request(
            cash_on_hand=10_000_000,
            quarterly_burn=25_000_000,
            stated_months=36,
            enrollment_completed=5,
            enrollment_target=200,
            enrollment_rate=2,
            n_simulations=5000,
            positive_signals=[],
            negative_signals=["small_sample_size", "prior_failed_trials"],
        )
        result = run_full_audit(request)
        not_funded = 1.0 - result.valuation.p_funded_through_catalyst
        assert not_funded > 0.50, f"Expected significant non-funded probability, got {not_funded:.1%}"

    def test_financing_penalty_strength_changes_rnpv(self):
        rng_seed = 123
        t_sci = np.full(2000, 24.0)
        t_fin = np.full(2000, 6.0)
        pos = np.full(2000, 0.8)
        base = ValuationInput(
            asset_value_success=500_000_000,
            downside_value=5_000_000,
            annual_discount_rate=0.12,
            expected_dilution_if_refinanced=0.35,
            financing_penalty_strength=0.0,
        )
        penalized = base.model_copy(update={"financing_penalty_strength": 1.0})
        r0 = run_valuation_simulation(
            t_sci, t_fin, pos, base, np.random.default_rng(rng_seed), market_condition_score=5.0
        )
        r1 = run_valuation_simulation(
            t_sci, t_fin, pos, penalized, np.random.default_rng(rng_seed), market_condition_score=5.0
        )
        assert r0.financing_adjusted_rnpv > r1.financing_adjusted_rnpv


class TestDisclosureIntegrity:
    def test_absolute_optimism_gap_flags_uniform_high_vs_low_scores(self):
        high = {category: 0.9 for category in (
            "runway_strength",
            "clinical_timeline_confidence",
            "dilution_risk",
            "trial_maturity",
            "endpoint_strength",
            "pipeline_diversification",
        )}
        low = {category: 0.1 for category in high}
        result = run_disclosure_consistency_analysis(
            DisclosureInput(
                company_narrative_distribution=high,
                structured_audit_distribution=low,
            )
        )
        assert result.jsd_score == pytest.approx(0.0)
        assert result.mean_absolute_gap == pytest.approx(0.8)
        assert result.optimism_bias == pytest.approx(0.8)
        assert result.max_category_gap == pytest.approx(0.8)
        assert result.gap_classification != "aligned"

    def test_disclosure_scores_must_be_between_zero_and_one(self):
        with pytest.raises(ValidationError, match="between 0 and 1"):
            DisclosureInput(
                company_narrative_distribution={"runway_strength": -0.1},
                structured_audit_distribution={"runway_strength": 0.5},
            )
        with pytest.raises(ValidationError, match="between 0 and 1"):
            DisclosureInput(
                company_narrative_distribution={"runway_strength": 0.5},
                structured_audit_distribution={"runway_strength": 1.2},
            )

    def test_disclosure_quality_penalizes_empty_audit_distribution(self):
        request = _make_audit_request(n_simulations=500)
        request = request.model_copy(update={
            "disclosure": DisclosureInput(
                company_narrative_distribution=request.disclosure.company_narrative_distribution,
                structured_audit_distribution={},
            )
        })
        result = run_full_audit(request)
        assert result.data_quality.disclosure_data_completeness < 0.70
        assert any("structured audit" in item.lower() for item in result.data_quality.primary_limitations)


class TestDomainValidation:
    def test_mismatched_clinical_and_pos_phase_is_rejected(self):
        request = _make_audit_request(n_simulations=500)
        with pytest.raises(ValidationError, match="trial_phase"):
            AuditRequest(
                financial=request.financial,
                clinical=request.clinical,
                success_probability=request.success_probability.model_copy(update={"trial_phase": "filed"}),
                valuation=request.valuation,
                disclosure=request.disclosure,
                simulation=request.simulation,
            )

    def test_zero_liquidity_caps_data_quality(self):
        request = _make_audit_request(cash_on_hand=0, quarterly_burn=30_000_000, n_simulations=500)
        fin = request.financial.model_copy(update={"marketable_securities": 0})
        request = request.model_copy(update={"financial": fin})
        result = run_full_audit(request)
        assert result.data_quality.financial_data_completeness <= 0.25
        assert result.data_quality.overall_completeness <= 0.50
        assert result.data_quality.data_quality_score != "high"
