"""
Tests for DRO robustness, BMA, and copula dependence engines.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engines.robustness import compute_robustness_bounds
from app.engines.model_averaging import compute_bma, _CANDIDATE_MODELS
from app.engines.dependence import run_dependence_analysis, simulate_with_copula
from app.engines.monte_carlo import run_full_audit
from app.engines.value_of_information import compute_evpi
from app.engines.real_options import RealOptionsInput, simulate_real_options_value


# ---------------------------------------------------------------------------
# Distributional robustness
# ---------------------------------------------------------------------------

class TestDROUnit:
    def _base(self, n=1000, cashout_frac=0.40, seed=0):
        rng = np.random.default_rng(seed)
        t_fin = rng.gamma(shape=3.0, scale=7.0, size=n)
        t_sci = rng.gamma(shape=4.0, scale=5.0, size=n)
        pos = rng.beta(3.0, 7.0, size=n)
        return compute_robustness_bounds(
            t_fin=t_fin, t_sci=t_sci, pos_samples=pos,
            nominal_cashout_prob=cashout_frac,
            nominal_ev=40_000_000,
        )

    def test_worst_case_ge_nominal(self):
        r = self._base()
        assert r.worst_case_cashout_prob_e05 >= r.nominal_cashout_prob
        assert r.worst_case_cashout_prob_e10 >= r.worst_case_cashout_prob_e05
        assert r.worst_case_cashout_prob_e20 >= r.worst_case_cashout_prob_e10

    def test_best_case_le_nominal(self):
        r = self._base()
        assert r.best_case_cashout_prob_e10 <= r.nominal_cashout_prob

    def test_worst_ev_le_nominal(self):
        r = self._base()
        assert r.worst_case_ev_e05 <= r.nominal_ev
        assert r.worst_case_ev_e10 <= r.worst_case_ev_e05

    def test_probabilities_in_unit_interval(self):
        r = self._base()
        for attr in ["worst_case_cashout_prob_e05", "worst_case_cashout_prob_e10",
                     "worst_case_cashout_prob_e20", "best_case_cashout_prob_e10"]:
            v = getattr(r, attr)
            assert 0.0 <= v <= 1.0, f"{attr} = {v} out of [0,1]"

    def test_interpretation_nonempty(self):
        r = self._base()
        assert len(r.robustness_interpretation) > 10


# ---------------------------------------------------------------------------
# Bayesian model averaging
# ---------------------------------------------------------------------------

class TestBMAUnit:
    def test_returns_nine_models(self):
        r = compute_bma(simple_runway=12.0, risk_multiplier=1.2,
                        base_cashout_prob=0.40, base_ev=30_000_000)
        assert len(r.model_weights) == len(_CANDIDATE_MODELS)

    def test_weights_sum_to_one(self):
        r = compute_bma(simple_runway=12.0, risk_multiplier=1.2,
                        base_cashout_prob=0.40, base_ev=30_000_000)
        total = sum(mw.posterior_weight for mw in r.model_weights)
        assert abs(total - 1.0) < 0.01

    def test_all_weights_nonnegative(self):
        r = compute_bma(simple_runway=24.0, risk_multiplier=1.0,
                        base_cashout_prob=0.25, base_ev=50_000_000)
        for mw in r.model_weights:
            assert mw.posterior_weight >= 0.0

    def test_effective_n_between_1_and_9(self):
        r = compute_bma(simple_runway=15.0, risk_multiplier=1.0,
                        base_cashout_prob=0.35, base_ev=40_000_000)
        assert 1.0 <= r.effective_n_models <= 9.0

    def test_bma_cashout_prob_in_unit_interval(self):
        r = compute_bma(simple_runway=10.0, risk_multiplier=2.0,
                        base_cashout_prob=0.60, base_ev=10_000_000)
        assert 0.0 <= r.bma_cashout_prob <= 1.0

    def test_short_runway_concentrates_on_high_lambda_models(self):
        """With very short runway, high-hazard models get more weight."""
        r_short = compute_bma(simple_runway=5.0, risk_multiplier=2.0,
                              base_cashout_prob=0.70, base_ev=5_000_000)
        r_long = compute_bma(simple_runway=30.0, risk_multiplier=0.5,
                             base_cashout_prob=0.15, base_ev=80_000_000)
        top_short = max(r_short.model_weights, key=lambda m: m.posterior_weight)
        top_long = max(r_long.model_weights, key=lambda m: m.posterior_weight)
        # Short runway → higher lambda models preferred
        assert top_short.lambda_ >= top_long.lambda_ - 0.01


# ---------------------------------------------------------------------------
# Copula dependence
# ---------------------------------------------------------------------------

class TestCopulaUnit:
    def _make_samples(self, n=5000, seed=0):
        rng = np.random.default_rng(seed)
        t_fin = np.sort(rng.gamma(shape=3.0, scale=8.0, size=n))
        t_sci = np.sort(rng.gamma(shape=4.0, scale=5.0, size=n))
        return t_fin, t_sci

    def test_positive_rho_raises_cashout_vs_zero(self):
        t_fin, t_sci = self._make_samples()
        rng = np.random.default_rng(0)
        r_pos = simulate_with_copula(t_fin, t_sci, rng, rho=0.30)
        r_zero = simulate_with_copula(t_fin, t_sci, np.random.default_rng(0), rho=0.0)
        assert r_pos.copula_cashout_prob >= r_zero.copula_cashout_prob - 0.02

    def test_negative_rho_lowers_cashout_vs_zero(self):
        t_fin, t_sci = self._make_samples()
        rng = np.random.default_rng(1)
        r_neg = simulate_with_copula(t_fin, t_sci, rng, rho=-0.30)
        r_zero = simulate_with_copula(t_fin, t_sci, np.random.default_rng(1), rho=0.0)
        assert r_neg.copula_cashout_prob <= r_zero.copula_cashout_prob + 0.02

    def test_cashout_prob_in_unit_interval(self):
        t_fin, t_sci = self._make_samples()
        for rho in [-0.30, 0.0, 0.30]:
            rng = np.random.default_rng(0)
            r = simulate_with_copula(t_fin, t_sci, rng, rho=rho)
            assert 0.0 <= r.copula_cashout_prob <= 1.0

    def test_run_dependence_analysis_returns_two_scenarios(self):
        t_fin, t_sci = self._make_samples()
        rng = np.random.default_rng(42)
        r = run_dependence_analysis(t_fin, t_sci, rng)
        assert r.positive_rho.rho == pytest.approx(0.30, abs=0.01)
        assert r.negative_rho == pytest.approx(-0.20, abs=0.01)

    def test_dependence_effect_consistent(self):
        t_fin, t_sci = self._make_samples()
        rng = np.random.default_rng(42)
        r = run_dependence_analysis(t_fin, t_sci, rng)
        assert r.positive_rho.dependence_effect == pytest.approx(
            r.positive_rho.copula_cashout_prob - r.positive_rho.independent_cashout_prob, abs=0.001
        )


# ---------------------------------------------------------------------------
# Integration: all three in full audit
# ---------------------------------------------------------------------------

class TestAdvancedMathIntegration:
    def _request(self, n: int = 400):
        from app.models.schemas import (
            AuditRequest, ClinicalCatalystInput, CompanyFinancialInput,
            DisclosureInput, SimulationConfig, SuccessProbabilityInput, ValuationInput,
        )
        return AuditRequest(
            financial=CompanyFinancialInput(
                company_name="AdvCo", ticker="ADV",
                cash_on_hand=20_000_000, marketable_securities=0,
                quarterly_operating_cash_burn=4_000_000, market_cap=70_000_000,
            ),
            clinical=ClinicalCatalystInput(
                asset_name="ADV-01", indication="Cardiology",
                trial_phase="phase_2", trial_status="recruiting",
                stated_months_to_catalyst=18,
                enrollment_target=90, enrollment_completed=30,
                enrollment_rate_per_month=5, number_of_sites=8,
            ),
            success_probability=SuccessProbabilityInput(trial_phase="phase_2"),
            valuation=ValuationInput(asset_value_success=250_000_000),
            disclosure=DisclosureInput(
                company_narrative_distribution={"runway_strength": 0.6, "clinical_timeline_confidence": 0.6, "dilution_risk": 0.4, "trial_maturity": 0.5, "endpoint_strength": 0.5, "pipeline_diversification": 0.3},
                structured_audit_distribution={"runway_strength": 0.5, "clinical_timeline_confidence": 0.5, "dilution_risk": 0.5, "trial_maturity": 0.4, "endpoint_strength": 0.5, "pipeline_diversification": 0.3},
            ),
            simulation=SimulationConfig(n_simulations=n, random_seed=42, monthly_horizon=24),
        )

    def test_robustness_populated(self):
        r = run_full_audit(self._request())
        assert r.robustness is not None

    def test_bma_populated(self):
        r = run_full_audit(self._request())
        assert r.bma is not None

    def test_dependence_populated(self):
        r = run_full_audit(self._request())
        assert r.dependence is not None

    def test_dro_report_section_present(self):
        r = run_full_audit(self._request())
        assert "Distributional Sensitivity Bounds" in r.markdown_report

    def test_bma_report_section_present(self):
        r = run_full_audit(self._request())
        assert "Bayesian Model Averaging" in r.markdown_report

    def test_dependence_report_section_present(self):
        r = run_full_audit(self._request())
        assert "Copula Dependence" in r.markdown_report

    def test_robustness_worst_case_ge_nominal(self):
        r = run_full_audit(self._request())
        assert r.robustness.worst_case_cashout_prob_e10 >= r.robustness.nominal_cashout_prob

    def test_bma_weights_sum_to_one(self):
        r = run_full_audit(self._request())
        total = sum(mw.posterior_weight for mw in r.bma.model_weights)
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# EVPI explicit decision model
# ---------------------------------------------------------------------------

class TestEVPIExplicitDecision:
    def test_evpi_positive_at_borderline_decision(self):
        """EVPI is positive when PoS is near the break-even point for investment."""
        # upside_value = 100, capital_required = 40, alpha=beta=1 (PoS=0.5)
        # invest_value = 0.5*100 - 40 = 10 > 0, so we invest under current belief.
        # ev_perfect = 0.5 * max(100-40, 0) = 0.5*60 = 30
        # EVPI = 30 - 10 = 20 > 0
        evpi = compute_evpi(
            alpha_posterior=1.0,
            beta_posterior=1.0,
            upside_value=100.0,
            capital_required=40.0,
        )
        assert evpi > 0.0

    def test_evpi_zero_when_investment_clearly_dominant(self):
        """When investment value is far above zero, EVPI approaches zero."""
        # PoS=1.0 equivalent: alpha >> beta → pos_mean ~ 1
        # ev_perfect = 1 * max(V-K, 0) = V-K = invest_value → EVPI ~ 0
        evpi = compute_evpi(
            alpha_posterior=1000.0,
            beta_posterior=1.0,
            upside_value=100.0,
            capital_required=10.0,
        )
        # With nearly certain success, EVPI is near zero
        assert evpi >= 0.0
        assert evpi < 5.0  # should be very small

    def test_evpi_zero_when_pass_dominates(self):
        """When investment is clearly negative (pass dominates), EVPI = 0."""
        # upside_value=10, capital=100 → invest_value = 0.5*10 - 100 = -95 < 0
        # decision_value = 0 (pass)
        # ev_perfect = pos_mean * max(10-100, 0) = 0
        # EVPI = 0 - 0 = 0
        evpi = compute_evpi(
            alpha_posterior=1.0,
            beta_posterior=1.0,
            upside_value=10.0,
            capital_required=100.0,
        )
        assert evpi == 0.0

    def test_evpi_nonnegative(self):
        """EVPI is always non-negative by definition."""
        for alpha, beta, upside, capital in [
            (2.0, 3.0, 50.0, 20.0),
            (5.0, 5.0, 80.0, 45.0),
            (1.0, 9.0, 100.0, 8.0),
        ]:
            evpi = compute_evpi(alpha, beta, upside, capital)
            assert evpi >= 0.0, f"EVPI={evpi} for alpha={alpha}, beta={beta}"


# ---------------------------------------------------------------------------
# Abandonment value test
# ---------------------------------------------------------------------------

class TestAbandonmentValue:
    def test_abandonment_value_positive_with_exercise_cost(self):
        """When K > 0, some GBM paths have V_T < K, creating abandonment value."""
        rng = np.random.default_rng(42)
        n = 5000
        t_sci = np.full(n, 24.0)  # 24-month milestone
        pos_samples = np.full(n, 0.5)
        inputs = RealOptionsInput(
            asset_value_success=100.0,
            exercise_cost=80.0,  # high K so many paths have V_T < K
            asset_volatility=0.60,
            annual_discount_rate=0.12,
        )
        result = simulate_real_options_value(t_sci, pos_samples, inputs, rng)
        assert result.abandonment_value > 0.0, (
            f"abandonment_value should be positive when K>0, got {result.abandonment_value}"
        )

    def test_abandonment_value_zero_when_no_exercise_cost(self):
        """When K=0, there's no exercise cost, so abandonment value is 0."""
        rng = np.random.default_rng(42)
        n = 2000
        t_sci = np.full(n, 18.0)
        pos_samples = np.full(n, 0.4)
        inputs = RealOptionsInput(
            asset_value_success=200.0,
            exercise_cost=0.0,  # K=0: no forced investment
            asset_volatility=0.60,
            annual_discount_rate=0.12,
        )
        result = simulate_real_options_value(t_sci, pos_samples, inputs, rng)
        # With K=0, forced_invest = pos * V_T * discount (always >=0), so max(0, -fi) = 0
        assert result.abandonment_value == pytest.approx(0.0, abs=1e-6)

    def test_rov_nonnegative(self):
        """ROV mean should always be >= 0."""
        rng = np.random.default_rng(0)
        n = 1000
        t_sci = rng.gamma(3.0, 6.0, n)
        pos_samples = rng.beta(2.0, 5.0, n)
        inputs = RealOptionsInput(asset_value_success=50.0, exercise_cost=30.0)
        result = simulate_real_options_value(t_sci, pos_samples, inputs, rng)
        assert result.rov_mean >= 0.0


# ---------------------------------------------------------------------------
# Copula baseline test
# ---------------------------------------------------------------------------

class TestCopulaBaseline:
    def test_independent_cashout_matches_base_cashout_prob(self):
        """When base_cashout_prob is provided, independent_cashout_prob should match it (modulo 4dp rounding)."""
        rng = np.random.default_rng(7)
        n = 3000
        t_fin = np.sort(rng.gamma(3.0, 8.0, n))
        t_sci = np.sort(rng.gamma(4.0, 5.0, n))
        # Compute base_cashout from original unsorted arrays
        t_fin_orig = rng.gamma(3.0, 8.0, n)
        t_sci_orig = rng.gamma(4.0, 5.0, n)
        base = float(np.mean(t_fin_orig < t_sci_orig))

        result = simulate_with_copula(t_fin, t_sci, np.random.default_rng(0), rho=0.0, base_cashout_prob=base)
        # CopulaResult rounds to 4dp, so allow for that
        assert result.independent_cashout_prob == pytest.approx(round(base, 4), abs=1e-9)

    def test_without_base_cashout_uses_sorted_comparison(self):
        """Without base_cashout_prob, independent baseline is from sorted arrays."""
        rng = np.random.default_rng(1)
        n = 2000
        t_fin = np.sort(rng.gamma(3.0, 8.0, n))
        t_sci = np.sort(rng.gamma(4.0, 5.0, n))
        expected_baseline = float(np.mean(t_fin < t_sci))

        result = simulate_with_copula(t_fin, t_sci, np.random.default_rng(0), rho=0.0)
        assert result.independent_cashout_prob == pytest.approx(expected_baseline, abs=1e-9)

    def test_run_dependence_base_cashout_from_paired_arrays(self):
        """run_dependence_analysis base_cashout_prob uses original paired arrays."""
        rng_data = np.random.default_rng(99)
        n = 2000
        t_fin = rng_data.gamma(3.0, 8.0, n)
        t_sci = rng_data.gamma(4.0, 5.0, n)
        expected_base = float(np.mean(t_fin < t_sci))

        dep_rng = np.random.default_rng(42)
        result = run_dependence_analysis(t_fin, t_sci, dep_rng)
        assert result.base_cashout_prob == pytest.approx(expected_base, abs=0.001)


# ---------------------------------------------------------------------------
# Methodology language test
# ---------------------------------------------------------------------------

class TestMethodologyLanguage:
    def test_robustness_note_uses_variance_scaled_language(self):
        """RobustnessResult methodology note should use softened language."""
        from app.engines.robustness import RobustnessResult
        r = RobustnessResult(
            nominal_cashout_prob=0.4, nominal_ev=1e7,
            worst_case_cashout_prob_e05=0.42, worst_case_cashout_prob_e10=0.44,
            worst_case_cashout_prob_e20=0.48, worst_case_ev_e05=9.5e6,
            worst_case_ev_e10=9.0e6, worst_case_ev_e20=8.0e6,
            best_case_cashout_prob_e10=0.36, best_case_ev_e10=1.1e7,
            robustness_interpretation="Moderate sensitivity.",
        )
        assert "Variance-scaled" in r.methodology_note
        assert "Wasserstein-ball DRO" not in r.methodology_note

    def test_bma_note_uses_proxy_language(self):
        """BMAResult methodology note should mention proxy heuristic."""
        from app.engines.model_averaging import BMAResult
        r = BMAResult(
            bma_cashout_prob=0.35, bma_ev=3e7,
            model_weights=[], effective_n_models=5.0,
            highest_weight_model_k=1.3, highest_weight_model_lambda=0.035,
        )
        assert "proxy" in r.methodology_note.lower()
        assert "heuristic" in r.methodology_note.lower()

    def test_risk_attribution_note_uses_sensitivity_language(self):
        """RiskAttributionResult note should use 'sensitivity-based approximation' language."""
        from app.engines.risk_attribution import compute_shapley_attribution
        # We need a mock sensitivity row object
        class MockRow:
            variable = "monthly_burn"
            high_cashout_prob = 0.5
            low_cashout_prob = 0.3
            high_expected_value = 2e7
            low_expected_value = 1e7

        result = compute_shapley_attribution(
            sensitivity_rows=[MockRow()],
            total_cashout_prob=0.4,
            total_ev=1.5e7,
            n_permutations=4,
        )
        assert "sensitivity-based approximation" in result.methodology_note
        assert "true Shapley decomposition" in result.methodology_note


# ---------------------------------------------------------------------------
# Method status field test
# ---------------------------------------------------------------------------

class TestMethodStatus:
    def _make_audit_request(self, n: int = 300):
        from app.models.schemas import (
            AuditRequest, ClinicalCatalystInput, CompanyFinancialInput,
            DisclosureInput, SimulationConfig, SuccessProbabilityInput, ValuationInput,
        )
        return AuditRequest(
            financial=CompanyFinancialInput(
                company_name="StatusCo", ticker="STS",
                cash_on_hand=15_000_000, marketable_securities=0,
                quarterly_operating_cash_burn=3_000_000, market_cap=60_000_000,
            ),
            clinical=ClinicalCatalystInput(
                asset_name="STS-01", indication="Oncology",
                trial_phase="phase_2", trial_status="recruiting",
                stated_months_to_catalyst=18,
                enrollment_target=80, enrollment_completed=20,
                enrollment_rate_per_month=4, number_of_sites=6,
            ),
            success_probability=SuccessProbabilityInput(trial_phase="phase_2"),
            valuation=ValuationInput(asset_value_success=200_000_000),
            disclosure=DisclosureInput(
                company_narrative_distribution={"runway_strength": 0.6, "clinical_timeline_confidence": 0.6, "dilution_risk": 0.4, "trial_maturity": 0.5, "endpoint_strength": 0.5, "pipeline_diversification": 0.3},
                structured_audit_distribution={"runway_strength": 0.5, "clinical_timeline_confidence": 0.5, "dilution_risk": 0.5, "trial_maturity": 0.4, "endpoint_strength": 0.5, "pipeline_diversification": 0.3},
            ),
            simulation=SimulationConfig(n_simulations=n, random_seed=7, monthly_horizon=24),
        )

    def test_value_of_information_has_method_status(self):
        r = run_full_audit(self._make_audit_request())
        assert r.value_of_information is not None
        assert hasattr(r.value_of_information, "method_status")
        assert r.value_of_information.method_status == "heuristic"

    def test_real_options_has_method_status(self):
        r = run_full_audit(self._make_audit_request())
        assert r.real_options is not None
        assert hasattr(r.real_options, "method_status")
        assert r.real_options.method_status == "heuristic"

    def test_robustness_has_method_status(self):
        r = run_full_audit(self._make_audit_request())
        assert r.robustness is not None
        assert hasattr(r.robustness, "method_status")
        assert r.robustness.method_status == "heuristic"

    def test_bma_has_method_status(self):
        r = run_full_audit(self._make_audit_request())
        assert r.bma is not None
        assert hasattr(r.bma, "method_status")
        assert r.bma.method_status == "heuristic"

    def test_dependence_has_method_status(self):
        r = run_full_audit(self._make_audit_request())
        assert r.dependence is not None
        assert hasattr(r.dependence, "method_status")
        assert r.dependence.method_status == "heuristic"

    def test_risk_attribution_has_method_status(self):
        r = run_full_audit(self._make_audit_request())
        assert r.risk_attribution is not None
        assert hasattr(r.risk_attribution, "method_status")
        assert r.risk_attribution.method_status == "heuristic"

    def test_state_space_has_method_status(self):
        r = run_full_audit(self._make_audit_request())
        assert r.state_space is not None
        assert hasattr(r.state_space, "method_status")
        assert r.state_space.method_status == "experimental_scaffold"
