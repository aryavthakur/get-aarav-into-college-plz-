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
        assert "Distributional Robustness" in r.markdown_report

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
