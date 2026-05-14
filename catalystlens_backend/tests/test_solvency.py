"""
Tests for the solvency (financial clock) engine.
"""

import math
import numpy as np
import pytest

from app.core.config import CatalystLensConfig, WeibullParams, get_default_config
from app.engines.solvency import (
    _compute_linear_predictor,
    baseline_survival,
    calculate_monthly_burn,
    calculate_risk_multiplier,
    calculate_simple_runway_months,
    compute_total_liquidity,
    run_solvency_analysis,
    sample_financial_failure_time,
    survival_probability,
)
from app.models.schemas import CompanyFinancialInput


def _make_financial(**kwargs) -> CompanyFinancialInput:
    defaults = dict(
        company_name="TestCo",
        ticker="TST",
        cash_on_hand=60_000_000,
        marketable_securities=10_000_000,
        quarterly_operating_cash_burn=15_000_000,
        market_cap=200_000_000,
        debt=5_000_000,
        going_concern_flag=False,
        recent_financing_flag=False,
        months_since_last_raise=12.0,
        biotech_market_condition_score=5.0,
        pipeline_concentration_score=0.5,
    )
    defaults.update(kwargs)
    return CompanyFinancialInput(**defaults)


class TestBasicCalculations:
    def test_monthly_burn_from_quarterly(self):
        assert calculate_monthly_burn(12_000_000) == pytest.approx(4_000_000)

    def test_monthly_burn_from_quarterly_28m(self):
        assert calculate_monthly_burn(28_000_000) == pytest.approx(9_333_333.33, rel=1e-4)

    def test_simple_runway_exact(self):
        """120M / 10M_per_month = 12 months."""
        assert calculate_simple_runway_months(120_000_000, 10_000_000) == pytest.approx(12.0)

    def test_simple_runway_zero_burn_returns_inf(self):
        assert calculate_simple_runway_months(100_000_000, 0.0) == float("inf")

    def test_total_liquidity(self):
        fin = _make_financial(cash_on_hand=80_000_000, marketable_securities=20_000_000)
        assert compute_total_liquidity(fin) == pytest.approx(100_000_000)

    def test_total_liquidity_zero_securities(self):
        fin = _make_financial(cash_on_hand=50_000_000, marketable_securities=0.0)
        assert compute_total_liquidity(fin) == pytest.approx(50_000_000)


class TestRiskMultiplier:
    def _get_lp(self, **fin_kwargs) -> float:
        config = get_default_config()
        fin = _make_financial(**fin_kwargs)
        monthly_burn = calculate_monthly_burn(fin.quarterly_operating_cash_burn)
        total_liq = compute_total_liquidity(fin)
        lp, _ = _compute_linear_predictor(
            monthly_burn=monthly_burn,
            total_liquidity=total_liq,
            burn_acceleration=0.0,
            market_cap=fin.market_cap,
            debt=fin.debt,
            going_concern_flag=fin.going_concern_flag,
            recent_financing_flag=fin.recent_financing_flag,
            months_since_last_raise=fin.months_since_last_raise,
            biotech_market_condition_score=fin.biotech_market_condition_score,
            pipeline_concentration_score=fin.pipeline_concentration_score,
            trial_phase="phase_2",
            coeff=config.cox_coefficients,
            phase_risk_map=config.trial_phase_risk_map,
        )
        return lp

    def test_going_concern_increases_risk(self):
        lp_no_gc = self._get_lp(going_concern_flag=False)
        lp_gc = self._get_lp(going_concern_flag=True)
        assert lp_gc > lp_no_gc

    def test_short_runway_higher_risk_than_long_runway(self):
        """Company with 5 months runway should have higher LP than one with 30 months."""
        # Short runway company: small cash, high burn
        lp_short = self._get_lp(
            cash_on_hand=15_000_000, marketable_securities=0, quarterly_operating_cash_burn=15_000_000
        )
        # Long runway company: large cash, low burn
        lp_long = self._get_lp(
            cash_on_hand=120_000_000, marketable_securities=20_000_000, quarterly_operating_cash_burn=10_000_000
        )
        assert lp_short > lp_long

    def test_risk_multiplier_exp_of_lp(self):
        lp = 0.5
        assert calculate_risk_multiplier(lp) == pytest.approx(math.exp(0.5), rel=1e-6)

    def test_risk_multiplier_clamped_upper(self):
        assert calculate_risk_multiplier(100.0) == pytest.approx(20.0)

    def test_risk_multiplier_clamped_lower(self):
        assert calculate_risk_multiplier(-100.0) == pytest.approx(0.05)

    def test_burn_acceleration_increases_risk(self):
        lp_no_accel = self._get_lp()
        config = get_default_config()
        fin = _make_financial()
        monthly_burn = calculate_monthly_burn(fin.quarterly_operating_cash_burn)
        total_liq = compute_total_liquidity(fin)
        lp_accel, _ = _compute_linear_predictor(
            monthly_burn=monthly_burn,
            total_liquidity=total_liq,
            burn_acceleration=0.50,
            market_cap=fin.market_cap,
            debt=fin.debt,
            going_concern_flag=False,
            recent_financing_flag=False,
            months_since_last_raise=12.0,
            biotech_market_condition_score=5.0,
            pipeline_concentration_score=0.5,
            trial_phase="phase_2",
            coeff=config.cox_coefficients,
            phase_risk_map=config.trial_phase_risk_map,
        )
        assert lp_accel > lp_no_accel

    def test_recent_financing_reduces_risk(self):
        lp_no_rf = self._get_lp(recent_financing_flag=False)
        lp_rf = self._get_lp(recent_financing_flag=True)
        assert lp_rf < lp_no_rf


class TestSurvivalFunctions:
    def test_baseline_survival_at_zero_is_one(self):
        wp = WeibullParams()
        assert baseline_survival(0.0, wp) == pytest.approx(1.0)

    def test_baseline_survival_decreases_over_time(self):
        wp = WeibullParams()
        s_12 = baseline_survival(12.0, wp)
        s_24 = baseline_survival(24.0, wp)
        s_36 = baseline_survival(36.0, wp)
        assert s_12 > s_24 > s_36

    def test_baseline_survival_between_0_and_1(self):
        wp = WeibullParams()
        for t in [1, 6, 12, 24, 36, 48]:
            s = baseline_survival(float(t), wp)
            assert 0.0 <= s <= 1.0, f"Survival at t={t} out of range: {s}"

    def test_survival_probability_with_high_risk_lower(self):
        """High risk multiplier should produce lower survival probability than low risk."""
        wp = WeibullParams()
        sp_low_risk = survival_probability(24.0, risk_multiplier=0.5, params=wp)
        sp_high_risk = survival_probability(24.0, risk_multiplier=3.0, params=wp)
        assert sp_high_risk < sp_low_risk

    def test_survival_probability_rm1_equals_baseline(self):
        """Risk multiplier = 1.0 should reproduce baseline survival exactly."""
        wp = WeibullParams()
        t = 18.0
        assert survival_probability(t, 1.0, wp) == pytest.approx(baseline_survival(t, wp))

    def test_survival_probability_at_zero_is_one(self):
        wp = WeibullParams()
        assert survival_probability(0.0, 2.0, wp) == pytest.approx(1.0)


class TestSampling:
    def test_sampled_failure_times_are_positive(self):
        rng = np.random.default_rng(42)
        wp = WeibullParams()
        samples = sample_financial_failure_time(rng, risk_multiplier=1.5, params=wp, n_samples=1000)
        assert np.all(samples > 0)

    def test_higher_risk_produces_shorter_median_failure_time(self):
        rng = np.random.default_rng(42)
        wp = WeibullParams()
        low_risk_samples = sample_financial_failure_time(rng, 0.5, wp, 5000)
        high_risk_samples = sample_financial_failure_time(rng, 4.0, wp, 5000)
        assert np.median(high_risk_samples) < np.median(low_risk_samples)


class TestRunSolvencyAnalysis:
    def test_result_contains_expected_fields(self):
        fin = _make_financial()
        result = run_solvency_analysis(fin)
        assert result.monthly_burn > 0
        assert result.simple_runway_months > 0
        assert result.risk_multiplier > 0
        assert len(result.survival_curve) > 0
        assert 0 <= result.p_survival_12m <= 1.0

    def test_high_burn_acceleration_increases_risk_multiplier(self):
        fin = _make_financial()
        result_no_accel = run_solvency_analysis(fin, burn_acceleration=0.0)
        result_accel = run_solvency_analysis(fin, burn_acceleration=0.5)
        assert result_accel.risk_multiplier > result_no_accel.risk_multiplier
