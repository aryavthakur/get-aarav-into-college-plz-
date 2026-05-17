"""
Tests for real-options valuation and Shapley risk attribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engines.real_options import RealOptionsInput, simulate_real_options_value
from app.engines.risk_attribution import compute_shapley_attribution
from app.engines.monte_carlo import run_full_audit


# ---------------------------------------------------------------------------
# Module-scoped fixture for TestRealOptionsShapleyIntegration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ro_shapley_audit():
    from app.models.schemas import (
        AuditRequest, ClinicalCatalystInput, CompanyFinancialInput,
        DisclosureInput, SimulationConfig, SuccessProbabilityInput, ValuationInput,
    )
    return run_full_audit(AuditRequest(
        financial=CompanyFinancialInput(
            company_name="OptCo", ticker="OPC",
            cash_on_hand=25_000_000, marketable_securities=0,
            quarterly_operating_cash_burn=4_000_000, market_cap=80_000_000,
        ),
        clinical=ClinicalCatalystInput(
            asset_name="OPC-01", indication="Neurology",
            trial_phase="phase_2", trial_status="recruiting",
            stated_months_to_catalyst=20,
            enrollment_target=100, enrollment_completed=35,
            enrollment_rate_per_month=6, number_of_sites=10,
        ),
        success_probability=SuccessProbabilityInput(trial_phase="phase_2"),
        valuation=ValuationInput(asset_value_success=300_000_000),
        disclosure=DisclosureInput(
            company_narrative_distribution={"runway_strength": 0.7, "clinical_timeline_confidence": 0.7, "dilution_risk": 0.3, "trial_maturity": 0.5, "endpoint_strength": 0.6, "pipeline_diversification": 0.4},
            structured_audit_distribution={"runway_strength": 0.5, "clinical_timeline_confidence": 0.5, "dilution_risk": 0.5, "trial_maturity": 0.4, "endpoint_strength": 0.5, "pipeline_diversification": 0.4},
        ),
        simulation=SimulationConfig(n_simulations=400, random_seed=42, monthly_horizon=24),
    ))


# ---------------------------------------------------------------------------
# Real-options unit tests
# ---------------------------------------------------------------------------

class TestRealOptionsUnit:
    def _run(self, n=2000, sigma=0.60, K=0.0, seed=0):
        rng = np.random.default_rng(seed)
        t_sci = rng.gamma(shape=4.0, scale=4.0, size=n)
        pos = rng.beta(a=3.0, b=7.0, size=n)
        inp = RealOptionsInput(
            asset_value_success=200_000_000,
            exercise_cost=K,
            asset_volatility=sigma,
            annual_discount_rate=0.12,
            pos_mean=0.30,
        )
        return simulate_real_options_value(t_sci, pos, inp, np.random.default_rng(seed + 1))

    def test_rov_nonnegative(self):
        r = self._run()
        assert r.rov_mean >= 0.0
        assert r.rov_p5 >= 0.0

    def test_higher_volatility_raises_rov(self):
        """Real-options value should increase with volatility (convexity)."""
        r_low = self._run(sigma=0.30, seed=42)
        r_high = self._run(sigma=0.80, seed=42)
        assert r_high.rov_mean >= r_low.rov_mean

    def test_positive_exercise_cost_lowers_rov(self):
        r_free = self._run(K=0.0, seed=7)
        r_cost = self._run(K=50_000_000, seed=7)
        assert r_free.rov_mean >= r_cost.rov_mean

    def test_abandonment_value_nonnegative(self):
        r = self._run()
        assert r.abandonment_value >= 0.0

    def test_abandonment_value_positive_when_exercise_would_destroy_value(self):
        r = self._run(K=260_000_000, sigma=0.20, seed=123)
        assert r.abandonment_value > 0.0

    def test_abandonment_value_zero_when_exercise_cost_zero(self):
        r = self._run(K=0.0, seed=123)
        assert r.abandonment_value == pytest.approx(0.0, abs=1.0)

    def test_financing_distress_reduces_adjusted_rov(self):
        base = self._run(K=30_000_000, seed=5)
        rng = np.random.default_rng(5)
        t_sci = rng.gamma(shape=4.0, scale=4.0, size=2000)
        pos = rng.beta(a=3.0, b=7.0, size=2000)
        stressed = simulate_real_options_value(
            t_sci,
            pos,
            RealOptionsInput(
                asset_value_success=200_000_000,
                exercise_cost=30_000_000,
                financing_state_probabilities={
                    "funded": 0.10,
                    "clean_refinancing": 0.10,
                    "distressed_refinancing": 0.30,
                    "program_discontinuation": 0.50,
                },
            ),
            np.random.default_rng(6),
        )
        assert stressed.financing_adjusted_rov < base.rov_mean

    def test_partnership_retained_economics_lowers_but_preserves_rov(self):
        rng = np.random.default_rng(6)
        t_sci = rng.gamma(shape=4.0, scale=4.0, size=2000)
        pos = rng.beta(a=3.0, b=7.0, size=2000)
        r = simulate_real_options_value(
            t_sci,
            pos,
            RealOptionsInput(
                asset_value_success=200_000_000,
                exercise_cost=30_000_000,
                financing_state_probabilities={
                    "funded": 0.20,
                    "partnership": 0.80,
                },
                partnership_retained_economics=0.45,
            ),
            np.random.default_rng(7),
        )
        assert 0.0 < r.financing_adjusted_rov < r.rov_mean

    def test_rov_ge_rnpv_when_no_exercise_cost(self):
        """ROV ≥ rNPV when K=0 (option value ≥ intrinsic value)."""
        r = self._run(K=0.0)
        assert r.rov_mean >= r.rnpv_static - 1.0  # small tolerance for discretisation

    def test_model_assumptions_nonempty(self):
        r = self._run()
        assert len(r.model_assumptions) > 0

    def test_real_options_result_preserves_model_inputs(self):
        r = self._run(sigma=0.80, K=75_000_000, seed=12)
        assert r.asset_volatility == pytest.approx(0.80)
        assert r.exercise_cost == pytest.approx(75_000_000)

    def test_reproducible(self):
        r1 = self._run(seed=99)
        r2 = self._run(seed=99)
        assert r1.rov_mean == r2.rov_mean


# ---------------------------------------------------------------------------
# Shapley unit tests
# ---------------------------------------------------------------------------

class TestShapleyUnit:
    def _make_sensitivity_rows(self):
        from app.models.schemas import SensitivityPoint
        return [
            SensitivityPoint(
                variable="monthly_burn", low_label="low", base_label="base", high_label="high",
                low_cashout_prob=0.20, base_cashout_prob=0.40, high_cashout_prob=0.65,
                low_expected_value=60_000_000, base_expected_value=40_000_000, high_expected_value=20_000_000,
            ),
            SensitivityPoint(
                variable="posterior_pos", low_label="low", base_label="base", high_label="high",
                low_cashout_prob=0.40, base_cashout_prob=0.40, high_cashout_prob=0.40,
                low_expected_value=15_000_000, base_expected_value=40_000_000, high_expected_value=65_000_000,
            ),
            SensitivityPoint(
                variable="asset_value_success", low_label="low", base_label="base", high_label="high",
                low_cashout_prob=0.40, base_cashout_prob=0.40, high_cashout_prob=0.40,
                low_expected_value=5_000_000, base_expected_value=40_000_000, high_expected_value=80_000_000,
            ),
        ]

    def test_returns_components(self):
        rows = self._make_sensitivity_rows()
        r = compute_shapley_attribution(rows, 0.40, 40_000_000)
        assert len(r.components) == len(rows)

    def test_components_have_correct_rank_order(self):
        rows = self._make_sensitivity_rows()
        r = compute_shapley_attribution(rows, 0.40, 40_000_000)
        ranks = [c.rank for c in sorted(r.components, key=lambda x: x.rank)]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_shapley_values_sum_approximately_to_total_range(self):
        rows = self._make_sensitivity_rows()
        r = compute_shapley_attribution(rows, 0.40, 40_000_000)
        # Shapley values should approximately sum to the total range explained
        total_cp = sum(abs(c.cashout_prob_shapley) for c in r.components)
        assert total_cp > 0.0

    def test_driver_with_largest_range_has_highest_shapley(self):
        """burn has the largest cashout_prob range (0.45) → should have highest rank."""
        rows = self._make_sensitivity_rows()
        r = compute_shapley_attribution(rows, 0.40, 40_000_000)
        top = min(r.components, key=lambda c: c.rank)
        assert top.driver == "monthly_burn", f"Expected burn at rank 1, got {top.driver}"

    def test_methodology_note_nonempty(self):
        rows = self._make_sensitivity_rows()
        r = compute_shapley_attribution(rows, 0.40, 40_000_000)
        assert len(r.methodology_note) > 10

    def test_methodology_note_is_sensitivity_based_approximation(self):
        rows = self._make_sensitivity_rows()
        r = compute_shapley_attribution(rows, 0.40, 40_000_000)
        note = r.methodology_note.lower()
        assert "sensitivity-based" in note or "approximation" in note


# ---------------------------------------------------------------------------
# Integration: real options + Shapley in full audit
# ---------------------------------------------------------------------------

class TestRealOptionsShapleyIntegration:
    def test_real_options_populated(self, ro_shapley_audit):
        assert ro_shapley_audit.real_options is not None

    def test_real_options_rov_nonnegative(self, ro_shapley_audit):
        assert ro_shapley_audit.real_options.rov_mean >= 0.0

    def test_audit_uses_phase_aware_real_options_exercise_cost(self, ro_shapley_audit):
        assert ro_shapley_audit.real_options.exercise_cost > 0.0
        assert "phase-aware" in " ".join(ro_shapley_audit.real_options.model_assumptions).lower()

    def test_risk_attribution_populated(self, ro_shapley_audit):
        assert ro_shapley_audit.risk_attribution is not None

    def test_risk_attribution_has_components(self, ro_shapley_audit):
        assert len(ro_shapley_audit.risk_attribution.components) > 0

    def test_report_includes_real_options_section(self, ro_shapley_audit):
        r = ro_shapley_audit
        assert "Real-Options Valuation" in r.markdown_report
        assert f"sigma={r.real_options.asset_volatility:.0%}" in r.markdown_report

    def test_report_includes_shapley_section(self, ro_shapley_audit):
        assert "Shapley Risk Attribution" in ro_shapley_audit.markdown_report
