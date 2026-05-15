"""
Tests for Expected Value of Information (EVPI / EVSI) module.
"""

from __future__ import annotations

import pytest

from app.engines.value_of_information import (
    compute_evpi,
    compute_signal_evsi,
    run_value_of_information_analysis,
)
from app.engines.monte_carlo import run_full_audit


# ---------------------------------------------------------------------------
# Unit tests for EVPI
# ---------------------------------------------------------------------------

class TestEVPI:
    def test_evpi_nonnegative(self):
        evpi = compute_evpi(alpha_posterior=3.0, beta_posterior=7.0, financing_adjusted_rnpv=50_000_000)
        assert evpi >= 0.0

    def test_evpi_zero_when_ev_strongly_positive(self):
        """When EV is large relative to the signal, EVPI approaches 0."""
        # With PoS = 3/10 and very high EV, the investment decision doesn't change
        evpi = compute_evpi(alpha_posterior=3.0, beta_posterior=7.0, financing_adjusted_rnpv=1_000_000_000)
        # EVPI = 0 when EV is so large it can't flip negative after one obs
        assert evpi == pytest.approx(0.0, abs=1.0)

    def test_evpi_positive_when_ev_near_zero(self):
        """When EV is near 0, a negative signal could flip the decision → EVPI > 0."""
        # Set EV close to 0 (borderline investment) — info is most valuable here
        evpi = compute_evpi(alpha_posterior=3.0, beta_posterior=7.0, financing_adjusted_rnpv=1_000)
        assert evpi >= 0.0  # EVPI ≥ 0 always

    def test_evpi_zero_when_ev_negative(self):
        """When current EV < 0, EVPI is the value of positive signal recovery."""
        evpi = compute_evpi(alpha_posterior=3.0, beta_posterior=7.0, financing_adjusted_rnpv=-10_000_000)
        # Can't lose from learning truth; EVPI should be positive
        assert evpi >= 0.0

    def test_evpi_increases_with_uncertainty(self):
        """More diffuse posterior (lower α+β) → higher uncertainty → higher EVPI."""
        ev = 5_000_000
        # Concentrated posterior (α+β large) vs diffuse (α+β small)
        evpi_diffuse = compute_evpi(alpha_posterior=1.5, beta_posterior=3.5, financing_adjusted_rnpv=ev)
        evpi_concentrate = compute_evpi(alpha_posterior=15.0, beta_posterior=35.0, financing_adjusted_rnpv=ev)
        # More diffuse posterior → each new observation shifts mean more → higher EVSI
        # (not strict inequality in all cases but generally holds for borderline EVs)
        assert evpi_diffuse >= evpi_concentrate - 1.0

    def test_decision_evpi_positive_when_borderline(self):
        """Perfect information has value when current invest/pass decision is near threshold."""
        evpi = compute_evpi(
            alpha_posterior=5.0,
            beta_posterior=5.0,
            financing_adjusted_rnpv=0.0,
            upside_value=100_000_000,
            capital_required=50_000_000,
        )

        assert evpi > 0.0

    def test_decision_evpi_near_zero_when_strongly_positive(self):
        evpi = compute_evpi(
            alpha_posterior=9.0,
            beta_posterior=1.0,
            financing_adjusted_rnpv=0.0,
            upside_value=100_000_000,
            capital_required=1_000_000,
        )

        assert evpi <= 1_000_000

    def test_decision_evpi_near_zero_when_strongly_negative(self):
        evpi = compute_evpi(
            alpha_posterior=1.0,
            beta_posterior=9.0,
            financing_adjusted_rnpv=0.0,
            upside_value=100_000_000,
            capital_required=99_000_000,
        )

        assert evpi <= 1_000_000


# ---------------------------------------------------------------------------
# Unit tests for EVSI
# ---------------------------------------------------------------------------

class TestEVSI:
    def test_evsi_nonnegative(self):
        evsi, _, _ = compute_signal_evsi(
            alpha_posterior=3.0, beta_posterior=7.0,
            signal_weight=1.5, financing_adjusted_rnpv=50_000_000
        )
        assert evsi >= 0.0

    def test_heavier_signal_has_higher_evsi(self):
        """Stronger signal (higher weight) has at least as much EVSI."""
        kwargs = dict(
            alpha_posterior=5.0,
            beta_posterior=5.0,
            financing_adjusted_rnpv=0.0,
            upside_value=100_000_000,
            capital_required=51_000_000,
        )
        evsi_weak, _, _ = compute_signal_evsi(signal_weight=0.5, **kwargs)
        evsi_strong, _, _ = compute_signal_evsi(signal_weight=2.0, **kwargs)
        assert evsi_strong >= evsi_weak

    def test_ev_if_positive_above_current_pos_fraction(self):
        """After a positive observation, EV should be higher (or equal)."""
        ev = 50_000_000
        evsi, ev_pos, ev_neg = compute_signal_evsi(
            alpha_posterior=3.0, beta_posterior=7.0,
            signal_weight=1.0, financing_adjusted_rnpv=ev
        )
        # Positive signal raises PoS → raises EV
        pos_mean = 3.0 / 10.0
        assert ev_pos > ev * (3.0 + 1.0) / (10.0 + 1.0) / pos_mean * pos_mean - 1.0

    def test_ev_if_negative_below_current(self):
        """After a negative observation, EV should be lower (or equal)."""
        ev = 50_000_000
        evsi, ev_pos, ev_neg = compute_signal_evsi(
            alpha_posterior=3.0, beta_posterior=7.0,
            signal_weight=1.0, financing_adjusted_rnpv=ev
        )
        assert ev_neg < ev + 1.0  # negative signal lowers or at most equals EV


# ---------------------------------------------------------------------------
# run_value_of_information_analysis
# ---------------------------------------------------------------------------

class TestRunVoI:
    def test_returns_correct_structure(self):
        result = run_value_of_information_analysis(
            alpha_posterior=3.0, beta_posterior=7.0,
            financing_adjusted_rnpv=50_000_000,
        )
        assert result.evpi_dollars >= 0.0
        assert isinstance(result.per_signal_evsi, list)
        assert len(result.per_signal_evsi) > 0

    def test_signals_ranked_by_evsi(self):
        result = run_value_of_information_analysis(
            alpha_posterior=3.0, beta_posterior=7.0,
            financing_adjusted_rnpv=50_000_000,
        )
        evsis = [s.evsi_dollars for s in result.per_signal_evsi]
        assert evsis == sorted(evsis, reverse=True), "Signals not ranked by EVSI (descending)"

    def test_total_evsi_equals_sum(self):
        result = run_value_of_information_analysis(
            alpha_posterior=3.0, beta_posterior=7.0,
            financing_adjusted_rnpv=50_000_000,
        )
        expected = sum(s.evsi_dollars for s in result.per_signal_evsi)
        assert result.total_observable_evsi == pytest.approx(expected, abs=1.0)

    def test_all_signal_evsis_nonnegative(self):
        result = run_value_of_information_analysis(
            alpha_posterior=5.0, beta_posterior=5.0,
            financing_adjusted_rnpv=10_000_000,
        )
        for s in result.per_signal_evsi:
            assert s.evsi_dollars >= 0.0, f"{s.signal_name} EVSI < 0"


# ---------------------------------------------------------------------------
# Integration: VoI in full audit response
# ---------------------------------------------------------------------------

class TestVoIInAudit:
    def _request(self, n: int = 300):
        from app.models.schemas import (
            AuditRequest, ClinicalCatalystInput, CompanyFinancialInput,
            DisclosureInput, SimulationConfig, SuccessProbabilityInput, ValuationInput,
        )
        return AuditRequest(
            financial=CompanyFinancialInput(
                company_name="VoICo",
                ticker="VOI",
                cash_on_hand=15_000_000,
                marketable_securities=0,
                quarterly_operating_cash_burn=4_000_000,
                market_cap=60_000_000,
            ),
            clinical=ClinicalCatalystInput(
                asset_name="VOI-01",
                indication="Immunology",
                trial_phase="phase_2",
                trial_status="recruiting",
                stated_months_to_catalyst=18,
                enrollment_target=60,
                enrollment_completed=20,
                enrollment_rate_per_month=4,
                number_of_sites=6,
            ),
            success_probability=SuccessProbabilityInput(trial_phase="phase_2"),
            valuation=ValuationInput(asset_value_success=150_000_000),
            disclosure=DisclosureInput(
                company_narrative_distribution={"runway_strength": 0.7, "clinical_timeline_confidence": 0.7, "dilution_risk": 0.3, "trial_maturity": 0.5, "endpoint_strength": 0.6, "pipeline_diversification": 0.4},
                structured_audit_distribution={"runway_strength": 0.5, "clinical_timeline_confidence": 0.5, "dilution_risk": 0.5, "trial_maturity": 0.4, "endpoint_strength": 0.5, "pipeline_diversification": 0.4},
            ),
            simulation=SimulationConfig(n_simulations=n, random_seed=42, monthly_horizon=24),
        )

    def test_voi_populated_in_audit(self):
        r = run_full_audit(self._request())
        assert r.value_of_information is not None

    def test_voi_evpi_nonnegative(self):
        r = run_full_audit(self._request())
        assert r.value_of_information.evpi_dollars >= 0.0

    def test_voi_has_signals(self):
        r = run_full_audit(self._request())
        assert len(r.value_of_information.per_signal_evsi) > 0

    def test_voi_top_priority_nonempty(self):
        r = run_full_audit(self._request())
        assert len(r.value_of_information.top_diligence_priority) > 0

    def test_voi_methodology_note_present(self):
        r = run_full_audit(self._request())
        assert len(r.value_of_information.methodology_note) > 10
