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
        evpi = compute_evpi(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=100_000_000, capital_required=30_000_000,
        )
        assert evpi >= 0.0

    def test_evpi_zero_when_no_capital_required(self):
        """When capital_required=0 investor always invests regardless — EVPI=0."""
        evpi = compute_evpi(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=100_000_000, capital_required=0.0,
        )
        assert evpi == pytest.approx(0.0, abs=1.0)

    def test_evpi_positive_when_decision_borderline(self):
        """When invest_value≈0 (borderline), perfect info is most valuable."""
        pos_mean = 3.0 / 10.0
        upside = 100_000_000
        # capital = pos_mean * upside → invest_value = 0 → decision is borderline
        capital = pos_mean * upside
        evpi = compute_evpi(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=upside, capital_required=capital,
        )
        assert evpi > 0.0

    def test_evpi_zero_when_upside_less_than_capital(self):
        """When success is never profitable (upside < capital), EVPI=0."""
        evpi = compute_evpi(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=10_000_000, capital_required=20_000_000,
        )
        assert evpi == pytest.approx(0.0, abs=1.0)

    def test_evpi_equals_perfect_info_minus_decision(self):
        """Verify EVPI = E[value under perfect info] - max(invest_value, 0)."""
        alpha, beta = 4.0, 6.0
        upside, capital = 80_000_000, 25_000_000
        pos_mean = alpha / (alpha + beta)
        invest_value = pos_mean * upside - capital
        decision_value = max(invest_value, 0.0)
        ev_perfect = pos_mean * max(upside - capital, 0.0)
        expected_evpi = max(0.0, ev_perfect - decision_value)
        evpi = compute_evpi(alpha, beta, upside, capital)
        assert evpi == pytest.approx(expected_evpi, abs=1.0)


# ---------------------------------------------------------------------------
# Unit tests for EVSI
# ---------------------------------------------------------------------------

class TestEVSI:
    def test_evsi_nonnegative(self):
        evsi, _, _ = compute_signal_evsi(
            alpha_posterior=3.0, beta_posterior=7.0,
            signal_weight=1.5, upside_value=100_000_000, capital_required=30_000_000,
        )
        assert evsi >= 0.0

    def test_heavier_signal_has_higher_evsi(self):
        """Stronger signal (higher weight) has at least as much EVSI."""
        kwargs = dict(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=100_000_000, capital_required=30_000_000,
        )
        evsi_weak, _, _ = compute_signal_evsi(signal_weight=0.5, **kwargs)
        evsi_strong, _, _ = compute_signal_evsi(signal_weight=2.0, **kwargs)
        assert evsi_strong >= evsi_weak

    def test_positive_signal_raises_invest_value(self):
        """After a positive signal, the investment value for the positive case increases."""
        _, iv_pos, iv_neg = compute_signal_evsi(
            alpha_posterior=3.0, beta_posterior=7.0,
            signal_weight=1.0, upside_value=100_000_000, capital_required=30_000_000,
        )
        # Positive signal raises PoS → raises investment value
        assert iv_pos > iv_neg

    def test_negative_signal_lowers_invest_value(self):
        """After a negative signal, investment value is lower than after positive."""
        _, iv_pos, iv_neg = compute_signal_evsi(
            alpha_posterior=3.0, beta_posterior=7.0,
            signal_weight=1.0, upside_value=100_000_000, capital_required=30_000_000,
        )
        assert iv_neg < iv_pos


# ---------------------------------------------------------------------------
# run_value_of_information_analysis
# ---------------------------------------------------------------------------

class TestRunVoI:
    _UPSIDE = 100_000_000
    _CAPITAL = 20_000_000

    def test_returns_correct_structure(self):
        result = run_value_of_information_analysis(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=self._UPSIDE, capital_required=self._CAPITAL,
        )
        assert result.evpi_dollars >= 0.0
        assert isinstance(result.per_signal_evsi, list)
        assert len(result.per_signal_evsi) > 0

    def test_signals_ranked_by_evsi(self):
        result = run_value_of_information_analysis(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=self._UPSIDE, capital_required=self._CAPITAL,
        )
        evsis = [s.evsi_dollars for s in result.per_signal_evsi]
        assert evsis == sorted(evsis, reverse=True), "Signals not ranked by EVSI (descending)"

    def test_total_evsi_equals_sum(self):
        result = run_value_of_information_analysis(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=self._UPSIDE, capital_required=self._CAPITAL,
        )
        expected = sum(s.evsi_dollars for s in result.per_signal_evsi)
        assert result.total_observable_evsi == pytest.approx(expected, abs=1.0)

    def test_all_signal_evsis_nonnegative(self):
        result = run_value_of_information_analysis(
            alpha_posterior=5.0, beta_posterior=5.0,
            upside_value=self._UPSIDE, capital_required=self._CAPITAL,
        )
        for s in result.per_signal_evsi:
            assert s.evsi_dollars >= 0.0, f"{s.signal_name} EVSI < 0"

    def test_evpi_positive_at_borderline(self):
        """EVPI is positive when the decision is exactly at the threshold."""
        pos_mean = 5.0 / 10.0
        upside = 100_000_000
        capital = pos_mean * upside  # invest_value = 0
        result = run_value_of_information_analysis(
            alpha_posterior=5.0, beta_posterior=5.0,
            upside_value=upside, capital_required=capital,
        )
        assert result.evpi_dollars > 0.0

    def test_higher_signal_weight_increases_evsi_at_borderline(self):
        """At the decision threshold, higher signal weight → higher EVSI."""
        pos_mean = 3.0 / 10.0
        upside = 100_000_000
        capital = pos_mean * upside
        r_low = run_value_of_information_analysis(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=upside, capital_required=capital,
            config_signal_weights={"dose_response_observed": 0.5},
        )
        r_high = run_value_of_information_analysis(
            alpha_posterior=3.0, beta_posterior=7.0,
            upside_value=upside, capital_required=capital,
            config_signal_weights={"dose_response_observed": 3.0},
        )
        dose_low = next(s for s in r_low.per_signal_evsi if s.signal_name == "dose_response_observed")
        dose_high = next(s for s in r_high.per_signal_evsi if s.signal_name == "dose_response_observed")
        assert dose_high.evsi_dollars >= dose_low.evsi_dollars


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
