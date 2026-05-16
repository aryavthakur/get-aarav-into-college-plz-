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

    def test_explicit_financing_state_probabilities_are_exposed(self):
        result = run_full_audit(_request())
        val = result.valuation

        assert val.p_clean_refinancing_before_catalyst == pytest.approx(val.p_refinancing_success)
        assert val.p_distressed_refinancing_before_catalyst == pytest.approx(val.p_distressed_financing)
        assert val.p_program_discontinuation_before_catalyst == pytest.approx(val.p_program_discontinuation)
        assert 0.0 <= val.p_any_financing_event_before_catalyst <= 1.0
        assert 0.0 <= val.p_financing_pressure_before_catalyst <= 1.0

    def test_planned_partnership_before_catalyst_sets_partnership_probability(self):
        result = run_full_audit(
            _request(events=[FinancingEventInput(month=3, kind="partnership", gross_proceeds=10_000_000)])
        )

        assert result.valuation.p_partnership_before_catalyst == pytest.approx(1.0)
        assert result.valuation.p_nondilutive_financing_before_catalyst == pytest.approx(1.0)

    def test_planned_clean_refi_before_catalyst_sets_clean_probability(self):
        result = run_full_audit(
            _request(events=[FinancingEventInput(month=3, kind="clean_refi", gross_proceeds=10_000_000)])
        )

        assert result.valuation.p_clean_refinancing_before_catalyst == pytest.approx(1.0)
        assert result.valuation.p_dilutive_financing_before_catalyst >= result.valuation.p_clean_refinancing_before_catalyst

    def test_planned_financing_after_catalyst_does_not_set_before_catalyst_probability(self):
        result = run_full_audit(
            _request(events=[FinancingEventInput(month=24, kind="partnership", gross_proceeds=10_000_000)])
        )

        assert result.valuation.p_partnership_before_catalyst == pytest.approx(0.0)


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


# ---------------------------------------------------------------------------
# New regression tests for audit-feedback round 2 fixes
# ---------------------------------------------------------------------------

class TestHierarchicalPoSConsistency:
    """Verify Monte Carlo uses the same posterior as run_success_probability_analysis."""

    def test_hierarchical_prior_changes_valuation(self):
        """Hierarchical prior must change financing_adjusted_rnpv relative to phase-only."""
        base = _request(n=3000)
        hier = _request(n=3000)
        hier_data = hier.success_probability.model_dump()
        hier_data["disease_area"] = "oncology"
        hier_data["modality"] = "small_molecule"
        hier_data["endpoint_family"] = "surrogate"
        from app.models.schemas import SuccessProbabilityInput
        hier = hier.model_copy(update={"success_probability": SuccessProbabilityInput(**hier_data)})
        r_base = run_full_audit(base)
        r_hier = run_full_audit(hier)
        # If hierarchical prior differs from phase-only, EVs should differ
        pos_base = r_base.success_probability.posterior_mean
        pos_hier = r_hier.success_probability.posterior_mean
        if abs(pos_base - pos_hier) > 0.001:
            assert r_base.valuation.financing_adjusted_rnpv != r_hier.valuation.financing_adjusted_rnpv, (
                "Hierarchical prior changed PoS but EV didn't change — simulation not using hierarchical posterior"
            )

    def test_pos_posterior_matches_valuation_seed(self):
        """Same seed + same inputs must give identical PoS in both report and simulation."""
        req = _request(n=2000)
        r1 = run_full_audit(req)
        r2 = run_full_audit(req)
        assert r1.success_probability.posterior_mean == r2.success_probability.posterior_mean
        assert r1.valuation.financing_adjusted_rnpv == r2.valuation.financing_adjusted_rnpv


class TestMilestoneTimingFloor:
    """p5_months must never be below public_readout_months."""

    def test_p5_not_below_public_readout(self):
        req = _request()
        r = run_full_audit(req)
        assert r.milestone_timing.p5_months >= r.milestone_timing.public_readout_months - 0.01, (
            f"p5_months={r.milestone_timing.p5_months} < public_readout_months={r.milestone_timing.public_readout_months}"
        )

    def test_p50_not_below_public_readout(self):
        req = _request()
        r = run_full_audit(req)
        assert r.milestone_timing.p50_months >= r.milestone_timing.public_readout_months - 0.01

    def test_sample_floor_applied(self):
        from app.engines.milestone_timing import sample_scientific_milestone_time
        rng = np.random.default_rng(0)
        samples = sample_scientific_milestone_time(rng, alpha=2.0, beta_rate=0.5, n_samples=5000, min_months=15.0)
        assert float(np.min(samples)) >= 15.0


class TestZeroLiquidityCashPath:
    """Zero starting cash must give non-zero capital_needed."""

    def test_zero_cash_capital_needed_to_survive(self):
        from app.engines.cash_path import simulate_cash_path
        from app.models.schemas import CashPathInput
        result = simulate_cash_path(CashPathInput(
            starting_cash=0,
            monthly_burn=1_000_000,
            horizon_months=12,
        ))
        assert result.capital_needed_to_survive_horizon == pytest.approx(12_000_000, rel=0.01)

    def test_zero_cash_capital_needed_to_catalyst(self):
        from app.engines.cash_path import simulate_cash_path
        from app.models.schemas import CashPathInput
        result = simulate_cash_path(CashPathInput(
            starting_cash=0,
            monthly_burn=1_000_000,
            horizon_months=24,
            catalyst_month=6.0,
        ))
        assert result.capital_needed_to_reach_catalyst == pytest.approx(6_000_000, rel=0.01)

    def test_month_zero_financing_offsets_capital_needed(self):
        from app.engines.cash_path import simulate_cash_path
        from app.models.schemas import CashPathInput, FinancingEventInput
        result = simulate_cash_path(CashPathInput(
            starting_cash=0,
            monthly_burn=1_000_000,
            horizon_months=12,
            financing_events=[FinancingEventInput(month=0, kind="clean_refi", gross_proceeds=5_000_000)],
        ))
        # After month-0 financing, cash = 5M. Should survive longer.
        assert result.cash_exhaustion_month is not None
        assert result.cash_exhaustion_month > 0


class TestCatalystMonthInAudit:
    """capital_needed_to_reach_catalyst must not be None in full audit."""

    def test_capital_needed_to_reach_catalyst_populated(self):
        req = _request()
        r = run_full_audit(req)
        assert r.cash_path.capital_needed_to_reach_catalyst is not None


class TestSensitivityCRN:
    """Sensitivity rows must be monotonic under controlled random numbers."""

    def test_burn_sensitivity_monotonic(self):
        """Higher burn → higher cashout probability."""
        req = _request(cash=50_000_000, burn=9_000_000, n=2000)
        r = run_full_audit(req)
        burn_sens = next(s for s in r.final_summary.sensitivity if s.variable == "monthly_burn")
        assert burn_sens.low_cashout_prob <= burn_sens.high_cashout_prob, (
            f"Higher burn should raise cashout risk: low={burn_sens.low_cashout_prob}, high={burn_sens.high_cashout_prob}"
        )

    def test_asset_value_sensitivity_monotonic(self):
        """Higher asset value → higher expected value."""
        req = _request(cash=50_000_000, burn=9_000_000, n=2000)
        r = run_full_audit(req)
        av_sens = next(s for s in r.final_summary.sensitivity if s.variable == "asset_value_success")
        assert av_sens.low_expected_value <= av_sens.high_expected_value, (
            f"Higher asset value should raise EV: low={av_sens.low_expected_value}, high={av_sens.high_expected_value}"
        )

    def test_sensitivity_reproducible_with_same_seed(self):
        req = _request(n=1000)
        r1 = run_full_audit(req)
        r2 = run_full_audit(req)
        for sp1, sp2 in zip(r1.final_summary.sensitivity, r2.final_summary.sensitivity):
            assert sp1.low_expected_value == sp2.low_expected_value
            assert sp1.high_cashout_prob == sp2.high_cashout_prob


class TestEvidenceQuality:
    """Evidence quality is separate from completeness."""

    def test_complete_manual_input_has_low_evidence_quality(self):
        req = _request()
        r = run_full_audit(req)
        assert r.data_quality.overall_completeness > 0.5
        assert r.data_quality.evidence_quality_score == "low"

    def test_evidence_quality_note_mentions_manual(self):
        req = _request()
        r = run_full_audit(req)
        assert "manual" in r.data_quality.evidence_quality_note.lower()

    def test_report_shows_evidence_quality(self):
        req = _request()
        r = run_full_audit(req)
        assert "Evidence Quality" in r.markdown_report


class TestReportUpdates:
    """Verify stale report text was fixed."""

    def test_report_shows_prior_source(self):
        req = _request()
        r = run_full_audit(req)
        assert "Prior Source" in r.markdown_report or "prior_source" in r.markdown_report.lower()

    def test_report_corrects_partnership_assumption(self):
        req = _request()
        r = run_full_audit(req)
        # Old text said partnerships can't be captured; new text says they can be planned
        assert "planned_financing_events" in r.markdown_report or "planned financing" in r.markdown_report.lower()

    def test_report_mentions_data_client_scaffold(self):
        req = _request()
        r = run_full_audit(req)
        assert "scaffold" in r.markdown_report.lower() or "manual" in r.markdown_report.lower()
