"""
Tests for multi-state competing-risk survival engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engines.multistate import (
    CAUSE_NAMES,
    CAUSE_TO_VALUATION_STATE,
    DEFAULT_CAUSE_SCALES,
    build_cause_lp,
    cif_at_time,
    compute_cif_curves,
    compute_overall_survival,
    sample_competing_risk,
)
from app.engines.cumulative_incidence import named_cif_at_time, survival_at_catalyst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_cause_scales(dominant_cause: int, dominant_scale: float = 5.0) -> dict:
    """All causes with very long scales except one dominant cause."""
    scales = {cid: (k, 10_000.0) for cid, (k, _) in DEFAULT_CAUSE_SCALES.items()}
    dominant_k = DEFAULT_CAUSE_SCALES[dominant_cause][0]
    scales[dominant_cause] = (dominant_k, dominant_scale)
    return scales


def _uniform_lp(cause_ids: list[int], value: float = 1.0) -> dict:
    return {cid: value for cid in cause_ids}


# ---------------------------------------------------------------------------
# CIF mathematical invariants
# ---------------------------------------------------------------------------

class TestCIFInvariants:
    def test_cif_starts_at_zero(self):
        grid = np.linspace(0.0, 48.0, 100)
        cause_lp = build_cause_lp(0.0)
        curves = compute_cif_curves(grid, DEFAULT_CAUSE_SCALES, cause_lp)
        for cid, arr in curves.items():
            assert arr[0] == pytest.approx(0.0, abs=1e-10), f"CIF[{cid}] not 0 at t=0"

    def test_cif_nonnegative_everywhere(self):
        grid = np.linspace(0.0, 48.0, 100)
        cause_lp = build_cause_lp(0.0)
        curves = compute_cif_curves(grid, DEFAULT_CAUSE_SCALES, cause_lp)
        for cid, arr in curves.items():
            assert np.all(arr >= -1e-10), f"CIF[{cid}] has negative values"

    def test_cif_sum_equals_one_minus_survival(self):
        grid = np.linspace(0.0, 48.0, 50)
        cause_lp = build_cause_lp(0.0)
        curves = compute_cif_curves(grid, DEFAULT_CAUSE_SCALES, cause_lp)
        s_values = compute_overall_survival(grid, DEFAULT_CAUSE_SCALES, cause_lp)
        cif_sum = sum(arr for arr in curves.values())
        expected = 1.0 - s_values
        np.testing.assert_allclose(cif_sum, expected, atol=0.01,
                                   err_msg="sum CIF_j != 1 - S(t) at all grid points")

    def test_cif_monotone_nondecreasing(self):
        grid = np.linspace(0.01, 60.0, 200)
        cause_lp = build_cause_lp(0.5)
        curves = compute_cif_curves(grid, DEFAULT_CAUSE_SCALES, cause_lp)
        for cid, arr in curves.items():
            diffs = np.diff(arr)
            assert np.all(diffs >= -1e-10), f"CIF[{cid}] is not non-decreasing"

    def test_overall_survival_monotone_decreasing(self):
        grid = np.linspace(0.0, 60.0, 100)
        cause_lp = build_cause_lp(0.0)
        s = compute_overall_survival(grid, DEFAULT_CAUSE_SCALES, cause_lp)
        assert s[0] == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.diff(s) <= 1e-10)

    def test_survival_at_zero_is_one(self):
        t = np.array([0.0])
        cause_lp = build_cause_lp(0.0)
        s = compute_overall_survival(t, DEFAULT_CAUSE_SCALES, cause_lp)
        assert s[0] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Sampling correctness
# ---------------------------------------------------------------------------

class TestSampleDistribution:
    def test_sample_times_positive(self):
        rng = np.random.default_rng(0)
        cause_lp = build_cause_lp(0.0)
        samples = sample_competing_risk(DEFAULT_CAUSE_SCALES, cause_lp, rng, 1000)
        assert np.all(samples[:, 0] > 0)

    def test_cause_ids_are_valid(self):
        rng = np.random.default_rng(1)
        cause_lp = build_cause_lp(0.0)
        samples = sample_competing_risk(DEFAULT_CAUSE_SCALES, cause_lp, rng, 1000)
        valid = set(CAUSE_NAMES.keys())
        observed = set(int(c) for c in samples[:, 1])
        assert observed.issubset(valid), f"Invalid cause IDs: {observed - valid}"

    def test_dominant_cause_wins(self):
        """When one cause has a very short scale, it should dominate."""
        rng = np.random.default_rng(42)
        scales = _make_single_cause_scales(dominant_cause=1, dominant_scale=2.0)
        cause_lp = _uniform_lp(list(scales.keys()))
        samples = sample_competing_risk(scales, cause_lp, rng, 10_000)
        frac_cause1 = np.mean(samples[:, 1] == 1)
        assert frac_cause1 > 0.90, f"Dominant cause 1 fraction: {frac_cause1:.3f}"

    def test_reproducible_with_same_seed(self):
        cause_lp = build_cause_lp(0.0)
        s1 = sample_competing_risk(DEFAULT_CAUSE_SCALES, cause_lp, np.random.default_rng(7), 100)
        s2 = sample_competing_risk(DEFAULT_CAUSE_SCALES, cause_lp, np.random.default_rng(7), 100)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        cause_lp = build_cause_lp(0.0)
        s1 = sample_competing_risk(DEFAULT_CAUSE_SCALES, cause_lp, np.random.default_rng(0), 1000)
        s2 = sample_competing_risk(DEFAULT_CAUSE_SCALES, cause_lp, np.random.default_rng(999), 1000)
        assert not np.array_equal(s1, s2)

    def test_shape_is_correct(self):
        rng = np.random.default_rng(0)
        cause_lp = build_cause_lp(0.0)
        n = 500
        samples = sample_competing_risk(DEFAULT_CAUSE_SCALES, cause_lp, rng, n)
        assert samples.shape == (n, 2)


# ---------------------------------------------------------------------------
# Cash exhaustion sensitivity
# ---------------------------------------------------------------------------

class TestCashExhaustionSensitivity:
    def test_high_lp_raises_cash_exhaustion_probability(self):
        """Higher aggregate LP → higher cause-7 (cash_exhaustion) fraction."""
        rng_low = np.random.default_rng(0)
        rng_high = np.random.default_rng(0)

        lp_low = build_cause_lp(0.0)
        lp_high = build_cause_lp(2.0)

        n = 50_000
        s_low = sample_competing_risk(DEFAULT_CAUSE_SCALES, lp_low, rng_low, n)
        s_high = sample_competing_risk(DEFAULT_CAUSE_SCALES, lp_high, rng_high, n)

        p_cashout_low = np.mean(s_low[:, 1] == 7)
        p_cashout_high = np.mean(s_high[:, 1] == 7)

        assert p_cashout_high > p_cashout_low, (
            f"High LP should raise cash_exhaustion prob: low={p_cashout_low:.3f}, high={p_cashout_high:.3f}"
        )
        assert p_cashout_high - p_cashout_low > 0.02

    def test_high_lp_lowers_funded_probability(self):
        """Higher distress should reduce probability of clean equity round."""
        rng_low = np.random.default_rng(1)
        rng_high = np.random.default_rng(1)

        lp_low = build_cause_lp(0.0)
        lp_high = build_cause_lp(2.0)

        n = 50_000
        s_low = sample_competing_risk(DEFAULT_CAUSE_SCALES, lp_low, rng_low, n)
        s_high = sample_competing_risk(DEFAULT_CAUSE_SCALES, lp_high, rng_high, n)

        p_funded_low = np.mean(s_low[:, 1] == 1)
        p_funded_high = np.mean(s_high[:, 1] == 1)

        assert p_funded_low >= p_funded_high, (
            f"Higher LP should lower funding probability: low={p_funded_low:.3f}, high={p_funded_high:.3f}"
        )


# ---------------------------------------------------------------------------
# Cause-to-valuation mapping
# ---------------------------------------------------------------------------

class TestCauseToValuationMapping:
    def test_funded_maps_to_state_zero(self):
        assert CAUSE_TO_VALUATION_STATE[1] == 0

    def test_cash_exhaustion_maps_to_discontinuation_state(self):
        assert CAUSE_TO_VALUATION_STATE[7] == 3

    def test_program_discontinuation_maps_to_downside(self):
        assert CAUSE_TO_VALUATION_STATE[6] == 3

    def test_partnership_treated_as_clean_refi(self):
        assert CAUSE_TO_VALUATION_STATE[4] == 1

    def test_distressed_maps_to_state_two(self):
        assert CAUSE_TO_VALUATION_STATE[3] == 2

    def test_all_causes_have_mapping(self):
        for cid in CAUSE_NAMES:
            assert cid in CAUSE_TO_VALUATION_STATE, f"Cause {cid} missing from valuation state map"


# ---------------------------------------------------------------------------
# CIF public API (cumulative_incidence.py)
# ---------------------------------------------------------------------------

class TestCumulativeIncidenceAPI:
    def test_named_cif_at_time_keys_match_cause_names(self):
        result = named_cif_at_time(24.0, aggregate_lp=0.0)
        assert set(result.keys()) == set(CAUSE_NAMES.values())

    def test_named_cif_values_are_probabilities(self):
        result = named_cif_at_time(24.0, aggregate_lp=0.0)
        for name, v in result.items():
            assert 0.0 <= v <= 1.0, f"CIF[{name}] = {v} out of [0,1]"

    def test_survival_at_catalyst_between_zero_and_one(self):
        s = survival_at_catalyst(18.0, aggregate_lp=0.0)
        assert 0.0 < s <= 1.0

    def test_survival_decreases_with_time(self):
        s12 = survival_at_catalyst(12.0, aggregate_lp=0.0)
        s24 = survival_at_catalyst(24.0, aggregate_lp=0.0)
        s36 = survival_at_catalyst(36.0, aggregate_lp=0.0)
        assert s12 >= s24 >= s36


# ---------------------------------------------------------------------------
# Monte Carlo integration
# ---------------------------------------------------------------------------

class TestMultiStateIntegration:
    def _make_request(self, use_multistate: bool = False, n: int = 500):
        from app.models.schemas import (
            AuditRequest, ClinicalCatalystInput, CompanyFinancialInput,
            DisclosureInput, SimulationConfig, SuccessProbabilityInput, ValuationInput,
        )
        return AuditRequest(
            financial=CompanyFinancialInput(
                company_name="MultiCo",
                ticker="MLC",
                cash_on_hand=20_000_000,
                marketable_securities=0,
                quarterly_operating_cash_burn=5_000_000,
                market_cap=80_000_000,
            ),
            clinical=ClinicalCatalystInput(
                asset_name="MLC-01",
                indication="Oncology",
                trial_phase="phase_2",
                trial_status="recruiting",
                stated_months_to_catalyst=18,
                enrollment_target=80,
                enrollment_completed=30,
                enrollment_rate_per_month=5,
                number_of_sites=8,
            ),
            success_probability=SuccessProbabilityInput(trial_phase="phase_2"),
            valuation=ValuationInput(asset_value_success=200_000_000),
            disclosure=DisclosureInput(
                company_narrative_distribution={"runway_strength": 0.6, "clinical_timeline_confidence": 0.6, "dilution_risk": 0.4, "trial_maturity": 0.5, "endpoint_strength": 0.5, "pipeline_diversification": 0.3},
                structured_audit_distribution={"runway_strength": 0.5, "clinical_timeline_confidence": 0.5, "dilution_risk": 0.5, "trial_maturity": 0.4, "endpoint_strength": 0.5, "pipeline_diversification": 0.3},
            ),
            simulation=SimulationConfig(
                n_simulations=n, random_seed=42, monthly_horizon=24,
                use_multistate=use_multistate,
            ),
        )

    def test_multistate_disabled_gives_none(self):
        from app.engines.monte_carlo import run_full_audit
        r = run_full_audit(self._make_request(use_multistate=False))
        assert r.multi_state is None

    def test_multistate_enabled_gives_result(self):
        from app.engines.monte_carlo import run_full_audit
        r = run_full_audit(self._make_request(use_multistate=True, n=500))
        assert r.multi_state is not None

    def test_absorbing_state_probs_are_valid(self):
        from app.engines.monte_carlo import run_full_audit
        r = run_full_audit(self._make_request(use_multistate=True, n=1000))
        ms = r.multi_state
        assert ms is not None
        for name, p in ms.absorbing_state_probs.items():
            assert 0.0 <= p <= 1.0, f"absorbing_state_probs[{name}] = {p}"
        total = sum(ms.absorbing_state_probs.values()) + ms.overall_survival_at_horizon
        assert abs(total - 1.0) <= 0.01, f"State probs don't sum to 1: {total}"

    def test_cif_at_catalyst_keys_are_cause_names(self):
        from app.engines.monte_carlo import run_full_audit
        r = run_full_audit(self._make_request(use_multistate=True, n=500))
        ms = r.multi_state
        assert ms is not None
        assert set(ms.cif_at_catalyst_month.keys()) == set(CAUSE_NAMES.values())

    def test_survival_at_catalyst_is_in_unit_interval(self):
        from app.engines.monte_carlo import run_full_audit
        r = run_full_audit(self._make_request(use_multistate=True, n=500))
        assert r.multi_state is not None
        s = r.multi_state.overall_survival_at_catalyst_month
        assert s is not None and 0.0 <= s <= 1.0
