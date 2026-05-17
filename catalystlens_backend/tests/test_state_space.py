"""
Tests for Bayesian state-space particle filter engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engines.state_space import (
    StateSpaceParams,
    extract_estimate,
    initialise_particles,
    propagate,
    run_particle_filter,
    run_state_space_analysis,
    update,
)
from app.engines.monte_carlo import run_full_audit


class TestParticleFilterMath:
    def _params(self, n=500):
        return StateSpaceParams(n_particles=n)

    def test_initial_weights_sum_to_one(self):
        params = self._params()
        obs = np.array([2.5, 0.0, 0.5, 0.2])
        state = initialise_particles(obs, params, np.random.default_rng(0))
        assert abs(state.weights.sum() - 1.0) < 1e-10

    def test_particle_shape(self):
        params = self._params(n=200)
        obs = np.array([1.0, 0.0, 0.5, 0.0])
        state = initialise_particles(obs, params, np.random.default_rng(0))
        assert state.particles.shape == (200, 4)
        assert state.weights.shape == (200,)

    def test_weights_after_update_sum_to_one(self):
        params = self._params()
        obs = np.array([2.0, -0.2, 0.4, 0.1])
        state = initialise_particles(obs, params, np.random.default_rng(0))
        state = update(state, obs, params, np.random.default_rng(1))
        assert abs(state.weights.sum() - 1.0) < 1e-8

    def test_weights_nonnegative(self):
        params = self._params()
        obs = np.array([1.5, 0.3, 0.6, -0.1])
        state = initialise_particles(obs, params, np.random.default_rng(0))
        state = update(state, obs, params, np.random.default_rng(1))
        assert np.all(state.weights >= 0.0)

    def test_propagate_changes_particles(self):
        params = self._params(n=100)
        obs = np.array([2.0, 0.0, 0.5, 0.0])
        state = initialise_particles(obs, params, np.random.default_rng(0))
        state2 = propagate(state, params, np.random.default_rng(1))
        # Particles should differ after propagation (noise added)
        assert not np.allclose(state.particles, state2.particles)

    def test_ess_between_1_and_n(self):
        params = self._params(n=500)
        obs = np.array([2.0, 0.0, 0.5, 0.0])
        state = initialise_particles(obs, params, np.random.default_rng(0))
        state = update(state, obs, params, np.random.default_rng(1))
        est = extract_estimate(state, params)
        assert 1.0 <= est.effective_sample_size <= 500.0

    def test_consistent_state_with_extreme_observation(self):
        """Particle filter should converge toward extreme observation."""
        params = StateSpaceParams(n_particles=2000, observation_noise=0.05)
        obs = np.array([3.5, 0.0, 0.8, 0.5])  # high runway, good clinical
        state = initialise_particles(obs, params, np.random.default_rng(0))
        state = update(state, obs, params, np.random.default_rng(1))
        est = extract_estimate(state, params)
        # Posterior mean for log_runway should be near obs[0]
        assert abs(est.posterior_mean[0] - obs[0]) < 0.8


class TestStateSpaceAnalysis:
    def test_scores_in_unit_interval(self):
        rng = np.random.default_rng(0)
        r = run_state_space_analysis(
            cash_months_runway=15.0,
            burn_acceleration=1.0,
            enrollment_fraction=0.5,
            biotech_market_score=5.0,
            rng=rng,
        )
        assert 0.0 <= r.cash_health_score <= 1.0
        assert 0.0 <= r.burn_acceleration_signal <= 1.0
        assert 0.0 <= r.clinical_progress_signal <= 1.0
        assert 0.0 <= r.market_condition_signal <= 1.0
        assert 0.0 <= r.anomaly_score <= 1.0

    def test_high_runway_gives_high_cash_health(self):
        r_high = run_state_space_analysis(36.0, 1.0, 0.5, 5.0, np.random.default_rng(0))
        r_low = run_state_space_analysis(3.0, 1.0, 0.5, 5.0, np.random.default_rng(0))
        assert r_high.cash_health_score >= r_low.cash_health_score

    def test_interpretation_nonempty(self):
        r = run_state_space_analysis(12.0, 1.0, 0.5, 5.0, np.random.default_rng(0))
        assert len(r.interpretation) > 10

    def test_particle_filter_seq(self):
        rng = np.random.default_rng(42)
        obs = np.array([
            [2.5, 0.0, 0.3, 0.2],
            [2.3, 0.1, 0.5, 0.1],
            [2.1, 0.2, 0.7, 0.0],
        ])
        params = StateSpaceParams(n_particles=500)
        estimates, final_state = run_particle_filter(obs, params, rng)
        assert len(estimates) == 3
        for est in estimates:
            assert est.effective_sample_size > 0
            assert len(est.posterior_mean) == params.d


class TestStateSpaceIntegration:
    def _request(self, n=300):
        from app.models.schemas import (
            AuditRequest, ClinicalCatalystInput, CompanyFinancialInput,
            DisclosureInput, SimulationConfig, SuccessProbabilityInput, ValuationInput,
        )
        return AuditRequest(
            financial=CompanyFinancialInput(
                company_name="SSCo", ticker="SSC",
                cash_on_hand=18_000_000, marketable_securities=0,
                quarterly_operating_cash_burn=4_000_000, market_cap=65_000_000,
            ),
            clinical=ClinicalCatalystInput(
                asset_name="SSC-01", indication="CNS",
                trial_phase="phase_2", trial_status="recruiting",
                stated_months_to_catalyst=18,
                enrollment_target=70, enrollment_completed=25,
                enrollment_rate_per_month=4, number_of_sites=7,
            ),
            success_probability=SuccessProbabilityInput(trial_phase="phase_2"),
            valuation=ValuationInput(asset_value_success=180_000_000),
            disclosure=DisclosureInput(
                company_narrative_distribution={"runway_strength": 0.7, "clinical_timeline_confidence": 0.7, "dilution_risk": 0.3, "trial_maturity": 0.5, "endpoint_strength": 0.6, "pipeline_diversification": 0.4},
                structured_audit_distribution={"runway_strength": 0.5, "clinical_timeline_confidence": 0.5, "dilution_risk": 0.5, "trial_maturity": 0.4, "endpoint_strength": 0.5, "pipeline_diversification": 0.4},
            ),
            simulation=SimulationConfig(n_simulations=n, random_seed=42, monthly_horizon=24),
        )

    @pytest.fixture(scope="class")
    def audit_result(self):
        return run_full_audit(TestStateSpaceIntegration()._request(n=300))

    def test_state_space_populated(self, audit_result):
        r = audit_result
        assert r.state_space is not None

    def test_state_space_scores_in_unit_interval(self, audit_result):
        r = audit_result
        ss = r.state_space
        for attr in ["cash_health_score", "burn_acceleration_signal",
                     "clinical_progress_signal", "market_condition_signal", "anomaly_score"]:
            v = getattr(ss, attr)
            assert 0.0 <= v <= 1.0, f"{attr} = {v} out of [0, 1]"

    def test_state_space_report_section(self, audit_result):
        r = audit_result
        assert "State-Space Model" in r.markdown_report

    def test_state_space_methodology_discloses_single_snapshot(self, audit_result):
        r = audit_result
        note = r.state_space.methodology_note.lower()
        report = r.markdown_report.lower()
        assert "single-snapshot" in note
        assert "not a fully dynamic historical filter" in report
