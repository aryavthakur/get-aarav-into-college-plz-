"""
Tests for the Bayesian probability of technical success engine.
"""

import numpy as np
import pytest
from scipy import stats

from app.engines.bayesian_success import (
    build_success_prior_by_phase,
    posterior_interval,
    posterior_mean,
    run_success_probability_analysis,
    sample_success_probability,
    update_beta_posterior,
)
from app.models.schemas import SuccessProbabilityInput


class TestPriorsByPhase:
    def test_preclinical_prior_mean_is_low(self):
        alpha, beta = build_success_prior_by_phase("preclinical")
        assert posterior_mean(alpha, beta) < 0.20

    def test_phase_1_prior_mean(self):
        alpha, beta = build_success_prior_by_phase("phase_1")
        assert 0.10 <= posterior_mean(alpha, beta) <= 0.35

    def test_phase_3_prior_higher_than_phase_2(self):
        a2, b2 = build_success_prior_by_phase("phase_2")
        a3, b3 = build_success_prior_by_phase("phase_3")
        assert posterior_mean(a3, b3) > posterior_mean(a2, b2)

    def test_filed_prior_high(self):
        alpha, beta = build_success_prior_by_phase("filed")
        assert posterior_mean(alpha, beta) > 0.70

    def test_custom_prior_overrides_phase(self):
        alpha, beta = build_success_prior_by_phase(
            "phase_1", custom_alpha=10.0, custom_beta=2.0
        )
        assert alpha == 10.0
        assert beta == 2.0

    def test_unknown_phase_falls_back_gracefully(self):
        alpha, beta = build_success_prior_by_phase("unknown_phase")
        # Should not raise; falls back to phase_2
        assert alpha > 0
        assert beta > 0


class TestPosteriorUpdate:
    def test_positive_signals_increase_posterior_mean(self):
        alpha_prior, beta_prior = build_success_prior_by_phase("phase_2")
        prior_mean = posterior_mean(alpha_prior, beta_prior)
        alpha_post, beta_post, _, _ = update_beta_posterior(
            alpha_prior, beta_prior,
            positive_signals=["validated_biomarker", "randomized_controlled_design"],
            negative_signals=[],
        )
        assert posterior_mean(alpha_post, beta_post) > prior_mean

    def test_negative_signals_decrease_posterior_mean(self):
        alpha_prior, beta_prior = build_success_prior_by_phase("phase_2")
        prior_mean = posterior_mean(alpha_prior, beta_prior)
        alpha_post, beta_post, _, _ = update_beta_posterior(
            alpha_prior, beta_prior,
            positive_signals=[],
            negative_signals=["safety_concerns", "prior_failed_trials"],
        )
        assert posterior_mean(alpha_post, beta_post) < prior_mean

    def test_no_signals_preserves_prior(self):
        alpha_prior, beta_prior = build_success_prior_by_phase("phase_2")
        alpha_post, beta_post, pos_w, neg_w = update_beta_posterior(
            alpha_prior, beta_prior, [], []
        )
        assert alpha_post == alpha_prior
        assert beta_post == beta_prior
        assert len(pos_w) == 0
        assert len(neg_w) == 0

    def test_unknown_signals_are_ignored(self):
        alpha_prior, beta_prior = build_success_prior_by_phase("phase_2")
        alpha_post, beta_post, pos_w, neg_w = update_beta_posterior(
            alpha_prior, beta_prior,
            positive_signals=["nonexistent_signal_xyz"],
            negative_signals=[],
        )
        assert alpha_post == alpha_prior
        assert len(pos_w) == 0

    def test_strong_positive_update_alpha_increases_by_weights(self):
        alpha_prior, beta_prior = 3.0, 7.0
        alpha_post, beta_post, pos_w, _ = update_beta_posterior(
            alpha_prior, beta_prior,
            positive_signals=["validated_biomarker"],
            negative_signals=[],
        )
        assert alpha_post == pytest.approx(alpha_prior + pos_w.get("validated_biomarker", 0))


class TestCredibleInterval:
    def test_interval_contains_posterior_mean(self):
        alpha, beta = 5.0, 10.0
        pm = posterior_mean(alpha, beta)
        lower, upper = posterior_interval(alpha, beta, 0.90)
        assert lower < pm < upper

    def test_interval_bounds_ordered(self):
        lower, upper = posterior_interval(4.0, 8.0, 0.90)
        assert lower < upper

    def test_interval_bounds_in_0_1(self):
        lower, upper = posterior_interval(3.0, 7.0, 0.95)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_wider_credibility_gives_wider_interval(self):
        alpha, beta = 3.0, 7.0
        l90, u90 = posterior_interval(alpha, beta, 0.90)
        l50, u50 = posterior_interval(alpha, beta, 0.50)
        assert (u90 - l90) > (u50 - l50)


class TestSampling:
    def test_samples_are_between_0_and_1(self):
        rng = np.random.default_rng(42)
        samples = sample_success_probability(rng, alpha=3.0, beta=7.0, n_samples=1000)
        assert np.all((samples >= 0.0) & (samples <= 1.0))

    def test_sample_mean_close_to_posterior_mean(self):
        rng = np.random.default_rng(42)
        alpha, beta = 5.0, 10.0
        expected_mean = posterior_mean(alpha, beta)
        samples = sample_success_probability(rng, alpha=alpha, beta=beta, n_samples=50000)
        assert np.mean(samples) == pytest.approx(expected_mean, abs=0.01)

    def test_samples_count_correct(self):
        rng = np.random.default_rng(42)
        samples = sample_success_probability(rng, 3.0, 7.0, n_samples=500)
        assert len(samples) == 500


class TestRunSuccessProbabilityAnalysis:
    def test_result_fields_present(self):
        inputs = SuccessProbabilityInput(
            trial_phase="phase_2",
            positive_signals=["validated_biomarker"],
            negative_signals=["small_sample_size"],
        )
        result = run_success_probability_analysis(inputs)
        assert 0 < result.posterior_mean < 1
        assert result.credible_interval_lower < result.posterior_mean
        assert result.posterior_mean < result.credible_interval_upper

    def test_result_probabilities_in_range(self):
        inputs = SuccessProbabilityInput(trial_phase="phase_3", positive_signals=[], negative_signals=[])
        result = run_success_probability_analysis(inputs)
        assert 0 <= result.posterior_mean <= 1
        assert 0 <= result.credible_interval_lower <= 1
        assert 0 <= result.credible_interval_upper <= 1
