"""
Bayesian Probability of Technical Success Engine.

Models probability of clinical success (PoS) using a Beta posterior.

Prior: Beta(alpha_prior, beta_prior) — phase-specific
Update: alpha_post = alpha_prior + sum(weights of present positive signals)
        beta_post  = beta_prior  + sum(weights of present negative signals)

Posterior mean: alpha_post / (alpha_post + beta_post)
Credible interval: from Beta(alpha_post, beta_post) CDF

Signal weights are UNTRAINED MVP ASSUMPTIONS. They represent informed
directional priors on what trial characteristics predict success; they have
not been calibrated to historical clinical approval rates.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from app.core.config import CatalystLensConfig, get_default_config
from app.models.schemas import SuccessProbabilityInput, SuccessProbabilityResult


def build_success_prior_by_phase(
    trial_phase: str,
    config: CatalystLensConfig | None = None,
    custom_alpha: float | None = None,
    custom_beta: float | None = None,
) -> Tuple[float, float]:
    """
    Return (alpha_prior, beta_prior) for the given trial phase.

    Custom priors override phase defaults when provided.
    """
    if config is None:
        config = get_default_config()

    if custom_alpha is not None and custom_beta is not None:
        return custom_alpha, custom_beta

    prior = config.phase_priors.get(trial_phase)
    if prior is None:
        prior = config.phase_priors["phase_2"]
    return prior.alpha, prior.beta


def _collect_signal_weights(
    signals_present: List[str],
    weight_map: Dict[str, float],
) -> Dict[str, float]:
    """Return dict of signal → weight for signals that are present."""
    return {s: weight_map[s] for s in signals_present if s in weight_map}


def update_beta_posterior(
    alpha_prior: float,
    beta_prior: float,
    positive_signals: List[str],
    negative_signals: List[str],
    config: CatalystLensConfig | None = None,
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    """
    Update Beta prior with evidence from trial-specific signals.

    Each present positive signal adds its weight to alpha (success evidence).
    Each present negative signal adds its weight to beta (failure evidence).

    Returns: (alpha_post, beta_post, applied_positive, applied_negative)
    """
    if config is None:
        config = get_default_config()

    positive_weights = _collect_signal_weights(
        positive_signals, config.signal_weights.positive
    )
    negative_weights = _collect_signal_weights(
        negative_signals, config.signal_weights.negative
    )

    alpha_post = alpha_prior + sum(positive_weights.values())
    beta_post = beta_prior + sum(negative_weights.values())

    return alpha_post, beta_post, positive_weights, negative_weights


def posterior_mean(alpha: float, beta: float) -> float:
    """E[PoS] = alpha / (alpha + beta)."""
    return alpha / (alpha + beta)


def posterior_interval(
    alpha: float,
    beta: float,
    credibility: float = 0.90,
) -> Tuple[float, float]:
    """
    Equal-tailed credible interval from Beta posterior.

    Returns (lower, upper) for the given credibility level (e.g. 0.90 = 90% CI).
    """
    tail = (1.0 - credibility) / 2.0
    lower = float(stats.beta.ppf(tail, a=alpha, b=beta))
    upper = float(stats.beta.ppf(1.0 - tail, a=alpha, b=beta))
    return lower, upper


def sample_success_probability(
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    n_samples: int,
) -> np.ndarray:
    """
    Sample PoS values from Beta(alpha, beta).

    Each sample represents a plausible probability of technical success
    for this program, given the prior and observed signals.
    """
    return rng.beta(a=alpha, b=beta, size=n_samples)


def run_success_probability_analysis(
    inputs: SuccessProbabilityInput,
    config: CatalystLensConfig | None = None,
) -> SuccessProbabilityResult:
    """Run the full Bayesian PoS analysis."""
    if config is None:
        config = get_default_config()

    alpha_prior, beta_prior = build_success_prior_by_phase(
        inputs.trial_phase,
        config,
        inputs.custom_alpha_prior,
        inputs.custom_beta_prior,
    )

    alpha_post, beta_post, pos_weights, neg_weights = update_beta_posterior(
        alpha_prior, beta_prior,
        inputs.positive_signals, inputs.negative_signals,
        config,
    )

    prior_mean_val = posterior_mean(alpha_prior, beta_prior)
    post_mean_val = posterior_mean(alpha_post, beta_post)
    ci_lower, ci_upper = posterior_interval(alpha_post, beta_post, 0.90)

    return SuccessProbabilityResult(
        alpha_prior=round(alpha_prior, 4),
        beta_prior=round(beta_prior, 4),
        prior_mean=round(prior_mean_val, 4),
        alpha_posterior=round(alpha_post, 4),
        beta_posterior=round(beta_post, 4),
        posterior_mean=round(post_mean_val, 4),
        credible_interval_lower=round(ci_lower, 4),
        credible_interval_upper=round(ci_upper, 4),
        credible_interval_pct=90.0,
        applied_positive_weights={k: round(v, 3) for k, v in pos_weights.items()},
        applied_negative_weights={k: round(v, 3) for k, v in neg_weights.items()},
        model_assumptions=[
            "Beta-binomial Bayesian model with additive signal weight updates.",
            "Signal weights are UNTRAINED MVP ASSUMPTIONS (see config.py).",
            "Phase-specific priors encode historical industry PoS rates directionally.",
            "Posterior PoS is a model estimate, not a validated prediction of approval.",
            "Unknown signals are neither positive nor negative (absent from update).",
        ],
    )
