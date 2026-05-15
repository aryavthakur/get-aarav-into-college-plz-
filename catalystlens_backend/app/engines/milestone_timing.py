"""
Milestone Timing Engine — Scientific Clock.

Models time-to-catalyst using a Gamma distribution.

T_sci ~ Gamma(alpha, beta_rate)

where:
  mean    = alpha / beta_rate  ≈ adjusted stated timeline
  std_dev = mean * CV          (CV scales with trial complexity)

Parameterization:
  alpha    = 1 / CV^2
  beta_rate = 1 / (mean * CV^2) = alpha / mean

The Gamma distribution captures the right-skewed, long-tailed nature of
clinical development delays: most trials run roughly on schedule, but a
meaningful minority experience severe delays.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy import stats

from app.core.config import CatalystLensConfig, MilestoneTimingParams, get_default_config
from app.models.schemas import ClinicalCatalystInput, MilestoneTimingResult


_STATUS_DELAY_MULTIPLIER = {
    "not_yet_recruiting": 1.30,
    "recruiting": 1.10,
    "active_not_recruiting": 1.05,
    "completed": 1.00,
    "suspended": 1.50,
    "withdrawn": 1.80,
}


def _enrollment_remaining_months(clinical: ClinicalCatalystInput) -> float:
    """Estimated months to complete enrollment from today."""
    remaining = max(0, clinical.enrollment_target - clinical.enrollment_completed)
    if clinical.enrollment_rate_per_month <= 0:
        return clinical.stated_months_to_catalyst
    return remaining / clinical.enrollment_rate_per_month


def _compute_delay_factor(
    clinical: ClinicalCatalystInput,
    params: MilestoneTimingParams,
) -> float:
    """
    Compute a delay multiplier applied to the stated timeline.

    Sources of delay:
    1. Optimism bias base factor (management timelines skew early)
    2. Enrollment progress (remaining enrollment implies minimum timeline)
    3. Complexity adjustments (indication, endpoint, regulatory)
    4. Trial status
    5. Site count (few sites → slower enrollment)
    """
    factor = params.base_delay_factor

    # Enrollment-based delay: incomplete enrollment extends timeline
    enrollment_fraction = clinical.enrollment_completed / clinical.enrollment_target
    if enrollment_fraction < 1.0:
        incomplete_fraction = 1.0 - enrollment_fraction
        factor += params.enrollment_delay_weight * incomplete_fraction

    # Complexity adjustments
    avg_complexity = (
        clinical.indication_complexity_score
        + clinical.endpoint_complexity_score
        + clinical.regulatory_complexity_score
    ) / 3.0
    factor += params.complexity_delay_weight * avg_complexity

    # Trial status
    factor *= _STATUS_DELAY_MULTIPLIER.get(clinical.trial_status, 1.1)

    # Few sites constrain enrollment speed (order matters: most restrictive condition first)
    if clinical.number_of_sites < 5:
        factor *= 1.20
    elif clinical.number_of_sites < 10:
        factor *= 1.10

    return max(1.0, factor)


def _compute_cv(
    clinical: ClinicalCatalystInput,
    params: MilestoneTimingParams,
) -> float:
    """
    Compute coefficient of variation for the Gamma milestone distribution.

    Higher CV = wider, more uncertain distribution.
    CV is anchored to a phase-specific base, then adjusted upward for complexity.
    """
    base_cv = params.base_cv_by_phase.get(clinical.trial_phase, 0.40)

    avg_complexity = (
        clinical.indication_complexity_score
        + clinical.endpoint_complexity_score
        + clinical.regulatory_complexity_score
    ) / 3.0
    cv = base_cv + params.complexity_cv_weight * avg_complexity

    # Enrollment far from completion adds timing uncertainty
    enrollment_fraction = clinical.enrollment_completed / clinical.enrollment_target
    if enrollment_fraction < 0.5:
        cv += 0.08 * (0.5 - enrollment_fraction) * 2

    return max(0.05, cv)


def estimate_gamma_parameters(
    clinical: ClinicalCatalystInput,
    config: CatalystLensConfig | None = None,
) -> Tuple[float, float, float, float]:
    """
    Estimate Gamma(alpha, beta_rate) parameters for milestone timing.

    Returns: (alpha, beta_rate, adjusted_mean, cv)
    """
    if config is None:
        config = get_default_config()
    params = config.milestone_timing

    delay_factor = _compute_delay_factor(clinical, params)
    cv = _compute_cv(clinical, params)

    # Adjusted mean, but floor at decomposed public-readout path.
    enroll_remaining = _enrollment_remaining_months(clinical)
    public_readout_lag = (
        clinical.followup_months_after_enrollment
        + clinical.data_cleaning_months
        + clinical.analysis_months
        + clinical.disclosure_lag_months
    )
    min_months = enroll_remaining + public_readout_lag
    stated_adjusted = clinical.stated_months_to_catalyst * delay_factor
    adjusted_mean = max(stated_adjusted, min_months, 1.0)

    alpha = 1.0 / (cv ** 2)
    beta_rate = alpha / adjusted_mean

    return alpha, beta_rate, adjusted_mean, cv


def expected_time_to_milestone(clinical: ClinicalCatalystInput) -> float:
    """Return the model-estimated mean time to milestone (months)."""
    _, _, adjusted_mean, _ = estimate_gamma_parameters(clinical)
    return adjusted_mean


def probability_milestone_before_month(
    month: float,
    alpha: float,
    beta_rate: float,
) -> float:
    """P(T_sci < month) = CDF of Gamma evaluated at month."""
    if month <= 0:
        return 0.0
    return float(stats.gamma.cdf(month, a=alpha, scale=1.0 / beta_rate))


def sample_scientific_milestone_time(
    rng: np.random.Generator,
    alpha: float,
    beta_rate: float,
    n_samples: int,
) -> np.ndarray:
    """
    Sample milestone times from Gamma(alpha, scale=1/beta_rate).

    Returns array of positive timing samples in months.
    """
    samples = rng.gamma(shape=alpha, scale=1.0 / beta_rate, size=n_samples)
    return np.maximum(samples, 0.1)


def run_milestone_timing_analysis(
    clinical: ClinicalCatalystInput,
    config: CatalystLensConfig | None = None,
    n_quantile_samples: int = 50_000,
    seed: int = 0,
) -> MilestoneTimingResult:
    """Run the full milestone timing analysis."""
    if config is None:
        config = get_default_config()

    alpha, beta_rate, adj_mean, cv = estimate_gamma_parameters(clinical, config)
    delay_factor = _compute_delay_factor(clinical, config.milestone_timing)
    enroll_remaining = _enrollment_remaining_months(clinical)
    enrollment_fraction = clinical.enrollment_completed / clinical.enrollment_target
    primary_completion_months = enroll_remaining + clinical.followup_months_after_enrollment
    public_readout_lag = (
        clinical.data_cleaning_months
        + clinical.analysis_months
        + clinical.disclosure_lag_months
    )
    public_readout_months = primary_completion_months + public_readout_lag

    rng = np.random.default_rng(seed)
    samples = sample_scientific_milestone_time(rng, alpha, beta_rate, n_quantile_samples)

    return MilestoneTimingResult(
        gamma_alpha=round(alpha, 4),
        gamma_beta_rate=round(beta_rate, 6),
        stated_months=clinical.stated_months_to_catalyst,
        adjusted_mean_months=round(adj_mean, 2),
        delay_factor=round(delay_factor, 4),
        cv=round(cv, 4),
        enrollment_fraction=round(enrollment_fraction, 4),
        enrollment_remaining_months=round(enroll_remaining, 2),
        enrollment_component_months=round(enroll_remaining, 2),
        followup_component_months=round(clinical.followup_months_after_enrollment, 2),
        data_cleaning_component_months=round(clinical.data_cleaning_months, 2),
        analysis_component_months=round(clinical.analysis_months, 2),
        disclosure_lag_months=round(clinical.disclosure_lag_months, 2),
        primary_completion_months=round(primary_completion_months, 2),
        public_readout_lag_months=round(public_readout_lag, 2),
        public_readout_months=round(public_readout_months, 2),
        p5_months=round(float(np.percentile(samples, 5)), 2),
        p25_months=round(float(np.percentile(samples, 25)), 2),
        p50_months=round(float(np.percentile(samples, 50)), 2),
        p75_months=round(float(np.percentile(samples, 75)), 2),
        p95_months=round(float(np.percentile(samples, 95)), 2),
        model_assumptions=[
            "Gamma distribution captures right-skewed clinical delay risk.",
            "Stated catalyst timing treated as management estimate subject to optimism bias.",
            f"Base delay factor: {config.milestone_timing.base_delay_factor}x stated timeline.",
            "Minimum timeline floored at enrollment_remaining * buffer to prevent impossible catalysts.",
            "Primary completion and public readout are modeled as separate timing concepts.",
            "CV is phase-anchored and increases with indication/endpoint/regulatory complexity.",
        ],
    )
