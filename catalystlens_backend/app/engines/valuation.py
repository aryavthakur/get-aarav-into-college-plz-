"""
Valuation Engine — Risk-Adjusted NPV with Four-State Financing Model.

The original binary funded/not-funded treatment was internally inconsistent:
a company could be marked as "cashout before catalyst" yet still receive
near-full success valuation via a minor dilution haircut. This rewrite
implements four explicit financing states that properly separate
cash exhaustion, refinancing outcomes, and program discontinuation.

FINANCING STATES
================
  0 — FUNDED:         T_fin >= T_sci. Company reaches catalyst without raising.
  1 — REFINANCED:     T_fin < T_sci. Company raises at market terms before cashout.
  2 — DISTRESSED:     T_fin < T_sci. Company raises but at heavy dilution / discount.
  3 — DISCONTINUED:   T_fin < T_sci. Company cannot raise. Program stops.

VALUATION PER STATE
===================
  FUNDED + success:       asset_value * discount_factor
  FUNDED + failure:       downside_value
  REFINANCED + success:   asset_value * discount_factor * (1 - dilution)
  REFINANCED + failure:   downside_value
  DISTRESSED + success:   asset_value * discount_factor * (1 - dilution * distressed_mult)
  DISTRESSED + failure:   downside_value
  DISCONTINUED:           downside_value (regardless of technical outcome)

The probability of successful refinancing falls with:
  - larger financing gap (more months of cash needed)
  - worse biotech market conditions

Distressed financing is triggered when:
  - financing gap > 12 months  OR
  - market condition score < 4

IMPORTANT: rNPV outputs are highly sensitive to user-supplied asset_value_success
and discount rate. These are inputs, not model outputs.
"""

from __future__ import annotations

import numpy as np

from app.core.config import CatalystLensConfig, get_default_config
from app.models.schemas import ValuationInput, ValuationResult


def _clamp_prob(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _compute_financing_states(
    t_sci: np.ndarray,
    t_fin: np.ndarray,
    rng: np.random.Generator,
    market_condition_score: float,
    expected_dilution: float,
    financing_penalty_strength: float,
    config: CatalystLensConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify each simulation into one of four financing states.

    Returns
    -------
    states : int array, values in {0, 1, 2, 3}
    value_adjustments : float array, multiplicative factor applied to asset_value_success
    """
    n = len(t_sci)
    states = np.zeros(n, dtype=int)

    funded_mask = t_fin >= t_sci
    needs_financing = ~funded_mask
    gap = np.maximum(t_sci - t_fin, 0.0)  # months of gap; zero if funded

    # Probability of successful refinancing conditional on needing it
    p_refin = np.clip(
        config.valuation_params.base_refinancing_success_prob
        + (market_condition_score - 5.0) * config.valuation_params.market_condition_refinancing_effect
        - gap * config.valuation_params.gap_refinancing_penalty_per_month,
        0.05,
        0.90,
    )

    # Sample refinancing success only for simulations that need financing
    refin_needed_prob = p_refin * needs_financing.astype(float)
    refin_success = rng.binomial(1, refin_needed_prob).astype(bool)

    # Distressed: gap too large or market too weak to support clean refinancing
    is_distressed = (
        (gap > config.valuation_params.distressed_gap_threshold)
        | (market_condition_score < config.valuation_params.distressed_market_threshold)
    )

    # Assign states
    states[funded_mask] = 0
    states[needs_financing & refin_success & ~is_distressed] = 1   # clean refinancing
    states[needs_financing & refin_success & is_distressed] = 2    # distressed
    states[needs_financing & ~refin_success] = 3                   # discontinued

    # Value adjustments
    adjustments = np.ones(n, dtype=float)
    adjustments[states == 1] = 1.0 - expected_dilution * financing_penalty_strength
    adjustments[states == 2] = np.clip(
        1.0
        - expected_dilution
        * config.valuation_params.distressed_dilution_multiplier
        * financing_penalty_strength,
        0.0,
        1.0,
    )
    adjustments[states == 3] = 0.0   # no equity value; falls to downside

    return states, adjustments


def run_valuation_simulation(
    t_sci_samples: np.ndarray,
    t_fin_samples: np.ndarray,
    pos_samples: np.ndarray,
    inputs: ValuationInput,
    rng: np.random.Generator,
    market_condition_score: float = 5.0,
    config: CatalystLensConfig | None = None,
) -> ValuationResult:
    """
    Compute the full Monte Carlo valuation distribution using the four-state
    financing model.

    Parameters
    ----------
    t_sci_samples       : scientific milestone times (months)
    t_fin_samples       : financial failure times (months) from Cox-Weibull model
    pos_samples         : sampled PoS values from Beta posterior
    inputs              : ValuationInput schema
    rng                 : NumPy random generator
    market_condition_score : biotech financing market condition (1–10); used to
                            adjust probability of successful refinancing

    Returns
    -------
    ValuationResult
    """
    if config is None:
        config = get_default_config()

    n = len(t_sci_samples)

    # Sample binary technical success for each simulation
    success = rng.binomial(1, pos_samples).astype(bool)

    # Determine financing state and per-simulation value adjustment
    states, value_adj = _compute_financing_states(
        t_sci_samples, t_fin_samples, rng,
        market_condition_score,
        inputs.expected_dilution_if_refinanced,
        inputs.financing_penalty_strength,
        config,
    )

    # Time-discount applied to milestone timing (discounts value for later catalysts)
    monthly_rate = inputs.annual_discount_rate / 12.0
    discount = 1.0 / ((1.0 + monthly_rate) ** t_sci_samples)

    # Per-simulation value:
    #   DISCONTINUED (state 3): value_adj == 0 → collapses to downside regardless of success
    #   FUNDED/REFINANCED/DISTRESSED + success: asset_value * discount * adjustment
    #   FUNDED/REFINANCED/DISTRESSED + failure: downside_value
    values = np.where(
        success & (states != 3),
        inputs.asset_value_success * discount * value_adj,
        float(inputs.downside_value),
    )

    # Technical-risk-only rNPV: no financing risk, no dilution
    mean_pos = float(np.mean(pos_samples))
    mean_discount = float(np.mean(discount))
    technical_rnpv = (
        inputs.asset_value_success * mean_pos * mean_discount
        + (1.0 - mean_pos) * float(inputs.downside_value)
    )

    financing_rnpv = float(np.mean(values))
    financing_discount_value = technical_rnpv - financing_rnpv

    # High-upside threshold: 50% of gross asset value
    high_upside_threshold = inputs.asset_value_success * 0.50
    prob_high_upside = float(np.mean(values >= high_upside_threshold))
    prob_downside = float(np.mean(values <= inputs.downside_value * 1.01))

    # Per-state value averages (for reporting)
    def _mean_state_value(state_id: int) -> float:
        mask = states == state_id
        return float(np.mean(values[mask])) if mask.any() else 0.0

    p_funded = _clamp_prob(float(np.mean(states == 0)))
    p_clean = _clamp_prob(float(np.mean(states == 1)))
    p_distressed = _clamp_prob(float(np.mean(states == 2)))
    p_discontinued = _clamp_prob(float(np.mean(states == 3)))
    p_partnership = 0.0
    p_debt_or_royalty = 0.0
    p_cash_exhaustion = 0.0
    p_any_financing = _clamp_prob(p_clean + p_distressed + p_partnership + p_debt_or_royalty + p_cash_exhaustion)
    p_pressure = _clamp_prob(p_distressed + p_cash_exhaustion + p_discontinued)
    p_nondilutive = _clamp_prob(p_partnership + p_debt_or_royalty)
    p_dilutive = _clamp_prob(p_clean + p_distressed)

    return ValuationResult(
        mean_value=round(float(np.mean(values)), 2),
        median_value=round(float(np.median(values)), 2),
        p5_value=round(float(np.percentile(values, 5)), 2),
        p95_value=round(float(np.percentile(values, 95)), 2),
        technical_risk_only_rnpv=round(technical_rnpv, 2),
        financing_adjusted_rnpv=round(financing_rnpv, 2),
        financing_risk_discount=round(financing_discount_value, 2),
        probability_downside=round(prob_downside, 4),
        probability_high_upside=round(prob_high_upside, 4),
        high_upside_threshold=round(high_upside_threshold, 2),
        p_funded_through_catalyst=round(p_funded, 4),
        p_refinancing_success=round(p_clean, 4),
        p_distressed_financing=round(p_distressed, 4),
        p_program_discontinuation=round(p_discontinued, 4),
        p_clean_refinancing_before_catalyst=round(p_clean, 4),
        p_distressed_refinancing_before_catalyst=round(p_distressed, 4),
        p_partnership_before_catalyst=round(p_partnership, 4),
        p_debt_or_royalty_before_catalyst=round(p_debt_or_royalty, 4),
        p_cash_exhaustion_before_catalyst=round(p_cash_exhaustion, 4),
        p_program_discontinuation_before_catalyst=round(p_discontinued, 4),
        p_any_financing_event_before_catalyst=round(p_any_financing, 4),
        p_financing_pressure_before_catalyst=round(p_pressure, 4),
        p_nondilutive_financing_before_catalyst=round(p_nondilutive, 4),
        p_dilutive_financing_before_catalyst=round(p_dilutive, 4),
        mean_value_if_funded=round(_mean_state_value(0), 2),
        mean_value_if_refinanced=round(_mean_state_value(1), 2),
        mean_value_if_distressed=round(_mean_state_value(2), 2),
        model_assumptions=[
            "Four financing states: Funded / Refinanced / Distressed / Discontinued.",
            "Refinancing success probability = f(market conditions, financing gap).",
            (
                "Distressed financing triggered when gap exceeds configured threshold "
                "or market score is below configured threshold."
            ),
            "Distressed dilution uses the configured distressed dilution multiplier.",
            "financing_penalty_strength scales clean and distressed refinancing haircuts.",
            "Discontinued state forces value to downside regardless of technical outcome.",
            "Asset value on success is a user-supplied assumption, not a model output.",
            "Discount rate is user-supplied; default 12% reflects typical biotech WACC.",
            "rNPV outputs are highly sensitive to asset_value_success and discount rate.",
            "This is not investment advice.",
        ],
    )
