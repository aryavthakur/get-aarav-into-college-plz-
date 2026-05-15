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

from app.models.schemas import ValuationInput, ValuationResult

# Multiplier applied to expected_dilution_if_refinanced in distressed state
_DISTRESSED_DILUTION_MULTIPLIER = 2.0

# Gap threshold (months) above which a refinancing is classified as distressed
_DISTRESSED_GAP_THRESHOLD = 12.0

# Market condition score below which all refinancings are classified as distressed
_DISTRESSED_MARKET_THRESHOLD = 4.0

# Base probability of successful refinancing when needed (at market_score=5, gap=0)
_BASE_REFINANCING_SUCCESS_PROB = 0.60

# How much each point of market condition shifts refinancing success probability
_MARKET_CONDITION_REFIN_EFFECT = 0.04

# How much each month of financing gap reduces refinancing success probability
_GAP_REFIN_PENALTY_PER_MONTH = 0.015


def _compute_financing_states(
    t_sci: np.ndarray,
    t_fin: np.ndarray,
    rng: np.random.Generator,
    market_condition_score: float,
    expected_dilution: float,
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
        _BASE_REFINANCING_SUCCESS_PROB
        + (market_condition_score - 5.0) * _MARKET_CONDITION_REFIN_EFFECT
        - gap * _GAP_REFIN_PENALTY_PER_MONTH,
        0.05,
        0.90,
    )

    # Sample refinancing success only for simulations that need financing
    refin_needed_prob = p_refin * needs_financing.astype(float)
    refin_success = rng.binomial(1, refin_needed_prob).astype(bool)

    # Distressed: gap too large or market too weak to support clean refinancing
    is_distressed = (gap > _DISTRESSED_GAP_THRESHOLD) | (market_condition_score < _DISTRESSED_MARKET_THRESHOLD)

    # Assign states
    states[funded_mask] = 0
    states[needs_financing & refin_success & ~is_distressed] = 1   # clean refinancing
    states[needs_financing & refin_success & is_distressed] = 2    # distressed
    states[needs_financing & ~refin_success] = 3                   # discontinued

    # Value adjustments
    adjustments = np.ones(n, dtype=float)
    adjustments[states == 1] = 1.0 - expected_dilution
    adjustments[states == 2] = np.clip(
        1.0 - expected_dilution * _DISTRESSED_DILUTION_MULTIPLIER, 0.0, 1.0
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
    n = len(t_sci_samples)

    # Sample binary technical success for each simulation
    success = rng.binomial(1, pos_samples).astype(bool)

    # Determine financing state and per-simulation value adjustment
    states, value_adj = _compute_financing_states(
        t_sci_samples, t_fin_samples, rng,
        market_condition_score, inputs.expected_dilution_if_refinanced,
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
    technical_rnpv = inputs.asset_value_success * mean_pos * mean_discount

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
        p_funded_through_catalyst=round(float(np.mean(states == 0)), 4),
        p_refinancing_success=round(float(np.mean(states == 1)), 4),
        p_distressed_financing=round(float(np.mean(states == 2)), 4),
        p_program_discontinuation=round(float(np.mean(states == 3)), 4),
        mean_value_if_funded=round(_mean_state_value(0), 2),
        mean_value_if_refinanced=round(_mean_state_value(1), 2),
        mean_value_if_distressed=round(_mean_state_value(2), 2),
        model_assumptions=[
            "Four financing states: Funded / Refinanced / Distressed / Discontinued.",
            "Refinancing success probability = f(market conditions, financing gap).",
            "Distressed financing triggered when gap > 12 months or market score < 4.",
            "Distressed dilution = 2× standard dilution assumption.",
            "Discontinued state forces value to downside regardless of technical outcome.",
            "Asset value on success is a user-supplied assumption, not a model output.",
            "Discount rate is user-supplied; default 12% reflects typical biotech WACC.",
            "rNPV outputs are highly sensitive to asset_value_success and discount rate.",
            "This is not investment advice.",
        ],
    )
