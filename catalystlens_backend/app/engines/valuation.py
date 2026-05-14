"""
Valuation Engine — Risk-Adjusted NPV / Monte Carlo Synthesis.

For each simulation:
  1. Sample PoS from Beta(alpha_post, beta_post)
  2. Sample binary success from Bernoulli(PoS)
  3. Use pre-sampled T_sci and T_fin
  4. Determine: funded_at_catalyst = (T_sci < T_fin)
  5. Apply discount factor: 1 / (1 + r)^(T_sci / 12)
  6. Apply financing penalty if refinancing is modelled
  7. Compute scenario value

Value cases:
  - funded + success:  asset_value * discount * financing_penalty
  - funded + failure:  downside_value
  - not funded + success (refinanced): asset_value * discount * (1 - dilution) * penalty
  - not funded + failure: downside_value

Technical-risk-only rNPV (ignoring financing):
  = asset_value * mean_pos * mean_discount_factor

The difference between technical-only and financing-adjusted rNPV measures
value erosion from financing risk.

IMPORTANT: rNPV outputs are highly sensitive to asset_value_success and
discount rate assumptions. These are user-supplied inputs, not model outputs.
"""

from __future__ import annotations

import numpy as np

from app.models.schemas import ValuationInput, ValuationResult


def _financing_adjustment(
    t_sci: np.ndarray,
    t_fin: np.ndarray,
    funded: np.ndarray,
    dilution: float,
    penalty_strength: float,
) -> np.ndarray:
    """
    Per-simulation financing adjustment factor.

    - Funded with comfortable margin (gap > 6 months): no penalty
    - Funded but margin < 6 months: partial dilution penalty
    - Not funded (must refinance): full dilution penalty scaled by penalty_strength
    """
    gap = t_fin - t_sci
    n = len(t_sci)
    adjustment = np.ones(n, dtype=float)

    # Not funded: company must raise capital with dilution
    not_funded_mask = ~funded
    adjustment[not_funded_mask] = (
        1.0 - dilution * penalty_strength
    )

    # Funded but close to cashout (within 6 months)
    tight_margin_mask = funded & (gap < 6.0) & (gap >= 0.0)
    if tight_margin_mask.any():
        # Linear interpolation of partial dilution
        closeness = (6.0 - gap[tight_margin_mask]) / 6.0  # 0 at gap=6, 1 at gap=0
        adjustment[tight_margin_mask] = 1.0 - dilution * closeness * penalty_strength * 0.5

    return adjustment


def run_valuation_simulation(
    t_sci_samples: np.ndarray,
    t_fin_samples: np.ndarray,
    pos_samples: np.ndarray,
    inputs: ValuationInput,
    rng: np.random.Generator,
) -> ValuationResult:
    """
    Compute the full Monte Carlo valuation distribution.

    Returns ValuationResult with distribution statistics and rNPV decomposition.
    """
    n = len(t_sci_samples)

    # Sample binary success for each simulation
    success = rng.binomial(1, pos_samples).astype(bool)

    # Funding status
    funded = t_sci_samples < t_fin_samples

    # Discount factor: r = annual_discount_rate / 12 per month
    monthly_rate = inputs.annual_discount_rate / 12.0
    discount = 1.0 / ((1.0 + monthly_rate) ** t_sci_samples)

    # Financing adjustment
    fin_adj = _financing_adjustment(
        t_sci_samples,
        t_fin_samples,
        funded,
        inputs.expected_dilution_if_refinanced,
        inputs.financing_penalty_strength,
    )

    # Scenario values
    values = np.where(
        success,
        inputs.asset_value_success * discount * fin_adj,
        float(inputs.downside_value),
    )

    # Technical-risk-only rNPV: no financing risk, just PoS and time discount
    mean_pos = float(np.mean(pos_samples))
    mean_discount = float(np.mean(discount))
    technical_rnpv = inputs.asset_value_success * mean_pos * mean_discount

    # Financing-adjusted rNPV = mean simulated value
    financing_rnpv = float(np.mean(values))
    financing_discount_value = technical_rnpv - financing_rnpv

    # High-upside threshold: 50% of asset value
    high_upside_threshold = inputs.asset_value_success * 0.50
    prob_high_upside = float(np.mean(values >= high_upside_threshold))
    prob_downside = float(np.mean(values <= inputs.downside_value * 1.01))

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
        model_assumptions=[
            "Asset value on success is a user-supplied assumption, not a model output.",
            "Discount rate is user-supplied; default 12% reflects typical biotech WACC.",
            "Financing adjustment penalises scenarios where company must raise before catalyst.",
            "Dilution is modelled as a flat fractional penalty on value, not as share-count modelling.",
            "rNPV outputs are highly sensitive to asset_value_success and discount rate.",
            "This is not investment advice.",
        ],
    )
