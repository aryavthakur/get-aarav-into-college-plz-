"""
Capital-to-Catalyst Gap Engine.

Core question: P(T_sci < T_fin)

Where:
  T_sci = sampled time to scientific milestone (from Gamma distribution)
  T_fin = sampled time to financial failure / capital exhaustion (from Cox-Weibull)

The gap = T_fin - T_sci
  Positive gap → company is funded beyond the catalyst (good)
  Negative gap → company runs out of money before reaching the catalyst (bad)

This is the central quantitative output of the CatalystLens system.
"""

from __future__ import annotations

import numpy as np

from app.core.config import CatalystLensConfig, get_default_config
from app.models.schemas import CapitalToCatalystResult


_RISK_LABELS = {
    "Low Risk":      "The company appears likely to reach the modeled catalyst window before cash exhaustion under current assumptions.",
    "Moderate Risk": "There is a meaningful probability of capital exhaustion before the modeled catalyst. Refinancing risk warrants monitoring.",
    "High Risk":     "The modeled probability of cash exhaustion before the catalyst is elevated. The investment case is materially dependent on the company's ability to raise additional capital.",
    "Critical Risk": "There is a high modeled probability of capital exhaustion before the stated catalyst. Financing risk dominates the risk profile.",
}


def classify_capital_risk(
    prob_cashout: float,
    config: CatalystLensConfig | None = None,
) -> str:
    """Classify capital-to-catalyst risk based on P(cashout before catalyst)."""
    if config is None:
        config = get_default_config()
    t = config.risk_thresholds
    if prob_cashout < t.low_cashout_max:
        return "Low Risk"
    if prob_cashout < t.moderate_cashout_max:
        return "Moderate Risk"
    if prob_cashout < t.high_cashout_max:
        return "High Risk"
    return "Critical Risk"


def run_capital_to_catalyst_analysis(
    t_sci_samples: np.ndarray,
    t_fin_samples: np.ndarray,
    config: CatalystLensConfig | None = None,
) -> CapitalToCatalystResult:
    """
    Compute gap statistics from Monte Carlo samples.

    t_sci_samples: array of scientific milestone times (months)
    t_fin_samples: array of financial failure times (months)

    Both arrays must have the same length (number of simulations).
    """
    if config is None:
        config = get_default_config()

    n = len(t_sci_samples)
    if len(t_fin_samples) != n:
        raise ValueError("Sample arrays must have equal length")

    # Gap: positive = funded beyond catalyst, negative = cashout before catalyst
    gap = t_fin_samples - t_sci_samples

    # Core probabilities
    prob_cashout = float(np.mean(t_fin_samples < t_sci_samples))
    prob_reaches = 1.0 - prob_cashout

    # Gap distribution statistics
    median_gap = float(np.median(gap))
    p5_gap = float(np.percentile(gap, 5))
    p95_gap = float(np.percentile(gap, 95))
    median_t_fin = float(np.median(t_fin_samples))
    median_t_sci = float(np.median(t_sci_samples))

    risk_class = classify_capital_risk(prob_cashout, config)
    interpretation = (
        f"There is a modeled probability of {prob_cashout:.1%} that the company's "
        f"capital is exhausted before the scientific catalyst is reached. "
        + _RISK_LABELS[risk_class]
    )

    return CapitalToCatalystResult(
        probability_reaches_catalyst=round(prob_reaches, 4),
        probability_cashout_before_catalyst=round(prob_cashout, 4),
        median_gap_months=round(median_gap, 2),
        p5_gap_months=round(p5_gap, 2),
        p95_gap_months=round(p95_gap, 2),
        median_financial_failure_time=round(median_t_fin, 2),
        median_catalyst_time=round(median_t_sci, 2),
        risk_classification=risk_class,
        interpretation=interpretation,
    )
