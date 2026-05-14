"""
Solvency Engine — Financial Clock.

Implements a Cox Proportional Hazards-style model on a Weibull baseline to
estimate the probability that a biotech company exhausts its capital over time.

Mathematical model:
  h(t|X) = h0(t) * exp(LP)   where LP = sum(beta_i * f_i(X))
  S(t|X) = S0(t)^exp(LP)     where S0(t) = exp(-( lambda*t )^k)  [Weibull]

Sampling (inverse CDF):
  T | X ~ Weibull(shape=k, scale=1 / (lambda * exp(LP/k)))

IMPORTANT: Cox coefficients are UNTRAINED MVP ASSUMPTIONS.
They encode reasonable directional intuitions about biotech financing risk
but have not been fit to historical outcome data.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from app.core.config import CatalystLensConfig, CoxCoefficients, WeibullParams, get_default_config
from app.models.schemas import CompanyFinancialInput, SolvencyResult, SurvivalPoint


# ---------------------------------------------------------------------------
# Core financial computations
# ---------------------------------------------------------------------------

def compute_total_liquidity(financial: CompanyFinancialInput) -> float:
    """Total modeled liquidity = cash + marketable securities."""
    return max(0.0, financial.cash_on_hand + financial.marketable_securities)


def calculate_monthly_burn(quarterly_burn: float) -> float:
    """Convert quarterly operating cash burn to monthly burn."""
    return quarterly_burn / 3.0


def calculate_simple_runway_months(total_liquidity: float, monthly_burn: float) -> float:
    """
    Simple (static) runway = total_liquidity / monthly_burn.

    This is the management-reported style runway estimate.
    It does not account for financing risk or burn trajectory.
    Returns infinity if burn is zero.
    """
    if monthly_burn <= 0:
        return float("inf")
    return total_liquidity / monthly_burn


# ---------------------------------------------------------------------------
# Cox linear predictor
# ---------------------------------------------------------------------------

def _compute_linear_predictor(
    monthly_burn: float,
    total_liquidity: float,
    burn_acceleration: float,
    market_cap: float,
    debt: float,
    going_concern_flag: bool,
    recent_financing_flag: bool,
    months_since_last_raise: float,
    biotech_market_condition_score: float,
    pipeline_concentration_score: float,
    trial_phase: str,
    coeff: CoxCoefficients,
    phase_risk_map: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Cox linear predictor LP = sum(beta_i * f_i(X)).

    Returns (LP, dict of per-covariate contributions).
    LP > 0 → higher-than-baseline risk.
    LP < 0 → lower-than-baseline risk.
    """
    contributions: Dict[str, float] = {}

    # --- Cash runway (centered at reference) ---
    runway = calculate_simple_runway_months(total_liquidity, monthly_burn)
    # Positive when runway < reference (higher risk), negative when runway > reference
    contributions["cash_runway"] = coeff.cash_runway_per_month_vs_reference * (
        coeff.reference_runway_months - runway
    )

    # --- Burn acceleration ---
    contributions["burn_acceleration"] = coeff.burn_acceleration * max(0.0, burn_acceleration)

    # --- Going-concern flag ---
    contributions["going_concern"] = coeff.going_concern_flag * (1.0 if going_concern_flag else 0.0)

    # --- Debt-to-cash ratio ---
    if total_liquidity > 0:
        debt_ratio = debt / total_liquidity
    else:
        debt_ratio = 5.0
    contributions["debt_to_cash"] = coeff.debt_to_cash_ratio * debt_ratio

    # --- Market-cap to cash ratio (log-centered at log(3)) ---
    if total_liquidity > 0 and market_cap > 0:
        log_mc = math.log(market_cap / total_liquidity)
        contributions["market_cap_to_cash"] = coeff.log_market_cap_to_cash_centered * (
            log_mc - math.log(3.0)
        )
    else:
        contributions["market_cap_to_cash"] = abs(coeff.log_market_cap_to_cash_centered) * 2.0

    # --- Months since last raise (centered at reference) ---
    contributions["months_since_raise"] = coeff.months_since_last_raise_vs_reference * (
        months_since_last_raise - coeff.reference_months_since_raise
    )

    # --- Recent financing flag ---
    contributions["recent_financing"] = coeff.recent_financing_flag * (
        1.0 if recent_financing_flag else 0.0
    )

    # --- Biotech market condition (centered at 5) ---
    contributions["market_condition"] = coeff.biotech_market_condition_per_point * (
        biotech_market_condition_score - 5.0
    )

    # --- Pipeline concentration ---
    contributions["pipeline_concentration"] = coeff.pipeline_concentration * pipeline_concentration_score

    # --- Trial phase risk ---
    phase_risk = phase_risk_map.get(trial_phase, 0.4)
    contributions["trial_phase"] = coeff.trial_phase_risk * phase_risk

    lp = sum(contributions.values())
    return lp, contributions


def calculate_risk_multiplier(lp: float) -> float:
    """
    Risk multiplier = exp(LP).

    > 1 → higher-than-baseline hazard.
    < 1 → lower-than-baseline hazard.
    Clamped to [0.05, 20.0] for numerical stability.
    """
    return float(np.clip(math.exp(lp), 0.05, 20.0))


# ---------------------------------------------------------------------------
# Survival functions
# ---------------------------------------------------------------------------

def baseline_survival(t: float, params: WeibullParams) -> float:
    """
    Baseline Weibull survival: S0(t) = exp(-(lambda * t)^k).

    t is in months. Returns probability in [0, 1].
    """
    if t <= 0:
        return 1.0
    return math.exp(-((params.lambda_ * t) ** params.k))


def survival_probability(t: float, risk_multiplier: float, params: WeibullParams) -> float:
    """
    Cox-adjusted survival: S(t|X) = S0(t)^risk_multiplier.

    Equivalent to: exp(-(lambda*t)^k * risk_multiplier)
    """
    if t <= 0:
        return 1.0
    s0 = baseline_survival(t, params)
    if s0 <= 0:
        return 0.0
    return float(s0 ** risk_multiplier)


def _median_failure_time(risk_multiplier: float, params: WeibullParams) -> float:
    """
    Closed-form median of the Cox-adjusted Weibull survival.

    S(t*) = 0.5  →  t* = log(2)^(1/k) / (lambda * risk_multiplier^(1/k))
    """
    return (math.log(2) ** (1.0 / params.k)) / (
        params.lambda_ * (risk_multiplier ** (1.0 / params.k))
    )


def compute_survival_curve(
    risk_multiplier: float,
    params: WeibullParams,
    monthly_horizon: int,
) -> List[SurvivalPoint]:
    """Return survival probability at each month from 1 to monthly_horizon."""
    curve = []
    for t in range(1, monthly_horizon + 1):
        sp = survival_probability(float(t), risk_multiplier, params)
        curve.append(SurvivalPoint(
            month=t,
            survival_probability=round(sp, 6),
            implied_cashout_risk=round(1.0 - sp, 6),
        ))
    return curve


# ---------------------------------------------------------------------------
# Monte Carlo sampling
# ---------------------------------------------------------------------------

def sample_financial_failure_time(
    rng: np.random.Generator,
    risk_multiplier: float,
    params: WeibullParams,
    n_samples: int,
) -> np.ndarray:
    """
    Sample financial failure times from the Cox-adjusted Weibull.

    Derivation (inverse-CDF method):
      S(T|X) = U  →  T = (-log U)^(1/k) / (lambda * exp(LP/k))
                        = (-log U)^(1/k) / (lambda * rm^(1/k))

    Returns array of positive failure-time samples in months.
    """
    u = rng.uniform(0.0, 1.0, size=n_samples)
    scale = 1.0 / (params.lambda_ * (risk_multiplier ** (1.0 / params.k)))
    t = (-np.log(u)) ** (1.0 / params.k) * scale
    return np.maximum(t, 0.1)


# ---------------------------------------------------------------------------
# High-level analysis runner
# ---------------------------------------------------------------------------

def run_solvency_analysis(
    financial: CompanyFinancialInput,
    burn_acceleration: float = 0.0,
    trial_phase: str = "phase_2",
    config: CatalystLensConfig | None = None,
    monthly_horizon: int = 48,
) -> SolvencyResult:
    """
    Run the full solvency (financial clock) analysis for one company.

    burn_acceleration should be supplied from the BurnRegimeEngine output.
    """
    if config is None:
        config = get_default_config()

    monthly_burn = calculate_monthly_burn(financial.quarterly_operating_cash_burn)
    total_liquidity = compute_total_liquidity(financial)
    simple_runway = calculate_simple_runway_months(total_liquidity, monthly_burn)

    lp, contributions = _compute_linear_predictor(
        monthly_burn=monthly_burn,
        total_liquidity=total_liquidity,
        burn_acceleration=burn_acceleration,
        market_cap=financial.market_cap,
        debt=financial.debt,
        going_concern_flag=financial.going_concern_flag,
        recent_financing_flag=financial.recent_financing_flag,
        months_since_last_raise=financial.months_since_last_raise,
        biotech_market_condition_score=financial.biotech_market_condition_score,
        pipeline_concentration_score=financial.pipeline_concentration_score,
        trial_phase=trial_phase,
        coeff=config.cox_coefficients,
        phase_risk_map=config.trial_phase_risk_map,
    )

    rm = calculate_risk_multiplier(lp)
    wp = config.weibull_params

    survival_curve = compute_survival_curve(rm, wp, monthly_horizon)
    median_ft = _median_failure_time(rm, wp)

    return SolvencyResult(
        monthly_burn=round(monthly_burn, 2),
        total_liquidity=round(total_liquidity, 2),
        simple_runway_months=round(simple_runway, 2),
        risk_multiplier=round(rm, 4),
        linear_predictor=round(lp, 4),
        covariate_contributions={k: round(v, 4) for k, v in contributions.items()},
        survival_curve=survival_curve,
        median_failure_time=round(median_ft, 2),
        p_survival_6m=round(survival_probability(6.0, rm, wp), 4),
        p_survival_12m=round(survival_probability(12.0, rm, wp), 4),
        p_survival_18m=round(survival_probability(18.0, rm, wp), 4),
        p_survival_24m=round(survival_probability(24.0, rm, wp), 4),
        model_assumptions=[
            "Cox coefficients are UNTRAINED MVP ASSUMPTIONS, not fit to historical data.",
            "Weibull baseline parameterized on general biotech financing intuition.",
            f"Baseline lambda={wp.lambda_}, k={wp.k} (configurable in config.py).",
            "Risk multiplier is clamped to [0.05, 20.0] for numerical stability.",
            "Model does not account for at-the-market equity programs or forward commitments.",
        ],
    )
