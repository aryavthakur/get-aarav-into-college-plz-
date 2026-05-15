"""
Real-options valuation for a single-asset biotech program.

Models the drug development program as a compound option:
  - At the catalyst milestone T_sci, the company holds the right (not obligation)
    to exercise into full development by paying K_phase3 (the next-stage investment).
  - The underlying asset value follows GBM: dV = r*V*dt + sigma*V*dW
  - The option value at t=0 is: ROV = E_Q[e^(-r*T) * max(V_T - K, 0) | T = T_sci]

Compared to the rNPV approach (which discounts at rate r and multiplies by PoS):
  - rNPV  = PoS * V_success * e^(-r*T)
  - ROV   = E[e^(-r*T) * max(V_T - K, 0)] — accounts for optionality (right to abandon)

The real-options premium is ROV - max(rNPV, 0): the value attributable to the
option structure beyond what a static rNPV calculation captures.

Reference: Black-Scholes with uncertain exercise time (integrated over T_sci distribution).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RealOptionsInput:
    """Inputs for real-options valuation of a single biotech program."""
    asset_value_success: float
    exercise_cost: float = 0.0
    asset_volatility: float = 0.60
    annual_discount_rate: float = 0.12
    pos_mean: float = 0.35


@dataclass
class RealOptionsResult:
    rov_mean: float
    rov_median: float
    rov_p5: float
    rov_p95: float
    rnpv_static: float
    real_options_premium: float
    real_options_premium_pct: float
    abandonment_value: float
    financing_adjusted_rov: float
    model_assumptions: list[str]


def _black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """European call option value via Black-Scholes."""
    if T <= 0 or S <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    if K <= 0:
        return S  # free option on positive underlying
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    from scipy.special import ndtr
    call = S * ndtr(d1) - K * math.exp(-r * T) * ndtr(d2)
    return float(max(call, 0.0))


def simulate_real_options_value(
    t_sci: np.ndarray,
    pos_samples: np.ndarray,
    inputs: RealOptionsInput,
    rng: np.random.Generator,
    p_funded: float = 1.0,
    p_clean_refi: float = 0.0,
    p_distressed_refi: float = 0.0,
    p_partnership: float = 0.0,
    dilution_clean: float = 0.25,
    dilution_distressed: float = 0.60,
    retained_partnership: float = 0.65,
) -> RealOptionsResult:
    """
    Monte Carlo real-options valuation using simulated milestone times.

    For each simulation path:
    1. Draw a GBM path to T_sci: V_T = V_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    2. At T_sci, the company chooses whether to exercise (invest K) given clinical outcome
    3. Value of path = PoS * max(V_T - K, 0) * e^(-r*T)
    4. Abandonment value: savings on paths where trial fails (V_T < K), i.e. not forced to invest

    financing_adjusted_rov conditions the mean ROV on the financing state probabilities,
    applying dilution / retained-economics adjustments per state.
    """
    n = len(t_sci)
    r = inputs.annual_discount_rate
    sigma = inputs.asset_volatility
    V0 = inputs.asset_value_success
    K = inputs.exercise_cost

    # Risk-neutral GBM terminal values
    T_years = t_sci / 12.0
    Z = rng.standard_normal(n)
    V_T = V0 * np.exp((r - 0.5 * sigma * sigma) * T_years + sigma * np.sqrt(np.maximum(T_years, 1e-6)) * Z)

    # Compound option: technical success AND positive intrinsic value
    intrinsic = np.maximum(V_T - K, 0.0)
    discount = np.exp(-r * T_years)
    path_values = pos_samples * intrinsic * discount

    # Abandonment value: savings from NOT being forced to invest on failure paths
    # Forced investment payoff (always invest): pos * (V_T - K) * discount
    # With abandonment (option): pos * max(V_T - K, 0) * discount
    # Abandonment savings per path = max(0, -(pos * (V_T - K) * discount))
    forced_invest = pos_samples * (V_T - K) * discount
    abandonment_savings = float(np.mean(np.maximum(0.0, -forced_invest)))

    rov_vals = path_values
    rov_mean = float(np.mean(rov_vals))
    rnpv_static = float(np.mean(pos_samples)) * V0 * float(np.mean(discount))

    # Financing-state-adjusted ROV
    # Negative states (discontinued / cash_exhaustion) contribute ~0
    p_negative = max(0.0, 1.0 - p_funded - p_clean_refi - p_distressed_refi - p_partnership)
    financing_adjusted_rov = (
        p_funded * rov_mean
        + p_clean_refi * rov_mean * (1.0 - dilution_clean)
        + p_distressed_refi * rov_mean * (1.0 - dilution_distressed)
        + p_partnership * rov_mean * retained_partnership
        # p_negative: near-zero contribution
    )

    return RealOptionsResult(
        rov_mean=round(rov_mean, 2),
        rov_median=round(float(np.median(rov_vals)), 2),
        rov_p5=round(float(np.percentile(rov_vals, 5)), 2),
        rov_p95=round(float(np.percentile(rov_vals, 95)), 2),
        rnpv_static=round(rnpv_static, 2),
        real_options_premium=round(rov_mean - max(rnpv_static, 0.0), 2),
        real_options_premium_pct=round(
            (rov_mean - max(rnpv_static, 0.0)) / max(abs(rnpv_static), 1.0) * 100.0,
            2,
        ),
        abandonment_value=round(abandonment_savings, 2),
        financing_adjusted_rov=round(financing_adjusted_rov, 2),
        model_assumptions=[
            f"GBM underlying: sigma={sigma:.0%}, r={r:.0%}.",
            "Exercise cost K defaults to 0 (pure upside option); set to next-phase investment for compound option.",
            "Real-options premium = ROV - max(rNPV, 0); positive when volatility adds value beyond rNPV.",
            "Technical PoS applied as probability weight, not binary event, to smooth the payoff surface.",
            "Abandonment value = E[max(0, -forced_invest)] = savings from walking away when V_T < K.",
            f"Financing-adjusted ROV applies state-specific economics: funded={p_funded:.1%}, clean_refi={p_clean_refi:.1%}, distressed={p_distressed_refi:.1%}, partnership={p_partnership:.1%}.",
        ],
    )
