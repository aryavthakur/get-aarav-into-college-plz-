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
from dataclasses import dataclass

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
) -> RealOptionsResult:
    """
    Monte Carlo real-options valuation using simulated milestone times.

    For each simulation path:
    1. Draw a GBM path to T_sci: V_T = V_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    2. At T_sci, the company chooses whether to exercise (invest K) given clinical outcome
    3. Value of path = PoS * max(V_T - K, 0) * e^(-r*T)
       where PoS is the per-path probability of technical success
    4. Optionality: on failure paths, value = max(0, residual) — no forced loss from K

    The real-options premium vs rNPV captures:
    - Abandonment option: company can walk away if V_T < K (no negative payoff forced)
    - Volatility value: higher sigma → fatter tails → higher option value
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
    # Success probability modulates whether the clinical trial succeeds;
    # given success, the option to invest in the asset is exercised only if V_T > K
    intrinsic = np.maximum(V_T - K, 0.0)
    discount = np.exp(-r * T_years)
    path_values = pos_samples * intrinsic * discount

    # Abandonment value: on paths where trial succeeds but V_T < K, walk away (value = 0)
    # This is already captured by max(V_T - K, 0) above.
    # Abandonment savings vs a naive "always invest" strategy:
    forced_invest = pos_samples * (V_T - K) * discount  # payoff if K always paid
    abandonment_savings = float(np.mean(np.maximum(0.0, forced_invest - path_values)))

    rov_vals = path_values
    rnpv_static = float(np.mean(pos_samples)) * V0 * float(np.mean(discount))

    return RealOptionsResult(
        rov_mean=round(float(np.mean(rov_vals)), 2),
        rov_median=round(float(np.median(rov_vals)), 2),
        rov_p5=round(float(np.percentile(rov_vals, 5)), 2),
        rov_p95=round(float(np.percentile(rov_vals, 95)), 2),
        rnpv_static=round(rnpv_static, 2),
        real_options_premium=round(float(np.mean(rov_vals)) - max(rnpv_static, 0.0), 2),
        real_options_premium_pct=round(
            (float(np.mean(rov_vals)) - max(rnpv_static, 0.0)) / max(abs(rnpv_static), 1.0) * 100.0,
            2,
        ),
        abandonment_value=round(abandonment_savings, 2),
        model_assumptions=[
            f"GBM underlying: sigma={sigma:.0%}, r={r:.0%}.",
            "Exercise cost K defaults to 0 (pure upside option); set to next-phase investment for compound option.",
            "Real-options premium = ROV - max(rNPV, 0); positive when volatility adds value beyond rNPV.",
            "Technical PoS applied as probability weight, not binary event, to smooth the payoff surface.",
        ],
    )
