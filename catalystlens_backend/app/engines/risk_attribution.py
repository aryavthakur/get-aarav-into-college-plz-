"""
Shapley-based risk attribution for CatalystLens.

Decomposes the total cashout probability and EV uncertainty into additive
contributions from each major risk driver using a Shapley-value approximation.

Shapley value for risk driver i:
  phi_i = E_S[v(S ∪ {i}) - v(S)] over all subsets S not containing i

Where v(S) is the "risk removed" by resolving the uncertainty in set S.

For tractability, we use Owen's "random-order" estimator (sum over M random
permutations of risk drivers, measuring marginal contribution when i is added).

The six primary risk drivers modelled are:
  1. cash_burn         — monthly burn rate uncertainty
  2. catalyst_timing   — milestone timing uncertainty (Gamma distribution)
  3. pos_uncertainty   — technical PoS uncertainty (Beta posterior variance)
  4. financing_market  — biotech market condition (Cox LP contribution)
  5. burn_acceleration — quarterly burn trajectory (burn_acceleration coeff)
  6. dilution_risk     — dilution if company must refinance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class ShapleyComponent:
    driver: str
    description: str
    cashout_prob_shapley: float
    ev_shapley: float
    rank: int


@dataclass
class RiskAttributionResult:
    components: List[ShapleyComponent]
    total_cashout_prob: float
    total_ev: float
    explained_cashout_prob: float
    explained_ev: float
    methodology_note: str


def _build_simulator(
    base_cashout: float,
    base_ev: float,
    sensitivity_rows: list,
) -> Callable[[frozenset], Tuple[float, float]]:
    """
    Build a coalition value function from sensitivity analysis outputs.

    When a subset S of drivers has its uncertainty resolved (set to base level),
    the residual variance is estimated by excluding those drivers' sensitivity ranges.
    This is an approximation: true Shapley requires re-running the simulation for each
    coalition, which is expensive. Instead we use the range from sensitivity rows as
    a proxy for each driver's individual contribution to uncertainty.

    v(S) = fraction of total variance explained by resolving S.
    """
    driver_ranges: Dict[str, Tuple[float, float]] = {}
    for row in sensitivity_rows:
        cp_range = abs(row.high_cashout_prob - row.low_cashout_prob)
        ev_range = abs(row.high_expected_value - row.low_expected_value)
        driver_ranges[row.variable] = (cp_range, ev_range)

    total_cp_range = sum(r[0] for r in driver_ranges.values()) or 1.0
    total_ev_range = sum(r[1] for r in driver_ranges.values()) or 1.0

    def value_fn(coalition: frozenset) -> Tuple[float, float]:
        cp_explained = sum(
            driver_ranges[d][0] for d in coalition if d in driver_ranges
        )
        ev_explained = sum(
            driver_ranges[d][1] for d in coalition if d in driver_ranges
        )
        return cp_explained / total_cp_range, ev_explained / total_ev_range

    return value_fn


_DRIVER_MAPPING: Dict[str, Tuple[str, str]] = {
    "monthly_burn": ("cash_burn", "Monthly operating cash burn rate"),
    "stated_months_to_catalyst": ("catalyst_timing", "Management-stated catalyst timeline"),
    "enrollment_rate": ("enrollment_speed", "Trial enrollment rate"),
    "posterior_pos": ("pos_uncertainty", "Bayesian posterior probability of success"),
    "annual_discount_rate": ("discount_rate", "Risk-adjusted discount rate"),
    "dilution_if_refinanced": ("dilution_risk", "Dilution if company must refinance"),
    "asset_value_success": ("asset_value", "Asset value on technical success"),
    "market_condition_score": ("financing_market", "Biotech financing market conditions"),
}


def compute_shapley_attribution(
    sensitivity_rows: list,
    total_cashout_prob: float,
    total_ev: float,
    n_permutations: int = 64,
    rng: np.random.Generator | None = None,
) -> RiskAttributionResult:
    """
    Approximate Shapley values using random-order Owen estimator.

    sensitivity_rows: list of SensitivityPoint objects
    n_permutations: number of random permutations to average over
    """
    if rng is None:
        rng = np.random.default_rng(0)

    drivers = [row.variable for row in sensitivity_rows if row.variable in _DRIVER_MAPPING]
    if not drivers:
        return RiskAttributionResult(
            components=[],
            total_cashout_prob=total_cashout_prob,
            total_ev=total_ev,
            explained_cashout_prob=0.0,
            explained_ev=0.0,
            methodology_note="No sensitivity rows available for Shapley decomposition.",
        )

    value_fn = _build_simulator(total_cashout_prob, total_ev, sensitivity_rows)

    # Marginal contribution accumulator
    phi_cp: Dict[str, float] = {d: 0.0 for d in drivers}
    phi_ev: Dict[str, float] = {d: 0.0 for d in drivers}

    for _ in range(n_permutations):
        perm = list(rng.permutation(drivers))
        coalition: set[str] = set()
        for d in perm:
            v_before = value_fn(frozenset(coalition))
            coalition.add(d)
            v_after = value_fn(frozenset(coalition))
            phi_cp[d] += (v_after[0] - v_before[0]) / n_permutations
            phi_ev[d] += (v_after[1] - v_before[1]) / n_permutations

    # Build driver_ranges for absolute contribution scaling
    driver_ranges: Dict[str, Tuple[float, float]] = {}
    for row in sensitivity_rows:
        if row.variable in _DRIVER_MAPPING:
            driver_ranges[row.variable] = (
                abs(row.high_cashout_prob - row.low_cashout_prob),
                abs(row.high_expected_value - row.low_expected_value),
            )

    total_cp_range = sum(r[0] for r in driver_ranges.values()) or 1.0
    total_ev_range = sum(r[1] for r in driver_ranges.values()) or 1.0

    components: List[ShapleyComponent] = []
    for d in drivers:
        short_name, description = _DRIVER_MAPPING[d]
        # Convert fractional Shapley to absolute units
        cp_abs = phi_cp[d] * total_cp_range
        ev_abs = phi_ev[d] * total_ev_range
        components.append(ShapleyComponent(
            driver=d,
            description=description,
            cashout_prob_shapley=round(cp_abs, 4),
            ev_shapley=round(ev_abs, 2),
            rank=0,  # filled below
        ))

    # Rank by absolute cashout Shapley value
    components.sort(key=lambda c: abs(c.cashout_prob_shapley), reverse=True)
    for i, c in enumerate(components):
        c.rank = i + 1

    explained_cp = sum(c.cashout_prob_shapley for c in components)
    explained_ev = sum(c.ev_shapley for c in components)

    return RiskAttributionResult(
        components=components,
        total_cashout_prob=total_cashout_prob,
        total_ev=total_ev,
        explained_cashout_prob=round(explained_cp, 4),
        explained_ev=round(explained_ev, 2),
        methodology_note=(
            "Shapley-style sensitivity attribution using Owen random-order estimator "
            f"({n_permutations} permutations). Driver contributions are proportional "
            "to sensitivity analysis ranges — a sensitivity-based approximation, not a "
            "true Shapley decomposition (which would require re-running the full simulation "
            "for each coalition subset)."
        ),
    )
