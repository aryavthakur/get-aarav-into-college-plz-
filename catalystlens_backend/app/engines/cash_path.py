"""
Explicit monthly cash-balance path simulation.

This module separates literal cash exhaustion from hazard-model financing
events. It is intentionally mechanical: cash changes only through burn and
declared capital inflows, making impossible states easier to detect.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from app.models.schemas import CashPathInput, CashPathMonth, CashPathResult


def _sample_monthly_burn(
    monthly_burn: float,
    monthly_burn_volatility: float,
    rng: np.random.Generator | None,
) -> float:
    if monthly_burn_volatility <= 0:
        return monthly_burn

    generator = rng if rng is not None else np.random.default_rng()
    # Mean-corrected lognormal so expected burn remains close to monthly_burn.
    sigma = monthly_burn_volatility
    multiplier = generator.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma)
    return monthly_burn * float(multiplier)


def simulate_cash_path(
    inputs: CashPathInput,
    rng: np.random.Generator | None = None,
) -> CashPathResult:
    """Simulate monthly cash balances until cash exhaustion or horizon."""
    events_by_month: dict[int, float] = defaultdict(float)
    for event in inputs.financing_events:
        events_by_month[event.month] += event.net_proceeds

    cash = float(inputs.starting_cash) + float(events_by_month.get(0, 0.0))
    total_raised = float(events_by_month.get(0, 0.0))

    if cash <= 0:
        return CashPathResult(
            cash_exhaustion_month=0,
            final_state="cash_exhaustion",
            minimum_cash_balance=0.0,
            ending_cash=0.0,
            total_burn=0.0,
            total_capital_raised=round(total_raised, 2),
            cash_shortfall_at_exhaustion=round(abs(cash), 2),
            maximum_cash_deficit=round(abs(cash), 2),
            capital_needed_to_survive_horizon=round(abs(cash), 2),
            capital_needed_to_reach_catalyst=round(abs(cash), 2) if inputs.catalyst_month is not None else None,
            monthly_balances=[],
        )

    raw_cash = cash
    min_raw_cash = raw_cash
    total_burn = 0.0
    balances: list[CashPathMonth] = []
    cashout_month: int | None = None
    final_state = "horizon_reached"
    first_shortfall = 0.0
    max_deficit_through_catalyst = 0.0

    for month in range(1, inputs.horizon_months + 1):
        starting_cash = max(raw_cash, 0.0)
        inflow = float(events_by_month.get(month, 0.0))
        burn = _sample_monthly_burn(inputs.monthly_burn, inputs.monthly_burn_volatility, rng)
        raw_cash = raw_cash + inflow - burn
        total_burn += burn
        total_raised += inflow
        min_raw_cash = min(min_raw_cash, raw_cash)
        current_deficit = max(0.0, -raw_cash)
        if inputs.catalyst_month is not None and month <= inputs.catalyst_month:
            max_deficit_through_catalyst = max(max_deficit_through_catalyst, current_deficit)

        state = "continue"
        if raw_cash <= 0:
            if cashout_month is None:
                first_shortfall = abs(raw_cash)
                cashout_month = month
            final_state = "cash_exhaustion"
            state = "cash_exhaustion"

        balances.append(
            CashPathMonth(
                month=month,
                starting_cash=round(starting_cash, 2),
                sampled_burn=round(burn, 2),
                capital_inflow=round(inflow, 2),
                ending_cash=round(max(raw_cash, 0.0), 2),
                state=state,
            )
        )

    return CashPathResult(
        cash_exhaustion_month=cashout_month,
        final_state=final_state,
        minimum_cash_balance=round(max(0.0, min_raw_cash), 2),
        ending_cash=round(max(raw_cash, 0.0), 2),
        total_burn=round(total_burn, 2),
        total_capital_raised=round(total_raised, 2),
        cash_shortfall_at_exhaustion=round(first_shortfall, 2),
        maximum_cash_deficit=round(max(0.0, -min_raw_cash), 2),
        capital_needed_to_survive_horizon=round(max(0.0, -min_raw_cash), 2),
        capital_needed_to_reach_catalyst=(
            round(max_deficit_through_catalyst, 2) if inputs.catalyst_month is not None else None
        ),
        monthly_balances=balances,
    )
