"""
Tests for explicit cash-balance path simulation.
"""

import numpy as np
import pytest

from app.engines.cash_path import simulate_cash_path
from app.models.schemas import CashPathInput, FinancingEventInput


class TestCashPathSimulation:
    def test_zero_liquidity_exhausts_immediately(self):
        result = simulate_cash_path(
            CashPathInput(
                starting_cash=0,
                monthly_burn=1_000_000,
                horizon_months=12,
            )
        )

        assert result.cash_exhaustion_month == 0
        assert result.final_state == "cash_exhaustion"
        assert result.minimum_cash_balance == 0

    def test_cash_exhaustion_month_matches_mechanical_burn(self):
        result = simulate_cash_path(
            CashPathInput(
                starting_cash=10_000_000,
                monthly_burn=2_500_000,
                horizon_months=12,
            )
        )

        assert result.cash_exhaustion_month == 4
        assert result.final_state == "cash_exhaustion"
        assert result.monthly_balances[3].ending_cash == pytest.approx(0)

    def test_clean_raise_extends_cash_path(self):
        no_raise = simulate_cash_path(
            CashPathInput(
                starting_cash=10_000_000,
                monthly_burn=2_500_000,
                horizon_months=12,
            )
        )
        with_raise = simulate_cash_path(
            CashPathInput(
                starting_cash=10_000_000,
                monthly_burn=2_500_000,
                horizon_months=12,
                financing_events=[
                    FinancingEventInput(month=3, kind="clean_refi", gross_proceeds=10_000_000)
                ],
            )
        )

        assert no_raise.cash_exhaustion_month == 4
        assert with_raise.cash_exhaustion_month == 8
        assert with_raise.total_capital_raised == pytest.approx(10_000_000)

    def test_burn_volatility_is_reproducible_with_seed(self):
        base = CashPathInput(
            starting_cash=20_000_000,
            monthly_burn=2_000_000,
            horizon_months=12,
            monthly_burn_volatility=0.15,
        )

        r1 = simulate_cash_path(base, rng=np.random.default_rng(123))
        r2 = simulate_cash_path(base, rng=np.random.default_rng(123))
        r3 = simulate_cash_path(base, rng=np.random.default_rng(321))

        assert [m.sampled_burn for m in r1.monthly_balances] == [
            m.sampled_burn for m in r2.monthly_balances
        ]
        assert [m.sampled_burn for m in r1.monthly_balances] != [
            m.sampled_burn for m in r3.monthly_balances
        ]
