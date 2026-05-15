"""
Tests for explicit cash-balance path simulation.
"""

import numpy as np
import pytest

from app.engines.cash_path import simulate_cash_path
from pydantic import ValidationError

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

    def test_month_zero_financing_applies_before_first_burn(self):
        result = simulate_cash_path(
            CashPathInput(
                starting_cash=1_000_000,
                monthly_burn=2_000_000,
                horizon_months=2,
                financing_events=[
                    FinancingEventInput(month=0, kind="clean_refi", gross_proceeds=5_000_000)
                ],
            )
        )

        assert result.monthly_balances[0].starting_cash == pytest.approx(6_000_000)
        assert result.cash_exhaustion_month is None

    def test_zero_starting_cash_with_month_zero_financing_does_not_immediately_exhaust(self):
        result = simulate_cash_path(
            CashPathInput(
                starting_cash=0,
                monthly_burn=1_000_000,
                horizon_months=3,
                financing_events=[
                    FinancingEventInput(month=0, kind="clean_refi", gross_proceeds=5_000_000)
                ],
            )
        )

        assert result.cash_exhaustion_month is None
        assert result.final_state == "horizon_reached"

    def test_negative_financing_month_fails_validation(self):
        with pytest.raises(ValidationError):
            FinancingEventInput(month=-1, kind="clean_refi", gross_proceeds=1_000_000)

    def test_deficit_fields_preserve_shortfall_while_ending_cash_is_non_negative(self):
        result = simulate_cash_path(
            CashPathInput(
                starting_cash=1_000_000,
                monthly_burn=3_000_000,
                horizon_months=3,
            )
        )

        assert result.cash_exhaustion_month == 1
        assert result.ending_cash == 0
        assert result.cash_shortfall_at_exhaustion == pytest.approx(2_000_000)
        assert result.maximum_cash_deficit == pytest.approx(8_000_000)
        assert result.capital_needed_to_survive_horizon == pytest.approx(8_000_000)

    def test_capital_needed_to_reach_catalyst_uses_supplied_catalyst_month(self):
        result = simulate_cash_path(
            CashPathInput(
                starting_cash=1_000_000,
                monthly_burn=3_000_000,
                horizon_months=3,
                catalyst_month=2,
            )
        )

        assert result.capital_needed_to_reach_catalyst == pytest.approx(5_000_000)
