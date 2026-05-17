"""
Benchmark 7 — Event-taxonomy scenario simulation.

Tests three financing scenarios to verify that the event-taxonomy
probability fields behave correctly. No network calls, no API key required.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.monte_carlo import run_full_audit
from app.models.schemas import (
    AuditRequest,
    ClinicalCatalystInput,
    CompanyFinancialInput,
    DisclosureInput,
    FinancingEventInput,
    SimulationConfig,
    SuccessProbabilityInput,
    ValuationInput,
)


def _base_request(planned_events=None) -> AuditRequest:
    return AuditRequest(
        financial=CompanyFinancialInput(
            company_name="TaxoBio",
            ticker="TAXO",
            cash_on_hand=80_000_000,
            marketable_securities=5_000_000,
            quarterly_operating_cash_burn=12_000_000,
            market_cap=400_000_000,
            debt=0,
            going_concern_flag=False,
            biotech_market_condition_score=6.0,
            planned_financing_events=planned_events or [],
        ),
        clinical=ClinicalCatalystInput(
            asset_name="TAXO-101",
            indication="Oncology",
            trial_phase="phase_2",
            trial_status="recruiting",
            stated_months_to_catalyst=18,
            enrollment_target=100,
            enrollment_completed=50,
            enrollment_rate_per_month=7,
            number_of_sites=12,
            catalyst_type="primary_readout",
        ),
        success_probability=SuccessProbabilityInput(
            trial_phase="phase_2",
            disease_area="oncology",
            positive_signals=[],
            negative_signals=[],
        ),
        valuation=ValuationInput(asset_value_success=500_000_000),
        disclosure=DisclosureInput(
            company_narrative_distribution={
                "runway_strength": 0.6,
                "clinical_timeline_confidence": 0.6,
                "dilution_risk": 0.4,
            },
            structured_audit_distribution={
                "runway_strength": 0.5,
                "clinical_timeline_confidence": 0.5,
                "dilution_risk": 0.5,
            },
        ),
        simulation=SimulationConfig(
            n_simulations=300,
            random_seed=7,
            monthly_horizon=48,
            use_llm_source_review=False,
        ),
    )


def scenario_a_debt_or_royalty() -> None:
    """Scenario A: debt/royalty financing event before catalyst."""
    print("Scenario A: debt_or_royalty before catalyst")
    events = [FinancingEventInput(month=6, kind="debt_or_royalty", gross_proceeds=30_000_000)]
    result = run_full_audit(_base_request(events))
    val = result.valuation

    print(f"  p_debt_or_royalty_before_catalyst:      {val.p_debt_or_royalty_before_catalyst:.4f}")
    print(f"  p_nondilutive_financing_before_catalyst:{val.p_nondilutive_financing_before_catalyst:.4f}")
    print(f"  p_any_financing_event_before_catalyst:  {val.p_any_financing_event_before_catalyst:.4f}")
    print(f"  p_financing_pressure_before_catalyst:   {val.p_financing_pressure_before_catalyst:.4f}")

    assert val.p_debt_or_royalty_before_catalyst == 1.0, (
        f"Expected p_debt_or_royalty==1.0 for planned debt event before catalyst, "
        f"got {val.p_debt_or_royalty_before_catalyst}"
    )
    assert val.p_nondilutive_financing_before_catalyst >= val.p_debt_or_royalty_before_catalyst - 1e-6, (
        "p_nondilutive must include debt/royalty"
    )
    assert val.p_any_financing_event_before_catalyst >= val.p_debt_or_royalty_before_catalyst - 1e-6, (
        "p_any must include debt/royalty"
    )
    # Nondistressed debt/royalty alone must NOT drive p_financing_pressure to 1.0
    assert val.p_financing_pressure_before_catalyst < 1.0, (
        "p_financing_pressure should not be 1.0 from debt/royalty alone (it is not distressed)"
    )
    print("  PASSED")


def scenario_b_distressed_refi() -> None:
    """Scenario B: distressed refinancing before catalyst."""
    print("Scenario B: distressed_refi before catalyst")
    events = [FinancingEventInput(month=6, kind="distressed_refi", gross_proceeds=20_000_000)]
    result = run_full_audit(_base_request(events))
    val = result.valuation

    print(f"  p_distressed_refinancing_before_catalyst:{val.p_distressed_refinancing_before_catalyst:.4f}")
    print(f"  p_dilutive_financing_before_catalyst:    {val.p_dilutive_financing_before_catalyst:.4f}")
    print(f"  p_financing_pressure_before_catalyst:    {val.p_financing_pressure_before_catalyst:.4f}")
    print(f"  p_any_financing_event_before_catalyst:   {val.p_any_financing_event_before_catalyst:.4f}")

    assert val.p_distressed_refinancing_before_catalyst == 1.0, (
        f"Expected p_distressed==1.0, got {val.p_distressed_refinancing_before_catalyst}"
    )
    assert val.p_dilutive_financing_before_catalyst >= val.p_distressed_refinancing_before_catalyst - 1e-6, (
        "p_dilutive must include distressed"
    )
    assert val.p_financing_pressure_before_catalyst >= val.p_distressed_refinancing_before_catalyst - 1e-6, (
        "p_financing_pressure must include distressed"
    )
    print("  PASSED")


def scenario_c_partnership() -> None:
    """Scenario C: partnership before catalyst."""
    print("Scenario C: partnership before catalyst")
    events = [FinancingEventInput(month=6, kind="partnership", gross_proceeds=50_000_000)]
    result = run_full_audit(_base_request(events))
    val = result.valuation

    print(f"  p_partnership_before_catalyst:          {val.p_partnership_before_catalyst:.4f}")
    print(f"  p_nondilutive_financing_before_catalyst:{val.p_nondilutive_financing_before_catalyst:.4f}")
    print(f"  p_financing_pressure_before_catalyst:   {val.p_financing_pressure_before_catalyst:.4f}")
    print(f"  p_any_financing_event_before_catalyst:  {val.p_any_financing_event_before_catalyst:.4f}")

    assert val.p_partnership_before_catalyst == 1.0, (
        f"Expected p_partnership==1.0, got {val.p_partnership_before_catalyst}"
    )
    assert val.p_nondilutive_financing_before_catalyst >= val.p_partnership_before_catalyst - 1e-6, (
        "p_nondilutive must include partnership"
    )
    # Partnership is nondistressed; should not automatically drive financing pressure to 1.0
    assert val.p_financing_pressure_before_catalyst < 1.0, (
        "p_financing_pressure should not be 1.0 from partnership alone"
    )
    print("  PASSED")


def main() -> None:
    print("Running event taxonomy scenarios (n=300 each, no LLM)...\n")
    scenario_a_debt_or_royalty()
    print()
    scenario_b_distressed_refi()
    print()
    scenario_c_partnership()
    print("\nAll scenario assertions passed.")
    print("smoke_event_taxonomy.py: OK")


if __name__ == "__main__":
    main()
