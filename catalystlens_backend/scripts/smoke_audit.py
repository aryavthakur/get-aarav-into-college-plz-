"""
Benchmark 6 — Deterministic audit smoke simulation.

Runs a full CatalystLens audit with n_simulations=300 and
use_llm_source_review=False. Prints key modelled outputs and asserts
all probability invariants. No network calls, no API key required.
"""

import sys
import os

# Allow running from the catalystlens_backend directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.monte_carlo import run_full_audit
from app.models.schemas import (
    AuditRequest,
    ClinicalCatalystInput,
    CompanyFinancialInput,
    DisclosureInput,
    SimulationConfig,
    SuccessProbabilityInput,
    ValuationInput,
)


def _build_request() -> AuditRequest:
    return AuditRequest(
        financial=CompanyFinancialInput(
            company_name="SmokeBio",
            ticker="SMK",
            cash_on_hand=100_000_000,
            marketable_securities=10_000_000,
            quarterly_operating_cash_burn=15_000_000,
            market_cap=500_000_000,
            debt=0,
            going_concern_flag=False,
            biotech_market_condition_score=6.0,
        ),
        clinical=ClinicalCatalystInput(
            asset_name="SMK-101",
            indication="Oncology",
            trial_phase="phase_2",
            trial_status="recruiting",
            stated_months_to_catalyst=18,
            enrollment_target=120,
            enrollment_completed=60,
            enrollment_rate_per_month=8,
            number_of_sites=15,
            catalyst_type="primary_readout",
        ),
        success_probability=SuccessProbabilityInput(
            trial_phase="phase_2",
            disease_area="oncology",
            modality="small molecule",
            positive_signals=["validated_biomarker"],
            negative_signals=[],
        ),
        valuation=ValuationInput(asset_value_success=600_000_000),
        disclosure=DisclosureInput(
            company_narrative_distribution={
                "runway_strength": 0.7,
                "clinical_timeline_confidence": 0.7,
                "dilution_risk": 0.3,
                "trial_maturity": 0.6,
                "endpoint_strength": 0.6,
                "pipeline_diversification": 0.4,
            },
            structured_audit_distribution={
                "runway_strength": 0.6,
                "clinical_timeline_confidence": 0.6,
                "dilution_risk": 0.4,
                "trial_maturity": 0.5,
                "endpoint_strength": 0.5,
                "pipeline_diversification": 0.4,
            },
        ),
        simulation=SimulationConfig(
            n_simulations=300,
            random_seed=42,
            monthly_horizon=48,
            use_llm_source_review=False,
        ),
    )


def main() -> None:
    print("Running smoke audit (n=300, no LLM)...")
    result = run_full_audit(_build_request())

    ctc = result.capital_to_catalyst
    val = result.valuation

    print("\n--- capital_to_catalyst ---")
    print(f"  probability_reaches_catalyst:       {ctc.probability_reaches_catalyst:.4f}")
    print(f"  probability_cashout_before_catalyst:{ctc.probability_cashout_before_catalyst:.4f}")
    print(f"  risk_classification:                {ctc.risk_classification}")

    print("\n--- valuation (event taxonomy) ---")
    print(f"  p_any_financing_event_before_catalyst:  {val.p_any_financing_event_before_catalyst:.4f}")
    print(f"  p_financing_pressure_before_catalyst:   {val.p_financing_pressure_before_catalyst:.4f}")
    print(f"  p_dilutive_financing_before_catalyst:   {val.p_dilutive_financing_before_catalyst:.4f}")
    print(f"  p_nondilutive_financing_before_catalyst:{val.p_nondilutive_financing_before_catalyst:.4f}")
    print(f"  p_debt_or_royalty_before_catalyst:      {val.p_debt_or_royalty_before_catalyst:.4f}")
    print(f"  p_program_discontinuation_before_cat:   {val.p_program_discontinuation_before_catalyst:.4f}")

    print("\n--- method_status fields ---")
    fs = result.final_summary
    print(f"  final_summary.risk_classification:  {fs.risk_classification}")

    print("\n--- guardrail check ---")
    print(f"  llm_source_review:  {result.llm_source_review}")

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------
    probs = [
        ctc.probability_reaches_catalyst,
        ctc.probability_cashout_before_catalyst,
        val.p_any_financing_event_before_catalyst,
        val.p_financing_pressure_before_catalyst,
        val.p_dilutive_financing_before_catalyst,
        val.p_nondilutive_financing_before_catalyst,
        val.p_debt_or_royalty_before_catalyst,
        val.p_program_discontinuation_before_catalyst,
    ]
    for p in probs:
        assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"

    assert (
        val.p_any_financing_event_before_catalyst
        >= val.p_dilutive_financing_before_catalyst - 1e-6
    ), "p_any < p_dilutive"

    assert (
        val.p_any_financing_event_before_catalyst
        >= val.p_nondilutive_financing_before_catalyst - 1e-6
    ), "p_any < p_nondilutive"

    # p_financing_pressure must not be 1.0 solely from nondilutive financing;
    # debt/royalty alone does not create pressure in the base case (no planned events)
    # Here we assert that pressure <= dilutive + exhaustion (no nondistressed debt contribution)
    assert (
        val.p_financing_pressure_before_catalyst
        <= val.p_dilutive_financing_before_catalyst
        + val.p_cash_exhaustion_before_catalyst
        + val.p_program_discontinuation_before_catalyst
        + 1e-6
    ), "p_financing_pressure unexpectedly includes nondilutive debt"

    assert result.llm_source_review is None, "llm_source_review should be None when disabled"

    print("\nAll assertions passed.")
    print("smoke_audit.py: OK")


if __name__ == "__main__":
    main()
