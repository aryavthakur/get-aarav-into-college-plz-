"""Target label definitions and probability mappings for historical backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.datasets.historical_schema import HistoricalCompanyCatalystExample


@dataclass(frozen=True)
class TargetDefinition:
    target_name: str
    label_description: str
    probability_description: str
    positive_label_definition: str
    probability_source: str
    approximate: bool
    fallback_logic: str


TARGET_DEFINITIONS: dict[str, TargetDefinition] = {
    "financing_before_catalyst": TargetDefinition(
        target_name="financing_before_catalyst",
        label_description=(
            "Historical label is true when clean refinancing, distressed refinancing, "
            "partnership/non-dilutive financing, debt/royalty financing, or distress occurred before catalyst."
        ),
        probability_description=(
            "Uses financing-state probabilities rather than raw cashout risk. This target is broader than cash "
            "exhaustion and includes clean refinancing, distressed refinancing, partnership/non-dilutive financing, "
            "debt/royalty financing, and distress outcomes. Exact field: "
            "p_any_financing_event_before_catalyst when available."
        ),
        positive_label_definition="financing_before_catalyst == true",
        probability_source="valuation.p_any_financing_event_before_catalyst with fallback",
        approximate=True,
        fallback_logic=(
            "If partnership/debt/cash-exhaustion fields are unavailable, use max(raw cashout risk, "
            "clean refi + distressed refi + program discontinuation)."
        ),
    ),
    "distressed_financing_or_cashout": TargetDefinition(
        target_name="distressed_financing_or_cashout",
        label_description="Historical label is true for distressed refinancing, discontinuation, cash distress, or cashout.",
        probability_description="Uses distressed financing plus program discontinuation; falls back to raw cashout risk.",
        positive_label_definition=(
            "financing_type == distressed_refinancing or program_discontinued_before_catalyst == true"
        ),
        probability_source="valuation.p_financing_pressure_before_catalyst with fallback",
        approximate=True,
        fallback_logic="If valuation state probabilities are unavailable, use capital-to-catalyst cashout risk.",
    ),
    "program_discontinued_before_catalyst": TargetDefinition(
        target_name="program_discontinued_before_catalyst",
        label_description="Historical label is true when the program was discontinued before the catalyst/readout.",
        probability_description="Uses p_program_discontinuation_before_catalyst when available.",
        positive_label_definition="program_discontinued_before_catalyst == true",
        probability_source="valuation.p_program_discontinuation_before_catalyst",
        approximate=False,
        fallback_logic="If unavailable, use 0.0 and emit a mapping note.",
    ),
    "reached_catalyst_before_financing_pressure": TargetDefinition(
        target_name="reached_catalyst_before_financing_pressure",
        label_description="Historical label requires actual readout date and no financing pressure or discontinuation before catalyst.",
        probability_description=(
            "Uses probability_reaches_catalyst adjusted downward by p_financing_pressure_before_catalyst, "
            "not all financing events. Clean/proactive financing can be tracked separately from distress."
        ),
        positive_label_definition=(
            "actual_readout_date exists and financing_before_catalyst == false and "
            "program_discontinued_before_catalyst == false and "
            "cash_distress_or_going_concern_before_catalyst == false"
        ),
        probability_source="max(0, probability_reaches_catalyst - p_financing_pressure_before_catalyst)",
        approximate=True,
        fallback_logic="If direct no-financing catalyst probability is unavailable, subtract mapped financing risk.",
    ),
    "clinical_success": TargetDefinition(
        target_name="clinical_success",
        label_description="Historical label is true for positive or mixed clinical outcome when reported.",
        probability_description="Uses posterior mean PoS.",
        positive_label_definition="clinical_outcome in {positive, mixed}",
        probability_source="success_probability.posterior_mean",
        approximate=True,
        fallback_logic="Not scored when clinical outcome is not reported.",
    ),
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _get(obj: Any, path: str, default: float | None = None) -> float | None:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return default
    return float(cur)


def financing_probability_components(audit) -> dict[str, float]:
    valuation = getattr(audit, "valuation", None)
    ctc = getattr(audit, "capital_to_catalyst", None)
    return {
        "raw_cashout": _get(ctc, "probability_cashout_before_catalyst", 0.0) or 0.0,
        "clean_refinancing": _get(valuation, "p_clean_refinancing_before_catalyst", _get(valuation, "p_refinancing_success", 0.0)) or 0.0,
        "distressed_refinancing": _get(valuation, "p_distressed_refinancing_before_catalyst", _get(valuation, "p_distressed_financing", 0.0)) or 0.0,
        "program_discontinuation": _get(valuation, "p_program_discontinuation_before_catalyst", _get(valuation, "p_program_discontinuation", 0.0)) or 0.0,
        "partnership_or_nondilutive": _get(valuation, "p_partnership_before_catalyst", 0.0) or 0.0,
        "debt_or_royalty": _get(valuation, "p_debt_or_royalty_before_catalyst", 0.0) or 0.0,
        "cash_exhaustion": _get(valuation, "p_cash_exhaustion_before_catalyst", 0.0) or 0.0,
    }


def mapped_probabilities(audit) -> dict[str, float | str]:
    components = financing_probability_components(audit)
    valuation = getattr(audit, "valuation", None)
    has_exact_fields = (
        valuation is not None
        and hasattr(valuation, "p_any_financing_event_before_catalyst")
        and hasattr(valuation, "p_financing_pressure_before_catalyst")
    )
    if has_exact_fields:
        financing_before = _clamp(_get(valuation, "p_any_financing_event_before_catalyst", 0.0) or 0.0)
        distressed_or_cashout = _clamp(_get(valuation, "p_financing_pressure_before_catalyst", 0.0) or 0.0)
        pressure = distressed_or_cashout
        note = "Exact financing-state fields used: p_any_financing_event_before_catalyst and p_financing_pressure_before_catalyst."
    else:
        pressure = None
        note = (
            "Approximate financing-state mapping: current AuditResponse lacks explicit partnership, debt/royalty, "
            "and cash-exhaustion financing probabilities; financing_before_catalyst uses max(raw cashout risk, "
            "clean refi + distressed refi + program discontinuation)."
        )
        financing_before = None
        distressed_or_cashout = None
    exact_sum = (
        components["clean_refinancing"]
        + components["distressed_refinancing"]
        + components["program_discontinuation"]
        + components["partnership_or_nondilutive"]
        + components["debt_or_royalty"]
        + components["cash_exhaustion"]
    )
    approximate_sum = (
        components["clean_refinancing"]
        + components["distressed_refinancing"]
        + components["program_discontinuation"]
    )
    if financing_before is None:
        has_extended_states = any(
            components[name] > 0.0
            for name in ("partnership_or_nondilutive", "debt_or_royalty", "cash_exhaustion")
        )
        financing_before = _clamp(exact_sum if has_extended_states else max(components["raw_cashout"], approximate_sum))
    if distressed_or_cashout is None:
        distressed_or_cashout = _clamp(
            components["distressed_refinancing"]
            + components["program_discontinuation"]
            + components["cash_exhaustion"]
        )
        if distressed_or_cashout == 0.0:
            distressed_or_cashout = _clamp(components["raw_cashout"])
    if pressure is None:
        pressure = distressed_or_cashout
    reaches = _get(getattr(audit, "capital_to_catalyst", None), "probability_reaches_catalyst", 0.0) or 0.0
    return {
        "predicted_cashout_risk": _clamp(components["raw_cashout"]),
        "predicted_financing_before_catalyst": financing_before,
        "predicted_distressed_or_cashout_before_catalyst": distressed_or_cashout,
        "predicted_clean_or_nondilutive_financing_before_catalyst": _clamp(
            components["clean_refinancing"]
            + components["partnership_or_nondilutive"]
            + components["debt_or_royalty"]
        ),
        "predicted_program_discontinuation": _clamp(components["program_discontinuation"]),
        "predicted_reaches_catalyst_before_financing_pressure": _clamp(max(0.0, reaches - pressure)),
        "probability_mapping_note": note,
    }


def probability_for_target(audit, target_name: str) -> tuple[float, str]:
    mapped = mapped_probabilities(audit)
    if target_name == "financing_before_catalyst":
        return float(mapped["predicted_financing_before_catalyst"]), str(mapped["probability_mapping_note"])
    if target_name == "distressed_financing_or_cashout":
        return float(mapped["predicted_distressed_or_cashout_before_catalyst"]), str(mapped["probability_mapping_note"])
    if target_name == "program_discontinued_before_catalyst":
        return float(mapped["predicted_program_discontinuation"]), "Uses valuation.p_program_discontinuation_before_catalyst."
    if target_name == "reached_catalyst_before_financing_pressure":
        return float(mapped["predicted_reaches_catalyst_before_financing_pressure"]), str(mapped["probability_mapping_note"])
    if target_name == "clinical_success":
        return _clamp(_get(getattr(audit, "success_probability", None), "posterior_mean", 0.0) or 0.0), "Uses posterior mean PoS."
    raise ValueError(f"Unsupported backtest target: {target_name}")


def extract_actual_label(example: HistoricalCompanyCatalystExample, target_name: str) -> bool | None:
    if target_name == "financing_before_catalyst":
        return example.financing_before_catalyst
    if target_name == "distressed_financing_or_cashout":
        return (
            example.financing_type == "distressed_refinancing"
            or example.program_discontinued_before_catalyst
        )
    if target_name == "program_discontinued_before_catalyst":
        return example.program_discontinued_before_catalyst
    if target_name == "reached_catalyst_before_financing_pressure":
        return (
            example.actual_readout_date is not None
            and not example.financing_before_catalyst
            and not example.program_discontinued_before_catalyst
            and not example.cash_distress_or_going_concern_before_catalyst
        )
    if target_name == "clinical_success":
        if example.clinical_outcome in {"positive", "mixed"}:
            return True
        if example.clinical_outcome == "negative":
            return False
        return None
    raise ValueError(f"Unsupported backtest target: {target_name}")
