"""Heuristic program-discontinuation taxonomy for CatalystLens."""

from __future__ import annotations

from pydantic import BaseModel, Field


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class ProgramDiscontinuationResult(BaseModel):
    p_scientific_discontinuation: float = Field(ge=0.0, le=1.0)
    p_financial_discontinuation: float = Field(ge=0.0, le=1.0)
    p_total_program_discontinuation: float = Field(ge=0.0, le=1.0)
    top_discontinuation_drivers: list[str]
    method_status: str = "heuristic"


def estimate_program_discontinuation(
    *,
    modality: str | None = None,
    disease_area: str | None = None,
    trial_phase: str,
    trial_status: str,
    endpoint_family: str | None = None,
    safety_sensitive_modality_score: float = 0.0,
    prior_human_signal: bool = False,
    open_label_design: bool = False,
    small_sample_size: bool = False,
    single_asset_dependency: float = 0.5,
    clinical_hold_or_safety_pause: bool = False,
    cash_runway_months: float = 12.0,
    posterior_pos: float = 0.35,
) -> ProgramDiscontinuationResult:
    """Separate scientific/biology discontinuation from financial discontinuation."""
    text = " ".join(str(x or "").lower() for x in (modality, disease_area, endpoint_family, trial_status))
    status = str(trial_status).lower()
    drivers: dict[str, float] = {}

    safety = _clamp(safety_sensitive_modality_score)
    if any(term in text for term in ("gene editing", "cell therapy", "adc", "gene therapy")):
        safety = max(safety, 0.7)
    drivers["safety_sensitive_modality"] = 0.22 * safety
    drivers["clinical_hold_or_safety_pause"] = 0.28 if clinical_hold_or_safety_pause or any(
        term in status for term in ("hold", "suspended", "paused")
    ) else 0.0
    drivers["low_posterior_pos"] = 0.24 * _clamp(1.0 - float(posterior_pos))
    drivers["small_open_label_or_sparse_signal"] = (
        0.08 * bool(open_label_design)
        + 0.08 * bool(small_sample_size)
        + 0.08 * (not prior_human_signal)
    )
    drivers["phase_risk"] = 0.08 if str(trial_phase).lower() in {"preclinical", "phase_1"} else 0.03

    scientific = 0.03 + sum(drivers.values())
    financial = (
        0.03
        + 0.20 * _clamp((9.0 - float(cash_runway_months)) / 9.0)
        + 0.15 * _clamp(float(single_asset_dependency))
    )
    if "terminated" in status or "withdrawn" in status:
        scientific = max(scientific, 0.65)

    scientific = _clamp(scientific)
    financial = _clamp(financial)
    total = _clamp(scientific + financial - scientific * financial)
    top = [
        name
        for name, value in sorted(drivers.items(), key=lambda item: item[1], reverse=True)
        if value > 0.02
    ][:4]
    if financial >= 0.20:
        top.append("limited_cash_runway_or_single_asset_dependency")

    return ProgramDiscontinuationResult(
        p_scientific_discontinuation=round(scientific, 4),
        p_financial_discontinuation=round(financial, 4),
        p_total_program_discontinuation=round(total, 4),
        top_discontinuation_drivers=top or ["no_dominant_discontinuation_driver"],
    )
