"""Heuristic financing-event taxonomy for CatalystLens.

This is not a trained financing model. It provides transparent, bounded
state estimates so validation targets can separate proactive financing,
partnerships, distress, and cash exhaustion without treating AI as an oracle.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class FinancingStrategyResult(BaseModel):
    p_proactive_clean_refinancing: float = Field(ge=0.0, le=1.0)
    p_partnership_or_nondilutive: float = Field(ge=0.0, le=1.0)
    p_distressed_financing: float = Field(ge=0.0, le=1.0)
    p_cash_exhaustion: float = Field(ge=0.0, le=1.0)
    p_dilutive_financing: float = Field(ge=0.0, le=1.0)
    p_nondilutive_financing: float = Field(ge=0.0, le=1.0)
    p_debt_or_royalty: float = Field(ge=0.0, le=1.0)
    method_status: str = "heuristic"


def estimate_financing_strategy(
    *,
    months_to_catalyst: float,
    simple_runway_months: float,
    market_cap: float,
    market_condition_score: float,
    trial_phase: str,
    posterior_pos: float,
    catalyst_type: str,
    recent_positive_signal: bool,
    partnerability_score: float = 0.0,
) -> FinancingStrategyResult:
    """Estimate financing-event mix with deterministic, untrained heuristics."""
    runway_gap = max(float(months_to_catalyst) - float(simple_runway_months), 0.0)
    market_norm = _clamp((float(market_condition_score) - 1.0) / 9.0)
    cap_norm = _clamp(float(market_cap) / 1_000_000_000.0)
    pos = _clamp(posterior_pos)
    partnerability = _clamp(partnerability_score)
    phase = str(trial_phase).lower()
    catalyst = str(catalyst_type).lower()

    clean = (
        0.05
        + 0.22 * market_norm
        + 0.18 * cap_norm
        + 0.12 * (phase in {"phase_2", "phase_3"})
        + 0.10 * (pos >= 0.35)
        + 0.08 * bool(recent_positive_signal)
        + 0.06 * ("readout" in catalyst or "primary" in catalyst)
    )
    clean *= 0.55 + 0.45 * _clamp(float(simple_runway_months) / max(float(months_to_catalyst), 1.0))

    partnership = (
        0.04
        + 0.45 * partnerability
        + 0.12 * (phase in {"phase_1", "phase_2"})
        + 0.08 * pos
        + 0.05 * bool(recent_positive_signal)
    )

    distressed = (
        0.04
        + 0.10 * (runway_gap > 0)
        + 0.22 * _clamp(runway_gap / 12.0)
        + 0.14 * (1.0 - market_norm)
        + 0.08 * (1.0 - cap_norm)
    )
    cash_exhaustion = (
        0.02
        + 0.24 * _clamp(runway_gap / 18.0)
        + 0.12 * (1.0 - market_norm)
        + 0.10 * (1.0 - cap_norm)
        - 0.08 * partnership
    )

    # Heuristic debt/royalty estimate: available for established companies with moderate
    # runway pressure and decent market conditions. Not distressed-driven.
    # Bounded away from clean equity (requires specific structure) and distress.
    debt_royalty = (
        0.03
        + 0.10 * cap_norm
        + 0.07 * market_norm
        + 0.05 * (phase in {"phase_2", "phase_3"})
        + 0.04 * (pos >= 0.30)
        + 0.08 * _clamp((float(simple_runway_months) - float(months_to_catalyst)) / max(float(months_to_catalyst), 1.0) + 0.5)
    )
    # Reduce debt/royalty likelihood when already in cash exhaustion territory
    debt_royalty *= 1.0 - 0.5 * _clamp(runway_gap / 12.0)

    clean = _clamp(clean)
    partnership = _clamp(partnership)
    distressed = _clamp(distressed)
    cash_exhaustion = _clamp(cash_exhaustion)
    debt_royalty = _clamp(debt_royalty)

    total = clean + partnership + distressed + cash_exhaustion + debt_royalty
    if total > 1.0:
        scale = 1.0 / total
        clean *= scale
        partnership *= scale
        distressed *= scale
        cash_exhaustion *= scale
        debt_royalty *= scale

    return FinancingStrategyResult(
        p_proactive_clean_refinancing=round(_clamp(clean), 4),
        p_partnership_or_nondilutive=round(_clamp(partnership), 4),
        p_distressed_financing=round(_clamp(distressed), 4),
        p_cash_exhaustion=round(_clamp(cash_exhaustion), 4),
        p_dilutive_financing=round(_clamp(clean + distressed), 4),
        p_nondilutive_financing=round(_clamp(partnership + debt_royalty), 4),
        p_debt_or_royalty=round(_clamp(debt_royalty), 4),
    )
