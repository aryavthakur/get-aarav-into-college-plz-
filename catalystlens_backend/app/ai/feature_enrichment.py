"""Deterministic feature enrichment scaffolds for CatalystLens."""

from __future__ import annotations

from typing import Any

from app.ai.schemas import AIFeatureEnrichment


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _text(data: dict[str, Any], *keys: str) -> str:
    return " ".join(str(data.get(key) or "").lower() for key in keys)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def enrich_company_features(input_data: dict[str, Any]) -> AIFeatureEnrichment:
    """Create auditable heuristic feature scores from structured inputs."""
    biology = _text(input_data, "disease_area", "indication", "modality", "endpoint_family")
    phase = str(input_data.get("trial_phase") or "").lower()
    status = str(input_data.get("trial_status") or "").lower()
    market_cap = float(input_data.get("market_cap") or 0.0)
    runway = float(input_data.get("simple_runway_months") or input_data.get("runway_months") or 0.0)
    posterior_pos = float(input_data.get("posterior_pos") or 0.0)
    going_concern = bool(input_data.get("going_concern_flag") or False)

    platform = _contains_any(biology, ("platform", "cell therapy", "gene therapy", "gene editing", "oncology", "rare"))
    high_partner_phase = phase in {"phase_1", "phase_2"} or "phase_2" in phase
    partnerability = 0.25
    partnerability += 0.25 if platform else 0.0
    partnerability += 0.20 if high_partner_phase else 0.0
    partnerability += 0.10 if posterior_pos >= 0.35 else 0.0
    partnerability -= 0.20 if going_concern or "bankrupt" in biology else 0.0

    proactive = 0.15
    proactive += 0.25 if market_cap >= 500_000_000 else 0.10 if market_cap >= 150_000_000 else 0.0
    proactive += 0.20 if runway >= 12 else 0.10 if runway >= 6 else 0.0
    proactive += 0.15 if phase in {"phase_2", "phase_3"} or "phase_2" in phase or "phase_3" in phase else 0.0
    proactive += 0.10 if platform else 0.0
    proactive -= 0.25 if going_concern else 0.0

    safety_sensitive = 0.25
    safety_sensitive += 0.35 if _contains_any(biology, ("gene editing", "cell therapy", "adc", "antibody-drug conjugate")) else 0.0
    safety_sensitive += 0.20 if _contains_any(biology, ("gene therapy", "first-in-human", "oncology")) else 0.0
    safety_sensitive -= 0.15 if _contains_any(biology, ("topical", "dermatology")) else 0.0
    safety_sensitive += 0.10 if _contains_any(biology, ("small molecule",)) else 0.0

    discontinuation = 0.15
    discontinuation += 0.25 if safety_sensitive >= 0.65 else 0.10 if safety_sensitive >= 0.45 else 0.0
    discontinuation += 0.30 if any(term in status for term in ("hold", "suspended", "terminated")) else 0.0
    discontinuation += 0.20 if posterior_pos and posterior_pos < 0.25 else 0.0
    discontinuation += 0.10 if _contains_any(biology, ("open label", "small sample", "no prior human signal")) else 0.0

    optimism = _clamp(float(input_data.get("management_narrative_optimism_score") or 0.5))
    grounding = _clamp(float(input_data.get("source_grounding_quality") or 0.35))

    explanation = (
        "Heuristic AI-assisted feature enrichment from structured inputs only; "
        "scores are candidate features for calibration support and do not overwrite model probabilities."
    )
    return AIFeatureEnrichment(
        example_id=str(input_data.get("example_id") or input_data.get("ticker") or "audit"),
        partnerability_score=round(_clamp(partnerability), 4),
        proactive_financing_likelihood=round(_clamp(proactive), 4),
        scientific_discontinuation_risk_score=round(_clamp(discontinuation), 4),
        safety_sensitive_modality_score=round(_clamp(safety_sensitive), 4),
        management_narrative_optimism_score=round(optimism, 4),
        source_grounding_quality=round(grounding, 4),
        explanation=explanation,
        requires_human_review=True,
    )
