"""Hierarchical probability-of-success prior lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.config import CatalystLensConfig, get_default_config


_DEFAULT_PRIORS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "pos_priors"
    / "default_hierarchical_priors.json"
)


@dataclass(frozen=True)
class HierarchicalPrior:
    alpha: float
    beta: float
    prior_source: str
    prior_confidence: float
    fallback_level: str


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    return normalized or None


def _load_default_priors(path: Path = _DEFAULT_PRIORS_PATH) -> dict:
    return json.loads(path.read_text())


def lookup_hierarchical_prior(
    phase: str,
    disease_area: str | None = None,
    modality: str | None = None,
    endpoint_family: str | None = None,
    config: CatalystLensConfig | None = None,
    priors: dict | None = None,
) -> HierarchicalPrior:
    """Lookup priors from exact stratum down to phase-only MVP fallback."""
    if config is None:
        config = get_default_config()
    if priors is None:
        priors = _load_default_priors()

    phase_key = _norm(phase)
    disease_key = _norm(disease_area)
    modality_key = _norm(modality)
    endpoint_key = _norm(endpoint_family)

    lookup_order = [
        ("phase_disease_modality_endpoint", [phase_key, disease_key, modality_key, endpoint_key]),
        ("phase_disease_modality", [phase_key, disease_key, modality_key]),
        ("phase_disease", [phase_key, disease_key]),
        ("phase_only", [phase_key]),
    ]
    for fallback_level, parts in lookup_order:
        if any(part is None for part in parts):
            continue
        key = "|".join(parts)
        row = priors.get(key)
        if row is not None:
            return HierarchicalPrior(
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                prior_source=str(row.get("source", "default_hierarchical_priors")),
                prior_confidence=float(row.get("confidence", 0.5)),
                fallback_level=fallback_level,
            )

    prior = config.phase_priors.get(phase_key or "", config.phase_priors["phase_2"])
    return HierarchicalPrior(
        alpha=float(prior.alpha),
        beta=float(prior.beta),
        prior_source="mvp_phase_prior",
        prior_confidence=0.35,
        fallback_level="mvp_phase_prior",
    )
