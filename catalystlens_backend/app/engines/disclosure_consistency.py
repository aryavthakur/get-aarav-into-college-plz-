"""
Disclosure Consistency Engine.

Measures the divergence between:
  1. Management narrative framing (probability-like scores per category)
  2. Structured quantitative audit output (model-derived scores per category)

Uses Jensen-Shannon Divergence (JSD), which is symmetric and bounded [0, ln(2)].
Normalised to [0, 1] by dividing by ln(2).

JSD(P||Q) = (H(M) - H(P)/2 - H(Q)/2) / ln(2)  where M = (P+Q)/2
           = (KL(P||M) + KL(Q||M)) / (2 * ln(2))

A JSD near 0 indicates that the narrative closely matches the model audit.
A JSD near 1 indicates severe divergence.

IMPORTANT: A material gap does not imply misconduct. It indicates that the
management narrative framing is materially more optimistic than the structured
quantitative estimate.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from app.core.config import CatalystLensConfig, get_default_config
from app.models.schemas import DisclosureInput, DisclosureConsistencyResult

_EPSILON = 1e-9

DISCLOSURE_CATEGORIES = [
    "runway_strength",
    "clinical_timeline_confidence",
    "dilution_risk",
    "trial_maturity",
    "endpoint_strength",
    "pipeline_diversification",
]


def normalize_distribution(dist: Dict[str, float]) -> Dict[str, float]:
    """
    Normalise a score dict to a probability distribution summing to 1.

    Clips values to [epsilon, 1] before normalising.
    Returns a dict keyed by the same keys.
    """
    clipped = {k: max(_EPSILON, float(v)) for k, v in dist.items()}
    total = sum(clipped.values())
    if total <= 0:
        n = len(clipped)
        return {k: 1.0 / n for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def _align_distributions(
    p: Dict[str, float],
    q: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two score dicts to the same key universe, returning numpy arrays.

    Missing keys default to epsilon so neither distribution has zero mass.
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_norm = normalize_distribution({k: p.get(k, _EPSILON) for k in keys})
    q_norm = normalize_distribution({k: q.get(k, _EPSILON) for k in keys})
    p_arr = np.array([p_norm[k] for k in keys], dtype=float)
    q_arr = np.array([q_norm[k] for k in keys], dtype=float)
    return p_arr, q_arr


def calculate_kl_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
) -> float:
    """
    KL(P || Q) = sum_i p_i * log(p_i / q_i).

    Asymmetric; measures extra bits needed to encode P using Q.
    """
    p_arr, q_arr = _align_distributions(p, q)
    # Guard against log(0)
    mask = p_arr > _EPSILON
    kl = float(np.sum(p_arr[mask] * np.log(p_arr[mask] / q_arr[mask])))
    return max(0.0, kl)


def calculate_jensen_shannon_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
) -> float:
    """
    Jensen-Shannon Divergence, normalised to [0, 1].

    JSD = (KL(P||M) + KL(Q||M)) / (2 * ln(2))  where M = (P+Q)/2

    Returns value in [0, 1].
    """
    p_arr, q_arr = _align_distributions(p, q)
    m_arr = 0.5 * (p_arr + q_arr)

    keys = sorted(set(p.keys()) | set(q.keys()))
    m_dict = {k: float(m_arr[i]) for i, k in enumerate(keys)}
    p_dict = {k: float(p_arr[i]) for i, k in enumerate(keys)}
    q_dict = {k: float(q_arr[i]) for i, k in enumerate(keys)}

    kl_pm = calculate_kl_divergence(p_dict, m_dict)
    kl_qm = calculate_kl_divergence(q_dict, m_dict)

    jsd_nats = 0.5 * kl_pm + 0.5 * kl_qm
    jsd_normalised = jsd_nats / math.log(2.0)
    return float(np.clip(jsd_normalised, 0.0, 1.0))


def classify_disclosure_gap(
    jsd: float,
    absolute_gap: float = 0.0,
    config: CatalystLensConfig | None = None,
) -> str:
    """Classify disclosure gap based on relative-shape and absolute-score gaps."""
    if config is None:
        config = get_default_config()
    t = config.disclosure_thresholds
    combined_gap = max(jsd, absolute_gap)
    if combined_gap <= t.aligned_jsd_max:
        return "aligned"
    if combined_gap <= t.mild_jsd_max:
        return "mild inconsistency"
    if combined_gap <= t.material_jsd_max:
        return "material inconsistency"
    return "severe inconsistency"


def _build_category_gaps(
    narrative: Dict[str, float],
    audit: Dict[str, float],
) -> Dict[str, float]:
    """Per-category raw score differences (narrative - audit)."""
    keys = sorted(set(narrative.keys()) | set(audit.keys()))
    return {k: round(narrative.get(k, 0.0) - audit.get(k, 0.0), 4) for k in keys}


def _build_interpretation(jsd: float, gap_class: str, category_gaps: Dict[str, float]) -> str:
    largest_gap_key = max(category_gaps, key=lambda k: abs(category_gaps[k])) if category_gaps else "N/A"
    direction = "over-stated" if category_gaps.get(largest_gap_key, 0) > 0 else "under-stated"

    if gap_class == "aligned":
        return (
            "The company's narrative framing is broadly consistent with the "
            "structured model estimates across both relative category shape and "
            "absolute category scores. No material disclosure gap detected."
        )
    elif gap_class == "mild inconsistency":
        return (
            f"A mild divergence exists between the management narrative and the "
            f"structured audit (JSD={jsd:.3f}). The category with the largest gap "
            f"is '{largest_gap_key}', which appears {direction} in the narrative. "
            "This level of divergence is common in biotech communications."
        )
    elif gap_class == "material inconsistency":
        return (
            f"A material divergence exists (JSD={jsd:.3f}). The management narrative "
            f"is measurably more optimistic than the model estimate, most prominently "
            f"in '{largest_gap_key}' ({direction}). This does not imply misconduct; "
            "it indicates investors should discount management guidance and conduct "
            "independent diligence on the flagged categories."
        )
    else:
        return (
            f"A severe divergence exists (JSD={jsd:.3f}). The management narrative "
            f"diverges substantially from the structured model output, with the largest "
            f"gap in '{largest_gap_key}' ({direction}). This level of inconsistency "
            "warrants heightened diligence scrutiny."
        )


def run_disclosure_consistency_analysis(
    inputs: DisclosureInput,
    config: CatalystLensConfig | None = None,
) -> DisclosureConsistencyResult:
    """Run the full disclosure consistency analysis."""
    if config is None:
        config = get_default_config()

    narrative = inputs.company_narrative_distribution
    audit = inputs.structured_audit_distribution

    jsd = calculate_jensen_shannon_divergence(narrative, audit)
    kl_nva = calculate_kl_divergence(normalize_distribution(narrative), normalize_distribution(audit))
    kl_avn = calculate_kl_divergence(normalize_distribution(audit), normalize_distribution(narrative))
    category_gaps = _build_category_gaps(narrative, audit)
    absolute_gap_values = [abs(v) for v in category_gaps.values()]
    mean_absolute_gap = float(np.mean(absolute_gap_values)) if absolute_gap_values else 0.0
    optimism_bias = float(np.mean(list(category_gaps.values()))) if category_gaps else 0.0
    max_category_gap = float(max(absolute_gap_values)) if absolute_gap_values else 0.0
    combined_gap = 0.5 * jsd + 0.5 * mean_absolute_gap
    gap_class = classify_disclosure_gap(jsd, combined_gap, config)
    interpretation = _build_interpretation(jsd, gap_class, category_gaps)

    keys = sorted(set(narrative.keys()) | set(audit.keys()))
    narrative_norm = normalize_distribution({k: narrative.get(k, _EPSILON) for k in keys})
    audit_norm = normalize_distribution({k: audit.get(k, _EPSILON) for k in keys})

    return DisclosureConsistencyResult(
        jsd_score=round(jsd, 6),
        kl_narrative_vs_audit=round(kl_nva, 6),
        kl_audit_vs_narrative=round(kl_avn, 6),
        mean_absolute_gap=round(mean_absolute_gap, 6),
        optimism_bias=round(optimism_bias, 6),
        max_category_gap=round(max_category_gap, 6),
        combined_gap_score=round(combined_gap, 6),
        gap_classification=gap_class,
        category_gaps={k: round(v, 4) for k, v in category_gaps.items()},
        narrative_normalized={k: round(v, 4) for k, v in narrative_norm.items()},
        audit_normalized={k: round(v, 4) for k, v in audit_norm.items()},
        interpretation=interpretation,
    )
