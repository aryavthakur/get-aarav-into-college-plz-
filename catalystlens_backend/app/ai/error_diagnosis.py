"""Rule-based backtest error diagnosis.

This module is intentionally deterministic. It labels likely missing feature
families for analyst review; it does not establish causal explanations.
"""

from __future__ import annotations

from typing import Any

from app.ai.schemas import AIBacktestErrorDiagnosis, AIBacktestFailureMode, AIErrorType


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(float(value or 0))


def _as_float(value: Any) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))


def _error_type(y_true: int, y_prob: float, absolute_error: float, target: str) -> AIErrorType:
    if target == "reached_catalyst_before_financing_pressure" and absolute_error > 0.5:
        return "ambiguous_label"
    if y_true == 1 and y_prob < 0.5:
        return "false_negative"
    if y_true == 0 and y_prob >= 0.5:
        return "false_positive"
    return "calibration_error"


def _confidence(row: dict[str, Any], base: float) -> float:
    fields = ["financing_type", "trial_status", "modality", "disease_area", "clinical_outcome"]
    available = sum(1 for field in fields if row.get(field) not in (None, ""))
    return max(0.1, min(0.9, base + available * 0.05))


def diagnose_prediction_error(row: dict[str, Any]) -> AIBacktestErrorDiagnosis:
    """Diagnose a per-example prediction error using transparent rules."""
    target = str(row.get("target", ""))
    y_true = _as_int(row.get("y_true"))
    y_prob = _as_float(row.get("y_prob"))
    absolute_error = _as_float(row.get("absolute_error", abs(y_prob - y_true)))
    financing_type = str(row.get("financing_type") or "").lower()

    mode: AIBacktestFailureMode = "other"
    missing_features: list[str] = ["source_grounding_quality"]
    patch = "Review source data and add target-specific calibrated features before changing model probabilities."
    confidence = _confidence(row, 0.35)

    if target == "financing_before_catalyst" and y_true == 1 and y_prob < 0.35:
        if financing_type == "partnership":
            mode = "partnership_underpredicted"
            missing_features = ["partnerability_score", "strategic_collaboration_likelihood"]
            patch = "Add partnerability and strategic-collaboration features to financing-state calibration."
            confidence = _confidence(row, 0.65)
        elif financing_type == "clean_refinancing":
            mode = "proactive_financing_underpredicted"
            missing_features = ["proactive_financing_likelihood", "market_window_strength", "high_value_catalyst"]
            patch = "Separate proactive clean financing from distress-driven financing pressure."
            confidence = _confidence(row, 0.65)
        elif financing_type == "distressed_refinancing":
            mode = "distressed_financing_underpredicted"
            missing_features = ["financing_pressure_score", "market_window_strength", "going_concern_risk"]
            patch = "Increase distressed-financing feature coverage and validate labels against source filings."
            confidence = _confidence(row, 0.6)

    elif target == "program_discontinued_before_catalyst" and y_true == 1 and y_prob < 0.35:
        mode = "scientific_discontinuation_underpredicted"
        missing_features = [
            "safety_sensitive_modality_score",
            "clinical_hold_status",
            "single_asset_dependency",
            "modality_safety_prior",
        ]
        patch = "Add biology/safety discontinuation features separate from financial runway features."
        confidence = _confidence(row, 0.6)

    elif target == "reached_catalyst_before_financing_pressure" and absolute_error > 0.5:
        mode = "reached_catalyst_label_ambiguous"
        missing_features = [
            "separate_reached_public_readout_label",
            "separate_reached_without_dilution_label",
        ]
        patch = "Split reached-catalyst labels into public readout, no-dilution, and no-pressure outcomes."
        confidence = _confidence(row, 0.55)

    elif y_true == 0 and y_prob > 0.65:
        mode = "cash_distress_overpredicted"
        missing_features = ["proactive_financing_likelihood", "financing_window_quality"]
        patch = "Check whether proactive financing or partnership avoided modeled cash distress."
        confidence = _confidence(row, 0.45)

    if not row.get("company_name") or not row.get("ticker"):
        mode = "missing_source_data"
        missing_features = sorted(set(missing_features + ["company_identifier_source"]))
        confidence = min(confidence, 0.4)

    return AIBacktestErrorDiagnosis(
        example_id=str(row.get("example_id") or "unknown"),
        company_name=str(row.get("company_name") or "Unknown"),
        ticker=str(row.get("ticker") or "UNKNOWN"),
        target=target,
        y_true=y_true,
        y_prob=y_prob,
        absolute_error=absolute_error,
        error_type=_error_type(y_true, y_prob, absolute_error, target),
        diagnosed_failure_mode=mode,
        likely_missing_features=missing_features,
        suggested_model_patch=patch,
        confidence=confidence,
    )
