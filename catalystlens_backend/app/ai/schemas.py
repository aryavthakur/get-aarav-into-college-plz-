"""Schemas for auditable AI-assisted CatalystLens utilities."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


AIErrorType = Literal[
    "false_positive",
    "false_negative",
    "calibration_error",
    "ambiguous_label",
]

AIBacktestFailureMode = Literal[
    "proactive_financing_underpredicted",
    "partnership_underpredicted",
    "distressed_financing_underpredicted",
    "scientific_discontinuation_underpredicted",
    "cash_distress_overpredicted",
    "reached_catalyst_label_ambiguous",
    "missing_source_data",
    "other",
]


class AIExtractionResult(BaseModel):
    raw_text: str
    extracted_value: Optional[str] = None
    normalized_value: Optional[str] = None
    extraction_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_url: Optional[str] = None
    evidence_span: Optional[str] = None
    model_name: Optional[str] = None
    requires_human_review: bool


class AIBacktestErrorDiagnosis(BaseModel):
    example_id: str
    company_name: str
    ticker: str
    target: str
    y_true: int
    y_prob: float = Field(ge=0.0, le=1.0)
    absolute_error: float = Field(ge=0.0, le=1.0)
    error_type: AIErrorType
    diagnosed_failure_mode: AIBacktestFailureMode
    likely_missing_features: list[str]
    suggested_model_patch: str
    confidence: float = Field(ge=0.0, le=1.0)
    method_status: Literal["heuristic_ai_assisted"] = "heuristic_ai_assisted"


class AIFeatureEnrichment(BaseModel):
    example_id: str
    partnerability_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    proactive_financing_likelihood: Optional[float] = Field(None, ge=0.0, le=1.0)
    scientific_discontinuation_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    safety_sensitive_modality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    management_narrative_optimism_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_grounding_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    explanation: str
    requires_human_review: bool
    method_status: Literal["heuristic_ai_assisted"] = "heuristic_ai_assisted"
