"""Point-in-time historical biotech catalyst dataset schemas."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

TrialPhase = Literal["preclinical", "phase_1", "phase_2", "phase_3", "filed", "approved"]
TrialStatus = Literal[
    "not_yet_recruiting",
    "recruiting",
    "active_not_recruiting",
    "completed",
    "suspended",
    "withdrawn",
    "terminated",
    "enrolling_by_invitation",
    "available",
    "no_longer_available",
    "temporarily_not_available",
    "approved_for_marketing",
    "withheld",
    "unknown",
]
CatalystType = Literal[
    "phase_completion",
    "interim_analysis",
    "primary_readout",
    "regulatory_submission",
    "approval_decision",
    "proof_of_concept",
]
FinancingType = Literal[
    "clean_refinancing",
    "distressed_refinancing",
    "partnership",
    "debt_or_royalty",
    "none",
]
ClinicalOutcome = Literal["positive", "negative", "mixed", "not_reported"]


class HistoricalCompanyCatalystExample(BaseModel):
    example_id: str
    ticker: str
    cik: Optional[str] = None
    company_name: str
    as_of_date: date
    fiscal_quarter: Optional[str] = None
    cash: float = Field(ge=0)
    marketable_securities: float = Field(0.0, ge=0)
    quarterly_operating_cash_burn: float = Field(ge=0)
    debt: float = Field(0.0, ge=0)
    market_cap: float = Field(ge=0)
    shares_outstanding: Optional[float] = Field(None, ge=0)
    trial_phase: TrialPhase
    trial_status: TrialStatus
    disease_area: Optional[str] = None
    modality: Optional[str] = None
    endpoint_family: Optional[str] = None
    enrollment_target: Optional[int] = Field(None, ge=0)
    enrollment_completed: Optional[int] = Field(None, ge=0)
    stated_catalyst_date: date
    actual_readout_date: Optional[date] = None
    catalyst_type: CatalystType
    financing_before_catalyst: bool
    financing_type: Optional[FinancingType] = "none"
    estimated_dilution: Optional[float] = Field(None, ge=0.0, le=1.0)
    program_discontinued_before_catalyst: bool
    clinical_outcome: Optional[ClinicalOutcome] = None
    delayed_readout: Optional[bool] = None
    source_notes: Optional[str] = None
    evidence_refs: Optional[str] = None

    @model_validator(mode="after")
    def validate_point_in_time_and_labels(self) -> "HistoricalCompanyCatalystExample":
        if self.actual_readout_date is not None and self.as_of_date >= self.actual_readout_date:
            raise ValueError("as_of_date must be before actual_readout_date")
        if self.financing_before_catalyst and self.financing_type in (None, "none"):
            raise ValueError("financing_type cannot be none when financing_before_catalyst is true")
        if (
            self.enrollment_target is not None
            and self.enrollment_completed is not None
            and self.enrollment_completed > self.enrollment_target
        ):
            raise ValueError("enrollment_completed cannot exceed enrollment_target")
        return self


class HistoricalDataset(BaseModel):
    dataset_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    examples: list[HistoricalCompanyCatalystExample]
    synthetic: bool
    source_description: str

    @property
    def n_examples(self) -> int:
        return len(self.examples)
