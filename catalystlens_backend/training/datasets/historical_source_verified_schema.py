"""
Source-verified historical biotech catalyst dataset schemas.

IMPORTANT: Rows in this schema are UNVERIFIED UNTIL individually reviewed and
marked review_status="source_verified". Do not treat any row as validation
evidence until that review is complete. This infrastructure supports future
validation work; it is not a validated dataset.

NOT INVESTMENT ADVICE. NOT EXTERNALLY VALIDATED.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Source evidence
# ---------------------------------------------------------------------------

class SourceEvidence(BaseModel):
    """A single documentary source supporting one input or outcome field."""

    source_url: str = Field(
        description="Direct URL to the source document (SEC EDGAR link, press release URL, etc.)."
    )
    source_type: Literal[
        "sec_10q",
        "sec_10k",
        "sec_8k",
        "press_release",
        "clinicaltrials",
        "investor_deck",
        "earnings_call",
        "other",
    ]
    source_date: date = Field(
        description="Date the source document was published or filed."
    )
    accessed_date: Optional[date] = Field(
        None,
        description="Date the reviewer accessed this URL. Leave None if unknown.",
    )
    quote: str = Field(
        description=(
            "Short verbatim or near-verbatim excerpt from the source directly "
            "supporting the field value. Keep under 300 characters."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high = exact figure stated in document; "
            "medium = figure inferred from context with low ambiguity; "
            "low = figure estimated or indirectly supported."
        )
    )
    notes: Optional[str] = Field(
        None,
        description="Reviewer notes on ambiguity, caveats, or proxy usage.",
    )


# ---------------------------------------------------------------------------
# Main schema
# ---------------------------------------------------------------------------

class HistoricalSourceVerifiedCatalystExample(BaseModel):
    """
    A single point-in-time biotech catalyst observation with source-verified
    inputs and outcome labels.

    All input fields reflect information publicly available as of `as_of_date`.
    All outcome fields reflect what actually happened after `as_of_date`.
    No lookahead leakage is permitted: no outcome information may influence
    any input field.

    Review status must be "source_verified" before a row is used as
    validation evidence.
    """

    # ------------------------------------------------------------------
    # Core identifiers
    # ------------------------------------------------------------------
    dataset_id: str = Field(
        description="Unique identifier for this row (e.g. 'TICKER-YYYYQN-v1')."
    )
    company_name: str
    ticker: str
    cik: Optional[str] = Field(
        None,
        description="SEC CIK number, if available.",
    )
    as_of_date: date = Field(
        description=(
            "The point-in-time date at which all input fields are observed. "
            "Must precede all outcome dates."
        )
    )
    stated_catalyst_date: date = Field(
        description=(
            "The company-stated expected catalyst date as of as_of_date, "
            "sourced from company guidance. Must be after as_of_date."
        )
    )
    catalyst_description: str = Field(
        description="Brief description of the expected catalyst event."
    )

    # ------------------------------------------------------------------
    # Input fields (as of as_of_date)
    # ------------------------------------------------------------------
    cash_and_equivalents: float = Field(
        ge=0,
        description="Cash and cash equivalents in USD as of as_of_date.",
    )
    quarterly_operating_cash_used: float = Field(
        ge=0,
        description=(
            "Most recently reported quarterly operating cash used (absolute value), "
            "in USD. From operating activities on the cash flow statement."
        ),
    )
    simple_runway_months: float = Field(
        ge=0,
        description=(
            "cash_and_equivalents / (quarterly_operating_cash_used / 3). "
            "Computed field; may be stored for convenience."
        ),
    )
    market_cap: float = Field(
        ge=0,
        description="Market capitalization in USD as of as_of_date.",
    )
    debt: float = Field(
        0.0,
        ge=0,
        description="Total debt (short-term + long-term) in USD as of as_of_date.",
    )
    trial_phase: Literal[
        "preclinical", "phase_1", "phase_1_2", "phase_2", "phase_2_3",
        "phase_3", "filed", "approved"
    ]
    trial_status: Literal[
        "not_yet_recruiting", "recruiting", "active_not_recruiting",
        "completed", "suspended", "withdrawn", "terminated",
        "enrolling_by_invitation", "unknown",
    ]
    disease_area: Optional[str] = None
    modality: Optional[str] = None
    endpoint_family: Optional[str] = None
    catalyst_type: Optional[Literal[
        "phase_completion", "interim_analysis", "primary_readout",
        "regulatory_submission", "approval_decision", "proof_of_concept",
    ]] = None
    prior_human_signal: Optional[bool] = Field(
        None,
        description=(
            "True if the asset has prior positive human data (e.g. Phase 1 PK/PD "
            "results available as of as_of_date)."
        ),
    )
    single_asset_dependency: Optional[bool] = Field(
        None,
        description=(
            "True if this asset is the only material pipeline asset as of as_of_date."
        ),
    )
    market_condition_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=10.0,
        description=(
            "Proxy for biotech market conditions as of as_of_date (0=hostile, "
            "10=euphoric). Must be sourced from a date-stamped index or proxy, "
            "not from hindsight."
        ),
    )

    # ------------------------------------------------------------------
    # Outcome labels (what happened after as_of_date)
    # ------------------------------------------------------------------
    financing_before_catalyst: bool = Field(
        description="True if any financing event closed before the actual catalyst readout date."
    )
    clean_refinancing_before_catalyst: bool = Field(
        description=(
            "True if an at-market or above-market equity raise closed before "
            "the catalyst readout, with no going-concern language or distress signal."
        )
    )
    distressed_refinancing_before_catalyst: bool = Field(
        description=(
            "True if a below-market equity raise, PIPE with deep discount, "
            "or raise accompanied by going-concern disclosure closed before the catalyst readout."
        )
    )
    partnership_before_catalyst: bool = Field(
        description=(
            "True if a licensing, co-development, or royalty partnership providing "
            "upfront cash closed before the catalyst readout."
        )
    )
    debt_or_royalty_before_catalyst: bool = Field(
        description=(
            "True if a debt facility, royalty monetization, or non-dilutive "
            "capital event (excluding partnerships) closed before the catalyst readout."
        )
    )
    cash_exhaustion_before_catalyst: bool = Field(
        description=(
            "True if the company disclosed cash exhaustion, a going-concern opinion, "
            "or ceased operations before the catalyst readout."
        )
    )
    program_discontinued_before_catalyst: bool = Field(
        description=(
            "True if the clinical program was formally discontinued, terminated, "
            "or placed on clinical hold before the catalyst readout."
        )
    )
    reached_public_readout: bool = Field(
        description="True if the company publicly reported top-line catalyst results."
    )
    reached_without_any_financing_event: bool = Field(
        description=(
            "True if reached_public_readout is True and no financing event of any "
            "type occurred between as_of_date and the readout."
        )
    )
    reached_without_dilutive_financing: bool = Field(
        description=(
            "True if reached_public_readout is True and no equity dilution "
            "(clean or distressed) occurred between as_of_date and the readout."
        )
    )
    reached_without_distress: bool = Field(
        description=(
            "True if reached_public_readout is True and no distressed_refinancing, "
            "cash_exhaustion, or going-concern event occurred before the readout."
        )
    )
    failed_before_readout_due_to_science: bool = Field(
        description=(
            "True if the program was discontinued or placed on clinical hold "
            "for scientific/safety/efficacy reasons before a public readout."
        )
    )
    failed_before_readout_due_to_finance: bool = Field(
        description=(
            "True if the program ceased before readout primarily due to cash "
            "exhaustion or inability to finance continued operations."
        )
    )

    # ------------------------------------------------------------------
    # Outcome dates
    # ------------------------------------------------------------------
    actual_financing_date: Optional[date] = Field(
        None,
        description=(
            "Date the first financing event (of any type) closed after as_of_date. "
            "None if no financing event occurred."
        ),
    )
    actual_readout_date: Optional[date] = Field(
        None,
        description=(
            "Date of public top-line data release. None if not yet reported "
            "or if program discontinued before readout."
        ),
    )
    program_discontinuation_date: Optional[date] = Field(
        None,
        description="Date the program was formally discontinued, if applicable.",
    )
    cash_distress_date: Optional[date] = Field(
        None,
        description=(
            "Date of first going-concern disclosure or cash exhaustion event, "
            "if applicable."
        ),
    )

    # ------------------------------------------------------------------
    # Source evidence
    # ------------------------------------------------------------------
    cash_evidence: SourceEvidence = Field(
        description="Evidence for cash_and_equivalents value."
    )
    burn_evidence: SourceEvidence = Field(
        description="Evidence for quarterly_operating_cash_used value."
    )
    catalyst_guidance_evidence: SourceEvidence = Field(
        description="Evidence for stated_catalyst_date from company guidance."
    )
    clinical_status_evidence: SourceEvidence = Field(
        description="Evidence for trial_phase and trial_status."
    )
    financing_event_evidence: Optional[SourceEvidence] = Field(
        None,
        description=(
            "Evidence for financing outcome labels. Required when any financing "
            "label is True."
        ),
    )
    discontinuation_evidence: Optional[SourceEvidence] = Field(
        None,
        description=(
            "Evidence for program_discontinued_before_catalyst. Required when True."
        ),
    )
    readout_evidence: Optional[SourceEvidence] = Field(
        None,
        description=(
            "Evidence for reached_public_readout and actual_readout_date. "
            "Required when reached_public_readout is True."
        ),
    )
    distress_evidence: Optional[SourceEvidence] = Field(
        None,
        description=(
            "Evidence for cash_exhaustion_before_catalyst or "
            "distressed_refinancing_before_catalyst. Required when either is True."
        ),
    )

    # ------------------------------------------------------------------
    # Review fields
    # ------------------------------------------------------------------
    review_status: Literal[
        "unreviewed", "needs_review", "source_verified", "excluded"
    ] = Field(
        "unreviewed",
        description=(
            "Only 'source_verified' rows may be used as validation evidence. "
            "'unreviewed' rows are preliminary and must not be cited as validation."
        ),
    )
    reviewer: Optional[str] = Field(
        None,
        description="Name or ID of the person who completed source verification.",
    )
    label_confidence: Literal["high", "medium", "low"] = Field(
        "low",
        description=(
            "high = all labels directly supported by dated primary sources; "
            "medium = most labels supported, minor ambiguity in ≤1 label; "
            "low = one or more labels are approximate or proxy-based."
        ),
    )
    exclusion_reason: Optional[str] = Field(
        None,
        description="Required when review_status='excluded'. Explain why.",
    )
    notes: Optional[str] = Field(
        None,
        description="Reviewer notes, caveats, or flags for future adjudication.",
    )
    synthetic_example_only: bool = Field(
        False,
        description=(
            "Set True for illustrative/template rows that must never be used "
            "as validation evidence or model training data."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_point_in_time_integrity(self) -> "HistoricalSourceVerifiedCatalystExample":
        # stated_catalyst_date must be after as_of_date
        if self.stated_catalyst_date <= self.as_of_date:
            raise ValueError(
                f"stated_catalyst_date ({self.stated_catalyst_date}) must be "
                f"after as_of_date ({self.as_of_date})"
            )

        # Outcome dates must not precede as_of_date
        for field_name, outcome_date in [
            ("actual_financing_date", self.actual_financing_date),
            ("actual_readout_date", self.actual_readout_date),
            ("program_discontinuation_date", self.program_discontinuation_date),
            ("cash_distress_date", self.cash_distress_date),
        ]:
            if outcome_date is not None and outcome_date <= self.as_of_date:
                raise ValueError(
                    f"{field_name} ({outcome_date}) must be after as_of_date "
                    f"({self.as_of_date})"
                )

        # Financing evidence required when any financing label is True
        any_financing = (
            self.financing_before_catalyst
            or self.clean_refinancing_before_catalyst
            or self.distressed_refinancing_before_catalyst
            or self.partnership_before_catalyst
            or self.debt_or_royalty_before_catalyst
        )
        if any_financing and self.financing_event_evidence is None:
            raise ValueError(
                "financing_event_evidence is required when any financing label is True"
            )

        # Discontinuation evidence required when label is True
        if self.program_discontinued_before_catalyst and self.discontinuation_evidence is None:
            raise ValueError(
                "discontinuation_evidence is required when "
                "program_discontinued_before_catalyst is True"
            )

        # Readout evidence required when reached_public_readout is True
        if self.reached_public_readout and self.readout_evidence is None:
            raise ValueError(
                "readout_evidence is required when reached_public_readout is True"
            )

        # Distress evidence required when distress labels are True
        any_distress = (
            self.cash_exhaustion_before_catalyst
            or self.distressed_refinancing_before_catalyst
        )
        if any_distress and self.distress_evidence is None:
            raise ValueError(
                "distress_evidence is required when cash_exhaustion_before_catalyst "
                "or distressed_refinancing_before_catalyst is True"
            )

        # source_verified rows must have high or medium label_confidence
        if (
            self.review_status == "source_verified"
            and self.label_confidence == "low"
        ):
            raise ValueError(
                "source_verified rows must have label_confidence 'high' or 'medium', "
                "not 'low'"
            )

        # excluded rows must provide exclusion_reason
        if self.review_status == "excluded" and not self.exclusion_reason:
            raise ValueError(
                "exclusion_reason is required when review_status='excluded'"
            )

        # synthetic rows can never be marked source_verified
        if self.synthetic_example_only and self.review_status == "source_verified":
            raise ValueError(
                "synthetic_example_only rows cannot be marked review_status='source_verified'. "
                "Synthetic rows must never be used as validation evidence."
            )

        return self
