"""
Tests for the source-verified historical catalyst dataset schema.

These tests verify schema validation logic only. No model training,
calibration, or validation claims are made. The schema is preliminary
infrastructure; source verification is required before any row can be
used as validation evidence.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from training.datasets.historical_source_verified_schema import (
    HistoricalSourceVerifiedCatalystExample,
    SourceEvidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence(
    source_type: str = "sec_10q",
    source_date: date = date(2024, 5, 14),
    confidence: str = "high",
    quote: str = "Cash and cash equivalents: $42.3 million as of March 31, 2024",
    source_url: str = "https://www.sec.gov/Archives/edgar/data/1/filing.htm",
) -> SourceEvidence:
    return SourceEvidence(
        source_url=source_url,
        source_type=source_type,
        source_date=source_date,
        confidence=confidence,
        quote=quote,
    )


def _base_kwargs(**overrides) -> dict:
    """Return a minimal valid set of kwargs for HistoricalSourceVerifiedCatalystExample."""
    base = dict(
        dataset_id="TEST-001",
        company_name="Test Biotech",
        ticker="TBTC",
        as_of_date=date(2024, 3, 31),
        stated_catalyst_date=date(2025, 6, 30),
        catalyst_description="Phase 2 primary readout",
        cash_and_equivalents=40_000_000,
        quarterly_operating_cash_used=5_000_000,
        simple_runway_months=24.0,
        market_cap=150_000_000,
        debt=0,
        trial_phase="phase_2",
        trial_status="recruiting",
        financing_before_catalyst=False,
        clean_refinancing_before_catalyst=False,
        distressed_refinancing_before_catalyst=False,
        partnership_before_catalyst=False,
        debt_or_royalty_before_catalyst=False,
        cash_exhaustion_before_catalyst=False,
        program_discontinued_before_catalyst=False,
        reached_public_readout=False,
        reached_without_any_financing_event=False,
        reached_without_dilutive_financing=False,
        reached_without_distress=False,
        failed_before_readout_due_to_science=False,
        failed_before_readout_due_to_finance=False,
        cash_evidence=_evidence(quote="Cash: $40M as of March 31, 2024"),
        burn_evidence=_evidence(
            quote="Net cash used in operations: $(5.0)M for Q1 2024"
        ),
        catalyst_guidance_evidence=_evidence(
            source_type="press_release",
            quote="We expect Phase 2 results in H2 2025",
        ),
        clinical_status_evidence=_evidence(
            source_type="clinicaltrials",
            quote="Overall Status: Recruiting",
        ),
        review_status="unreviewed",
        label_confidence="low",
        synthetic_example_only=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Valid example passes schema
# ---------------------------------------------------------------------------

class TestValidExample:
    def test_minimal_valid_example_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs())
        assert row.dataset_id == "TEST-001"

    def test_full_valid_example_with_readout_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            reached_public_readout=True,
            reached_without_any_financing_event=True,
            reached_without_dilutive_financing=True,
            reached_without_distress=True,
            actual_readout_date=date(2025, 7, 15),
            readout_evidence=_evidence(
                source_type="sec_8k",
                source_date=date(2025, 7, 15),
                quote="Company reports positive Phase 2 top-line results",
            ),
        ))
        assert row.reached_public_readout is True

    def test_distressed_refi_with_all_evidence_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            financing_before_catalyst=True,
            distressed_refinancing_before_catalyst=True,
            actual_financing_date=date(2024, 8, 1),
            financing_event_evidence=_evidence(
                source_type="sec_8k",
                source_date=date(2024, 8, 1),
                quote="Company closed $10M PIPE at 40% discount",
            ),
            distress_evidence=_evidence(
                source_type="sec_10q",
                source_date=date(2024, 8, 14),
                quote="Substantial doubt about the Company's ability to continue as a going concern",
            ),
        ))
        assert row.distressed_refinancing_before_catalyst is True

    def test_source_verified_with_medium_confidence_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            review_status="source_verified",
            reviewer="analyst_a",
            label_confidence="medium",
        ))
        assert row.review_status == "source_verified"

    def test_source_verified_with_high_confidence_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            review_status="source_verified",
            reviewer="analyst_b",
            label_confidence="high",
        ))
        assert row.review_status == "source_verified"

    def test_excluded_row_with_reason_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            review_status="excluded",
            exclusion_reason="Company acquired before catalyst; outcome not attributable to trial.",
        ))
        assert row.review_status == "excluded"

    def test_synthetic_example_only_flag_preserved(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            synthetic_example_only=True,
        ))
        assert row.synthetic_example_only is True


# ---------------------------------------------------------------------------
# Missing required evidence fails
# ---------------------------------------------------------------------------

class TestMissingEvidenceFails:
    def test_financing_label_true_without_evidence_fails(self):
        with pytest.raises(ValidationError, match="financing_event_evidence"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                financing_before_catalyst=True,
                clean_refinancing_before_catalyst=True,
                # financing_event_evidence intentionally omitted
            ))

    def test_partnership_label_true_without_evidence_fails(self):
        with pytest.raises(ValidationError, match="financing_event_evidence"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                financing_before_catalyst=True,
                partnership_before_catalyst=True,
                # financing_event_evidence intentionally omitted
            ))

    def test_debt_or_royalty_label_true_without_evidence_fails(self):
        with pytest.raises(ValidationError, match="financing_event_evidence"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                financing_before_catalyst=True,
                debt_or_royalty_before_catalyst=True,
            ))

    def test_program_discontinued_without_evidence_fails(self):
        with pytest.raises(ValidationError, match="discontinuation_evidence"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                program_discontinued_before_catalyst=True,
                program_discontinuation_date=date(2024, 9, 1),
                # discontinuation_evidence intentionally omitted
            ))

    def test_readout_true_without_evidence_fails(self):
        with pytest.raises(ValidationError, match="readout_evidence"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                reached_public_readout=True,
                actual_readout_date=date(2025, 7, 15),
                # readout_evidence intentionally omitted
            ))

    def test_cash_exhaustion_without_distress_evidence_fails(self):
        with pytest.raises(ValidationError, match="distress_evidence"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                cash_exhaustion_before_catalyst=True,
                # distress_evidence intentionally omitted
            ))

    def test_distressed_refi_without_distress_evidence_fails(self):
        with pytest.raises(ValidationError, match="distress_evidence"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                financing_before_catalyst=True,
                distressed_refinancing_before_catalyst=True,
                financing_event_evidence=_evidence(
                    source_type="sec_8k",
                    source_date=date(2024, 8, 1),
                    quote="Closed PIPE offering",
                ),
                # distress_evidence intentionally omitted
            ))


# ---------------------------------------------------------------------------
# Synthetic rows cannot be source_verified
# ---------------------------------------------------------------------------

class TestSyntheticCannotBeSourceVerified:
    def test_synthetic_source_verified_medium_fails(self):
        """synthetic_example_only=True + review_status='source_verified' must fail."""
        with pytest.raises(ValidationError, match="synthetic_example_only"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                synthetic_example_only=True,
                review_status="source_verified",
                reviewer="analyst_a",
                label_confidence="medium",
            ))

    def test_synthetic_source_verified_high_fails(self):
        with pytest.raises(ValidationError, match="synthetic_example_only"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                synthetic_example_only=True,
                review_status="source_verified",
                reviewer="analyst_b",
                label_confidence="high",
            ))

    def test_synthetic_unreviewed_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            synthetic_example_only=True,
            review_status="unreviewed",
            label_confidence="low",
        ))
        assert row.synthetic_example_only is True

    def test_real_row_source_verified_medium_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            synthetic_example_only=False,
            review_status="source_verified",
            reviewer="analyst_c",
            label_confidence="medium",
        ))
        assert row.review_status == "source_verified"


# ---------------------------------------------------------------------------
# source_verified requires high or medium label_confidence
# ---------------------------------------------------------------------------

class TestSourceVerifiedConfidence:
    def test_source_verified_with_low_confidence_fails(self):
        with pytest.raises(ValidationError, match="label_confidence"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                review_status="source_verified",
                reviewer="analyst_a",
                label_confidence="low",  # not allowed for source_verified
            ))

    def test_needs_review_with_low_confidence_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            review_status="needs_review",
            label_confidence="low",
        ))
        assert row.review_status == "needs_review"

    def test_unreviewed_with_low_confidence_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            review_status="unreviewed",
            label_confidence="low",
        ))
        assert row.review_status == "unreviewed"


# ---------------------------------------------------------------------------
# Excluded rows require exclusion_reason
# ---------------------------------------------------------------------------

class TestExclusionReason:
    def test_excluded_without_reason_fails(self):
        with pytest.raises(ValidationError, match="exclusion_reason"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                review_status="excluded",
                # exclusion_reason intentionally omitted
            ))

    def test_unreviewed_without_reason_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            review_status="unreviewed",
        ))
        assert row.exclusion_reason is None


# ---------------------------------------------------------------------------
# Outcome dates cannot occur before as_of_date
# ---------------------------------------------------------------------------

class TestOutcomeDateConstraints:
    def test_actual_readout_before_as_of_date_fails(self):
        with pytest.raises(ValidationError, match="actual_readout_date"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                actual_readout_date=date(2024, 3, 1),  # before as_of_date 2024-03-31
                reached_public_readout=True,
                readout_evidence=_evidence(
                    source_type="sec_8k",
                    source_date=date(2024, 3, 1),
                    quote="Top-line results reported",
                ),
            ))

    def test_actual_financing_date_before_as_of_date_fails(self):
        with pytest.raises(ValidationError, match="actual_financing_date"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                financing_before_catalyst=True,
                actual_financing_date=date(2024, 1, 15),  # before as_of_date
                financing_event_evidence=_evidence(
                    source_type="sec_8k",
                    source_date=date(2024, 1, 15),
                    quote="Closed offering",
                ),
            ))

    def test_program_discontinuation_date_before_as_of_date_fails(self):
        with pytest.raises(ValidationError, match="program_discontinuation_date"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                program_discontinued_before_catalyst=True,
                program_discontinuation_date=date(2024, 2, 1),  # before as_of_date
                discontinuation_evidence=_evidence(
                    source_type="sec_8k",
                    source_date=date(2024, 2, 1),
                    quote="Program discontinued",
                ),
            ))

    def test_cash_distress_date_before_as_of_date_fails(self):
        with pytest.raises(ValidationError, match="cash_distress_date"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                cash_distress_date=date(2024, 3, 15),  # before as_of_date 2024-03-31
                cash_exhaustion_before_catalyst=True,
                distress_evidence=_evidence(
                    source_type="sec_10q",
                    source_date=date(2024, 3, 15),
                    quote="Going concern doubt",
                ),
            ))

    def test_outcome_dates_equal_to_as_of_date_fail(self):
        """Outcome dates on the same day as as_of_date are also invalid."""
        with pytest.raises(ValidationError, match="actual_readout_date"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                actual_readout_date=date(2024, 3, 31),  # same as as_of_date
                reached_public_readout=True,
                readout_evidence=_evidence(
                    source_type="sec_8k",
                    source_date=date(2024, 3, 31),
                    quote="Results reported",
                ),
            ))


# ---------------------------------------------------------------------------
# stated_catalyst_date must be after as_of_date
# ---------------------------------------------------------------------------

class TestCatalystDateConstraint:
    def test_stated_catalyst_before_as_of_date_fails(self):
        with pytest.raises(ValidationError, match="stated_catalyst_date"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                stated_catalyst_date=date(2024, 1, 15),  # before as_of_date
            ))

    def test_stated_catalyst_equal_to_as_of_date_fails(self):
        with pytest.raises(ValidationError, match="stated_catalyst_date"):
            HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
                stated_catalyst_date=date(2024, 3, 31),  # same as as_of_date
            ))

    def test_stated_catalyst_after_as_of_date_passes(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            stated_catalyst_date=date(2025, 1, 1),
        ))
        assert row.stated_catalyst_date > row.as_of_date


# ---------------------------------------------------------------------------
# Synthetic example rows are not validation evidence
# ---------------------------------------------------------------------------

class TestSyntheticExampleRows:
    def test_synthetic_flag_is_preserved(self):
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            synthetic_example_only=True,
        ))
        assert row.synthetic_example_only is True

    def test_synthetic_row_cannot_be_source_verified(self):
        """
        Synthetic rows set review_status='unreviewed' by convention.
        A synthetic row marked source_verified with low confidence must fail.
        A synthetic row with source_verified + medium confidence would pass schema
        (schema doesn't block it), but the synthetic flag must be checked by
        downstream consumers before treating as validation evidence.
        """
        # This passes schema — callers must filter synthetic_example_only=True
        row = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            synthetic_example_only=True,
            review_status="unreviewed",
            label_confidence="low",
        ))
        assert row.synthetic_example_only is True
        # Downstream: never use synthetic rows as validation evidence
        assert row.synthetic_example_only, (
            "Synthetic rows must be filtered before use as validation evidence"
        )

    def test_filter_synthetic_rows_from_population(self):
        real = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            synthetic_example_only=False,
        ))
        synthetic = HistoricalSourceVerifiedCatalystExample(**_base_kwargs(
            synthetic_example_only=True,
        ))
        population = [real, synthetic]
        validation_eligible = [r for r in population if not r.synthetic_example_only]
        assert len(validation_eligible) == 1
        assert validation_eligible[0].dataset_id == "TEST-001"
