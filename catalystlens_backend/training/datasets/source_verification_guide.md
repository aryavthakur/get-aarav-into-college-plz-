# Source Verification Guide

**Status: Preliminary infrastructure — unverified until reviewed.**
**NOT INVESTMENT ADVICE. NOT EXTERNALLY VALIDATED.**

This guide defines the standards a human reviewer must meet before marking a row
`review_status = "source_verified"`. Every claim about a field value must be
supported by a dated primary source available on or before `as_of_date`.

---

## Core Principle: Point-in-Time Integrity

Every input field must reflect what was **publicly known on or before `as_of_date`**.
No information that became available after `as_of_date` may influence any input field.
Outcome labels reflect what happened **after** `as_of_date`, and must be sourced
from documents dated **after** `as_of_date`.

Violations of this rule constitute **lookahead leakage** and invalidate the row.

---

## Field-by-Field Evidence Standards

### Cash and Burn (`cash_evidence`, `burn_evidence`)

- **Source**: SEC 10-Q or 10-K filing.
- **Acceptable**: The most recently filed quarterly or annual balance sheet and
  cash flow statement as of `as_of_date`.
- **Quote requirement**: Cite the exact line item name and dollar figure from
  the filing (e.g., `"Cash and cash equivalents: $42.3 million as of March 31, 2024"`).
- **Date constraint**: The filing must have been made publicly available (EDGAR
  effective date) **on or before** `as_of_date`.
- **Not acceptable**: Analyst estimates, press release summaries without SEC
  cross-reference, or figures from a filing filed after `as_of_date`.
- **Confidence guide**:
  - `high`: Figure taken directly from the GAAP balance sheet.
  - `medium`: Figure taken from a non-GAAP table or management discussion that
    is consistent with the balance sheet.
  - `low`: Figure estimated from partial disclosure or prior quarter.

### Catalyst Timing (`catalyst_guidance_evidence`)

- **Source**: Company press release, earnings call transcript, SEC 8-K, or
  investor deck dated on or before `as_of_date`.
- **Quote requirement**: Cite the specific forward-looking statement (e.g.,
  `"We expect to report top-line Phase 2 results in mid-2025"`).
- **Date constraint**: Guidance must have been issued on or before `as_of_date`.
- **Not acceptable**: Analyst consensus estimates; ClinicalTrials.gov estimated
  completion dates alone (may be used as corroborating evidence, not primary).
- **Confidence guide**:
  - `high`: Company stated an explicit calendar quarter or month.
  - `medium`: Company stated a year or half-year window.
  - `low`: Timing inferred from enrollment rates or ClinicalTrials.gov only.

### Trial Status (`clinical_status_evidence`)

- **Source**: ClinicalTrials.gov record (NCT ID), SEC filing, or company press
  release dated on or before `as_of_date`.
- **Quote requirement**: Cite the overall status field from ClinicalTrials.gov
  (e.g., `"Overall Status: Recruiting"`) or equivalent company disclosure.
- **Date constraint**: For ClinicalTrials.gov, note the last-updated date
  shown on the record and confirm it is on or before `as_of_date`.
- **Not acceptable**: Trial status inferred from enrollment counts alone.

### Financing Outcomes (`financing_event_evidence`)

- **Source**: SEC 8-K (announcement of financing), press release, or
  prospectus supplement.
- **Required when**: Any of `financing_before_catalyst`,
  `clean_refinancing_before_catalyst`, `distressed_refinancing_before_catalyst`,
  `partnership_before_catalyst`, `debt_or_royalty_before_catalyst` is `True`.
- **Quote requirement**: Include the filing date, transaction type, and gross
  proceeds (e.g., `"The Company closed a $25M registered direct offering on
  June 14, 2024"`).
- **Date constraint**: The source must be dated **after** `as_of_date` but
  **before** `stated_catalyst_date` to qualify as a before-catalyst event.
- **Distress distinction**: If the offering included a going-concern disclosure,
  material discount to market, or was accompanied by a warrant coverage >50%,
  treat as `distressed_refinancing_before_catalyst`.

### Discontinuation Outcomes (`discontinuation_evidence`)

- **Source**: SEC 8-K, press release, or SEC 10-Q/10-K disclosing program
  termination.
- **Required when**: `program_discontinued_before_catalyst` is `True`.
- **Quote requirement**: Include the program name, reason for discontinuation,
  and effective date.

### Readout Outcomes (`readout_evidence`)

- **Source**: SEC 8-K, press release, or peer-reviewed publication with a
  dated embargo lift.
- **Required when**: `reached_public_readout` is `True`.
- **Quote requirement**: Include top-line result language and the public
  release date.

### Distress Outcomes (`distress_evidence`)

- **Source**: SEC 10-Q/10-K with going-concern paragraph, SEC 8-K disclosing
  cash position below operating requirements, or press release.
- **Required when**: `cash_exhaustion_before_catalyst` or
  `distressed_refinancing_before_catalyst` is `True`.

---

## Lookahead Leakage Checklist

Before submitting a row for review, confirm:

1. Every input field value appears in a source dated **on or before** `as_of_date`.
2. No outcome information (readout result, financing details, discontinuation)
   influenced how input fields were populated.
3. `market_condition_score`, if used, is derived from a date-stamped index
   value (e.g., XBI closing price on `as_of_date`), not from retrospective
   assessment.
4. `stated_catalyst_date` reflects what the company said as of `as_of_date`,
   not the actual readout date.

---

## Quote Standards

- Quotes must be **short** (under 300 characters).
- Quotes must be **directly tied** to the field being verified (not generic
  boilerplate).
- Paraphrases are acceptable only when the source is a table; mark confidence
  `medium` and note the column name.
- Do not quote forward-looking language as evidence for an outcome label.

---

## Confidence Calibration

| Confidence | Meaning |
|---|---|
| `high` | Exact figure stated in a primary dated document |
| `medium` | Figure inferred from context with low ambiguity; minor interpretation required |
| `low` | Figure estimated, proxied, or indirectly supported |

Rows with `low` confidence evidence for required fields should be set
`review_status = "needs_review"` until strengthened.

---

## Synthetic / Template Rows

Rows with `synthetic_example_only = True` exist to illustrate the schema and
CSV format. They must never be:
- used as model training data
- cited as validation evidence
- included in accuracy or calibration calculations

Remove or filter them before any downstream use.

---

*This guide is preliminary. Standards should be updated as real data is added
and ambiguous cases are adjudicated.*
