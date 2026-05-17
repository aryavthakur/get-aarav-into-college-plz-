# Adjudication Checklist

**Status: Preliminary infrastructure — unverified until reviewed.**
**NOT INVESTMENT ADVICE. NOT EXTERNALLY VALIDATED.**

Use this checklist before setting `review_status = "source_verified"` on any
row in the source-verified historical dataset. Every item must be confirmed.
A single unresolved item requires setting `review_status = "needs_review"`.

---

## Section 1: Point-in-Time Integrity

- [ ] **Is `as_of_date` strictly before all outcome dates?**
  Confirm that `as_of_date` < `actual_readout_date` (if set),
  `actual_financing_date` (if set), `program_discontinuation_date` (if set),
  and `cash_distress_date` (if set). Any violation is a critical error.

- [ ] **Is `stated_catalyst_date` strictly after `as_of_date`?**
  The company-stated expected catalyst timing must have been a future date
  as of the observation point.

- [ ] **Are cash and burn sourced from filings available on or before `as_of_date`?**
  Confirm the SEC filing's EDGAR effective date (or press release date) is ≤
  `as_of_date`. Note: a Q1 10-Q filed on May 15 is not available on April 1.

- [ ] **Is catalyst timing sourced from guidance issued on or before `as_of_date`?**
  Do not use the actual readout date or any post-`as_of_date` communication
  to set `stated_catalyst_date`.

- [ ] **Is trial status sourced from a document dated on or before `as_of_date`?**
  For ClinicalTrials.gov, confirm the "Last Updated" date shown on the record
  is ≤ `as_of_date`.

- [ ] **Is there any lookahead leakage in input fields?**
  Review each input field and confirm no outcome information (financing details,
  readout results, discontinuation) influenced the value. If uncertain, set
  `review_status = "needs_review"` and document in `notes`.

---

## Section 2: Source Evidence Quality

- [ ] **Is `cash_evidence` a dated SEC filing (10-Q or 10-K)?**
  Confirm the quote cites a specific line item and dollar figure. Check that
  `confidence` is appropriate.

- [ ] **Is `burn_evidence` a dated SEC filing showing operating cash flows?**
  Confirm the figure is from the cash flow statement (operating activities),
  not from P&L. Check for YTD vs. quarterly distinction.

- [ ] **Is `catalyst_guidance_evidence` a primary company source?**
  Analyst estimates are not acceptable as primary evidence. Confirm the quote
  contains forward-looking language explicitly tied to the trial.

- [ ] **Is `clinical_status_evidence` from ClinicalTrials.gov or an SEC filing?**
  Confirm the NCT ID or filing reference is included in the `source_url`.

- [ ] **Are all required financing evidence fields populated?**
  If any financing label is `True`, confirm `financing_event_evidence` is set
  with the closing date and gross proceeds.

- [ ] **Are all required discontinuation evidence fields populated?**
  If `program_discontinued_before_catalyst = True`, confirm
  `discontinuation_evidence` is set with the formal announcement date.

- [ ] **Are all required readout evidence fields populated?**
  If `reached_public_readout = True`, confirm `readout_evidence` is set with
  the public release date.

- [ ] **Are all required distress evidence fields populated?**
  If `cash_exhaustion_before_catalyst = True` or
  `distressed_refinancing_before_catalyst = True`, confirm `distress_evidence`
  is set with the going-concern paragraph reference or discount documentation.

---

## Section 3: Label Consistency

- [ ] **Are financing labels internally consistent?**
  If any specific financing label is `True`, `financing_before_catalyst` must
  also be `True`. Check the consistency rules in `labeling_rules.md`.

- [ ] **Are `reached_without_*` labels consistent with their dependencies?**
  `reached_without_any_financing_event`, `reached_without_dilutive_financing`,
  and `reached_without_distress` must be consistent with
  `reached_public_readout` and the relevant financing labels.

- [ ] **Are outcome dates consistent with labels?**
  If `financing_before_catalyst = True`, confirm `actual_financing_date` is set.
  If `reached_public_readout = True`, confirm `actual_readout_date` is set.
  If `program_discontinued_before_catalyst = True`, confirm
  `program_discontinuation_date` is set.

- [ ] **Is label confidence set appropriately?**
  `label_confidence = "high"` requires all labels directly supported by dated
  primary sources. Downgrade to `"medium"` if any label required minor
  interpretation. Do not set `"high"` if any label is a proxy or approximation.

---

## Section 4: Exclusion Assessment

- [ ] **Should this row be excluded?**
  Consider exclusion if:
  - The as_of_date cannot be established without ambiguity.
  - Cash or burn cannot be verified from a dated filing.
  - A major corporate event (acquisition, spin-off, bankruptcy restructuring)
    occurred between `as_of_date` and the outcome, making outcome labels
    attributable to the event rather than the trial.
  - Lookahead leakage cannot be ruled out.
  - The company or trial does not match the CatalystLens model assumptions
    (e.g., multi-asset company where the specific asset's cash allocation is
    unknown).

  If excluding, set `review_status = "excluded"` and populate `exclusion_reason`.

---

## Section 5: Final Sign-off

- [ ] **Is the label exact or approximate?**
  If any label is approximate (e.g., financing date inferred from proxy),
  set `label_confidence = "low"` and document in `notes`. Do not mark
  `review_status = "source_verified"` with `label_confidence = "low"`.

- [ ] **Is `synthetic_example_only` set correctly?**
  Confirm `False` for real historical rows. `True` rows must never be used
  as validation evidence.

- [ ] **Is the reviewer field populated?**
  All `source_verified` rows must have a named reviewer.

- [ ] **Have notes been added for all ambiguities?**
  Any judgment call, approximation, or flag must be documented in `notes`
  before the row is marked verified.

---

*Completing this checklist does not constitute external validation. The dataset
remains a preliminary seed panel. External validation requires independent
data verification by a party not involved in constructing the panel.*
