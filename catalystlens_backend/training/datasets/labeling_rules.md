# Labeling Rules

**Status: Preliminary infrastructure — labels are unverified until source_verified.**
**NOT INVESTMENT ADVICE. NOT EXTERNALLY VALIDATED.**

Each label below is a boolean outcome field in
`HistoricalSourceVerifiedCatalystExample`. Labels reflect events that occurred
**after** `as_of_date` and **before or at** the catalyst readout (or the end
of the observation window, if no readout occurred).

---

## `financing_before_catalyst`

**Definition**: `True` if **any** financing event of any type closed after
`as_of_date` and before `actual_readout_date` (or before
`program_discontinuation_date` if readout never occurred).

**Positive examples**:
- Company raises $30M in a registered direct offering two months before data.
- Company signs a co-development partnership with $15M upfront six weeks before results.
- Company draws down a $10M credit facility three months before readout.

**Negative examples**:
- Company raises equity after top-line results are reported.
- Company raises equity after program is discontinued.
- Company raises equity on the same day as readout (concurrent — treat as after).

**Ambiguous cases**:
- Financing closed on the same day as readout: label `False` for
  `financing_before_catalyst`; note in `notes`.
- ATM (at-the-market) program ongoing: label `True` only if shares were
  actually sold (confirmed by prospectus supplement or 8-K) before catalyst.

**Required source evidence**: `financing_event_evidence` with closing date
and gross proceeds.

**Approximate/proxy warning**: If the exact closing date cannot be verified
from a dated SEC filing or press release, set `label_confidence = "low"`.

---

## `clean_refinancing_before_catalyst`

**Definition**: `True` if an equity raise (follow-on offering, registered
direct, ATM) closed before the catalyst with **no** distress indicators:
no going-concern language in accompanying disclosures, no >30% discount to
30-day VWAP, no warrant coverage >100%.

**Positive examples**:
- $50M follow-on at-market priced follow-on offering filed under S-3.
- $20M registered direct at a 5% discount to prior close.

**Negative examples**:
- Offering with going-concern disclosure in the same 10-Q.
- PIPE priced at 40% below 30-day VWAP.
- Offering where the S-3 was replaced by S-1 due to eligibility issues.

**Ambiguous cases**:
- Offering at 15–30% discount: review warrants and going-concern language;
  default to `distressed_refinancing_before_catalyst` if uncertain.

**Required source evidence**: Prospectus supplement or 8-K confirming pricing
and no distress language.

---

## `distressed_refinancing_before_catalyst`

**Definition**: `True` if an equity raise closed before catalyst with one or
more distress indicators: going-concern opinion in concurrent SEC filing,
>30% discount to 30-day VWAP, warrant coverage >100%, or explicit statement
of insufficient funds to complete the trial.

**Positive examples**:
- PIPE at 45% discount with full-coverage warrants.
- $8M raise with going-concern paragraph in the same 10-Q.

**Negative examples**:
- At-market offering by a company that had a going-concern opinion in a
  prior period but not currently.

**Ambiguous cases**:
- Going-concern in a prior quarter but not the concurrent quarter: label
  `False` unless current disclosure references ongoing concern.

**Required source evidence**: `distress_evidence` referencing the going-concern
paragraph, discount, or distress language. Also set `financing_event_evidence`.

**Note**: `distressed_refinancing_before_catalyst = True` implies
`financing_before_catalyst = True`. Inconsistent combinations are invalid.

---

## `partnership_before_catalyst`

**Definition**: `True` if a licensing, co-development, or royalty partnership
providing **upfront cash** to the company closed before the catalyst readout.

**Positive examples**:
- $50M upfront license fee from a pharma company for ex-US rights.
- $10M co-development agreement with milestone payments starting immediately.

**Negative examples**:
- Non-binding MOU with no upfront cash.
- Partnership signed but all payments contingent on future milestones only.
- Service or CRO agreement (not a drug-asset license).

**Ambiguous cases**:
- License with nominal upfront ($100K) and large milestones: label `True` if
  cash was transferred before catalyst; note in `notes`.

**Required source evidence**: 8-K or press release confirming upfront payment
amount and transaction close date.

---

## `debt_or_royalty_before_catalyst`

**Definition**: `True` if a debt facility, term loan, royalty monetization,
or other non-dilutive capital event (excluding equity and partnerships)
closed before the catalyst readout.

**Positive examples**:
- $20M term loan from a specialty lender.
- $15M royalty monetization on a legacy product.
- $5M convertible note (convertible debt counts here, not as equity, unless
  converted before catalyst).

**Negative examples**:
- Line of credit available but not drawn.
- Convertible note that converted to equity before catalyst (reclassify as
  equity event).

**Ambiguous cases**:
- Convertible notes: label as `debt_or_royalty_before_catalyst` at issuance;
  update to equity label if converted before readout.

**Required source evidence**: 8-K or loan agreement disclosure confirming
closing date and proceeds.

---

## `cash_exhaustion_before_catalyst`

**Definition**: `True` if the company disclosed cash exhaustion (zero or
negative cash), a going-concern opinion without a subsequent financing event,
or formal cessation of operations before the catalyst readout.

**Positive examples**:
- 10-Q disclosing cash balance of $0.2M with burn of $3M/month and going-concern
  paragraph, with no subsequent financing event.
- Company files for bankruptcy before readout.

**Negative examples**:
- Going-concern opinion accompanied by a financing event within 30 days.
- Company had low cash but disclosed a subsequent financing event in the
  same filing (footnote "subsequent event").

**Ambiguous cases**:
- Going-concern opinion with a subsequent-event financing: label `False`; note
  the financing as `distressed_refinancing_before_catalyst` if applicable.

**Required source evidence**: `distress_evidence` referencing the going-concern
paragraph or cash balance disclosure with no subsequent rescue financing.

---

## `program_discontinued_before_catalyst`

**Definition**: `True` if the clinical program was formally discontinued,
placed on full clinical hold, voluntarily terminated, or withdrawn before
the catalyst readout.

**Positive examples**:
- 8-K announcing "the Company has decided to discontinue development of Drug X."
- FDA clinical hold with no resumption before readout.
- ClinicalTrials.gov status updated to "Terminated" before `actual_readout_date`.

**Negative examples**:
- Partial clinical hold limited to one site.
- Study pause due to protocol amendment (resumed before readout).
- Company acquired and program transferred to acquirer (not discontinued).

**Ambiguous cases**:
- "Strategic prioritization" language without explicit discontinuation: require
  explicit confirmation from a dated SEC filing.

**Required source evidence**: `discontinuation_evidence` with the formal
announcement date and program name.

---

## `reached_public_readout`

**Definition**: `True` if the company publicly reported top-line catalyst
results (positive, negative, or mixed) via press release, 8-K, or peer-reviewed
publication before the observation window closes.

**Positive examples**:
- 8-K reporting Phase 2 top-line data.
- Press release with primary endpoint results.

**Negative examples**:
- Conference abstract without top-line data release.
- ClinicalTrials.gov status updated to "Completed" with no results posted.

**Required source evidence**: `readout_evidence` with the public release date.

---

## `reached_without_any_financing_event`

**Definition**: `True` if `reached_public_readout = True` **and**
`financing_before_catalyst = False`.

**Note**: Computed from other labels. Verify both underlying labels are correct.

---

## `reached_without_dilutive_financing`

**Definition**: `True` if `reached_public_readout = True` **and** both
`clean_refinancing_before_catalyst = False` **and**
`distressed_refinancing_before_catalyst = False`.

**Note**: Non-dilutive events (partnership, debt/royalty) do not affect this label.

---

## `reached_without_distress`

**Definition**: `True` if `reached_public_readout = True` **and**
`distressed_refinancing_before_catalyst = False` **and**
`cash_exhaustion_before_catalyst = False`.

**Note**: A company may have done a clean equity raise and still satisfy this label.

---

## `failed_before_readout_due_to_science`

**Definition**: `True` if `reached_public_readout = False` **and** the
program was discontinued or placed on clinical hold for scientific, safety,
or efficacy reasons (not primarily financial).

**Positive examples**:
- Trial stopped for futility at interim analysis.
- Clinical hold due to serious adverse events.
- Company announced "data did not support continued development."

**Negative examples**:
- Program stopped because the company ran out of cash (→ `failed_before_readout_due_to_finance`).

**Ambiguous cases**:
- Both financial and scientific issues present: use the **primary** stated
  reason; note the secondary in `notes`. If indeterminate, set both `False`
  and note in `exclusion_reason`.

---

## `failed_before_readout_due_to_finance`

**Definition**: `True` if `reached_public_readout = False` **and** the
program ceased before readout primarily because the company could not fund
continued operations.

**Positive examples**:
- Company disclosed cash exhaustion and halted dosing before any data release.
- Program discontinued in 8-K citing "inability to raise additional capital."

**Negative examples**:
- Program stopped for safety reasons even though the company also had low cash.

---

## Consistency Rules

The following combinations are internally inconsistent and must be resolved
before marking `review_status = "source_verified"`:

| Invalid combination | Resolution |
|---|---|
| `financing_before_catalyst=False` and any specific financing label `True` | Set `financing_before_catalyst=True` |
| `reached_public_readout=True` and `program_discontinued_before_catalyst=True` | Only valid if program discontinued **after** readout; verify dates |
| `reached_without_any_financing_event=True` and `financing_before_catalyst=True` | Inconsistent; fix one |
| `failed_before_readout_due_to_science=True` and `failed_before_readout_due_to_finance=True` | Only valid if both factors contributed and both are clearly documented |
| `cash_exhaustion_before_catalyst=True` and `distress_evidence=None` | Evidence required |

---

*These rules are preliminary. Ambiguous cases should be tracked and adjudicated
to build consistent labeling precedent.*
