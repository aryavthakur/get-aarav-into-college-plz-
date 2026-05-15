"""Rule-based first-pass financing and discontinuation labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


FinancingEventType = Literal[
    "clean_refinancing",
    "distressed_refinancing",
    "partnership",
    "program_discontinuation",
    "none",
]


@dataclass(frozen=True)
class FinancingEventLabel:
    event_type: FinancingEventType
    event_date: str | None
    source_accession: str | None
    source_form: str | None
    evidence_text: str


def classify_financing_event(text: str) -> FinancingEventType:
    normalized = text.lower()
    if any(term in normalized for term in ("discontinued", "terminate program", "strategic review")):
        return "program_discontinuation"
    if any(term in normalized for term in ("license agreement", "collaboration", "upfront", "royalty")):
        return "partnership"
    if any(term in normalized for term in ("pipe", "warrants", "discount", "at-the-market")):
        return "distressed_refinancing"
    if any(term in normalized for term in ("424b5", "prospectus supplement", "public offering", "registered direct")):
        return "clean_refinancing"
    return "none"


def label_financing_events(filings: list[dict[str, Any]]) -> list[FinancingEventLabel]:
    labels: list[FinancingEventLabel] = []
    for filing in filings:
        text = " ".join(
            str(filing.get(key, ""))
            for key in ("form", "primaryDocDescription", "text", "description")
        )
        event_type = classify_financing_event(text)
        if event_type == "none":
            continue
        labels.append(
            FinancingEventLabel(
                event_type=event_type,
                event_date=filing.get("filingDate"),
                source_accession=filing.get("accessionNumber"),
                source_form=filing.get("form"),
                evidence_text=text.strip(),
            )
        )
    return labels
