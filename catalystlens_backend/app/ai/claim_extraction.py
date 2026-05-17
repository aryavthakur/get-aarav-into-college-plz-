"""Rule-based claim extraction scaffolds for filings and press releases."""

from __future__ import annotations

import re

from app.ai.schemas import AIExtractionResult


def _result(
    text: str,
    extraction_type: str,
    extracted: str | None,
    normalized: str | None,
    confidence: float,
) -> AIExtractionResult:
    return AIExtractionResult(
        raw_text=text,
        extracted_value=extracted,
        normalized_value=normalized,
        extraction_type=extraction_type,
        confidence=confidence,
        evidence_span=extracted,
        requires_human_review=True,
    )


def extract_runway_guidance(text: str) -> AIExtractionResult:
    lowered = text.lower()
    patterns = [
        r"(?:funded|cash runway|runway)\s+(?:into|through|until)\s+((?:q[1-4]\s+)?20\d{2})",
        r"(?:funded|cash runway|runway)\s+(?:into|through|until)\s+((?:first|second|third|fourth)\s+quarter\s+20\d{2})",
        r"(?:sufficient to )?fund operations\s+(?:into|through|until)\s+((?:q[1-4]\s+)?20\d{2})",
        r"fund operating expenses and capital expenditure requirements\s+(?:through|until)\s+(?:at least\s+)?(?:the\s+)?((?:first|second|third|fourth)\s+quarter\s+of\s+20\d{2})",
        r"cash runway extends\s+(?:into|through|until)\s+(?:the\s+)?((?:first|second)\s+half\s+of\s+20\d{2})",
        r"expected to fund operations\s+(?:into|through|until)\s+(20\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            value = match.group(1).upper().replace("  ", " ")
            year_match = re.search(r"20\d{2}", value)
            return _result(text, "runway_guidance", match.group(0), year_match.group(0) if year_match else value, 0.75)
    return _result(text, "runway_guidance", None, None, 0.1)


def extract_catalyst_guidance(text: str) -> AIExtractionResult:
    lowered = text.lower()
    patterns = [
        r"((?:topline|interim|data|readout|topline results|data readout)[^.]{0,60}?(?:expected|anticipated|planned)[^.]{0,60}?(?:h[12]|q[1-4]|20\d{2})[^.]*)",
        r"((?:primary completion)[^.]{0,60}?(?:expected|anticipated|planned)[^.]{0,60}?(?:h[12]|q[1-4]|20\d{2})[^.]*)",
        r"((?:pdufa date)[^.]{0,80}?(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+20\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            span = match.group(1)
            normalized_match = re.search(
                r"(h[12]\s+20\d{2}|q[1-4]\s+20\d{2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+20\d{2}|20\d{2})",
                span,
                flags=re.IGNORECASE,
            )
            return _result(text, "catalyst_guidance", span, normalized_match.group(1).upper() if normalized_match else span, 0.7)
    return _result(text, "catalyst_guidance", None, None, 0.1)


def extract_financing_event(text: str) -> AIExtractionResult:
    patterns = [
        r"(?:announced|closed|completed)?\s*(\$[0-9,.]+\s*(?:million|billion)[^.]{0,50}?(?:public offering|private placement|pipe|registered direct|debt|royalty|partnership))",
        r"(gross proceeds of approximately\s+\$[0-9,.]+\s*(?:million|billion))",
        r"(entered into a private placement[^.]*)",
        r"((?:established|entered into|launched)?\s*(?:an?\s+)?atm facility[^.]*)",
        r"((?:completed|closed|announced)?\s*debt financing[^.]*)",
        r"(upfront payment under collaboration agreement[^.]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = " ".join(match.group(1).split())
            return _result(text, "financing_event", value, value, 0.75)
    return _result(text, "financing_event", None, None, 0.1)


def extract_program_discontinuation(text: str) -> AIExtractionResult:
    lowered = text.lower()
    negative_restructuring_context = (
        "finance department",
        "lease obligation",
        "lease obligations",
        "administrative operation",
        "administrative operations",
    )
    if "strategic restructuring" in lowered and not any(
        term in lowered for term in negative_restructuring_context
    ):
        match = re.search(r"(strategic restructuring[^.]*)", lowered, flags=re.IGNORECASE)
        if match:
            return _result(text, "program_discontinuation", match.group(1), "program_discontinuation", 0.65)

    patterns = [
        r"(discontinue development of [^.]+)",
        r"(strategic restructuring[^.]{0,120}?(?:discontinue development of|pipeline prioritization|pause(?:d)? (?:of )?enrollment|termination of the study|terminate(?:d)? the study)[^.]*)",
        r"(paus(?:e|ed) enrollment(?:\s+(?:in|of|for)\s+[^.]+)?)",
        r"(terminate(?:d)? the study(?:\s+(?:of|for|in)\s+[^.]+)?)",
        r"(terminate(?:d)? development of [^.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return _result(text, "program_discontinuation", match.group(1), "program_discontinuation", 0.8)
    return _result(text, "program_discontinuation", None, None, 0.1)
