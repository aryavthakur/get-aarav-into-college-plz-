"""
LLM-assisted structured claim extraction from biotech filings.

GUARDRAIL: Extracted claims are for diligence support only. They must not
overwrite CatalystLens core model probabilities or investment conclusions.
All extractions with confidence < 0.70 are automatically flagged for human
review. Parse failures return a safe error dict with requires_human_review=True.
"""

from __future__ import annotations

import json
import re

from app.ai.llm_client import call_ai

_REQUIRED_KEYS = [
    "runway_claim",
    "normalized_runway_date",
    "catalyst_claim",
    "normalized_catalyst_date",
    "financing_event_claim",
    "financing_event_type",
    "program_discontinuation_claim",
    "safety_or_clinical_hold_claim",
    "evidence_spans",
    "confidence",
    "requires_human_review",
    "method_status",
    "source_url",
]

_EXTRACTION_PROMPT_TEMPLATE = """\
You are extracting source-grounded biotech-finance claims for CatalystLens.

Rules:
- Return valid JSON only.
- Extract only claims explicitly supported by the text.
- Do not infer hidden facts.
- Do not provide investment advice.
- Do not estimate final model probabilities.
- Preserve evidence spans from the source text.
- If uncertain, set requires_human_review to true.

Return exactly this JSON shape:

{{
  "runway_claim": "verbatim quote or null",
  "normalized_runway_date": "e.g. Q3 2025 or null",
  "catalyst_claim": "verbatim quote or null",
  "normalized_catalyst_date": "e.g. H2 2025 or null",
  "financing_event_claim": "verbatim quote or null",
  "financing_event_type": "clean_refi | distressed_refi | partnership | debt_or_royalty | atm | unknown | null",
  "program_discontinuation_claim": "verbatim quote or null",
  "safety_or_clinical_hold_claim": "verbatim quote or null",
  "evidence_spans": ["short quote 1", "short quote 2"],
  "confidence": 0.0,
  "requires_human_review": true,
  "method_status": "llm_assisted_claim_extraction",
  "source_url": null
}}

No markdown fences. No explanation. Valid JSON only.

Source text:
{text}\
"""


async def llm_extract_claims(
    text: str,
    source_url: str | None = None,
) -> dict:
    """
    Use LLM to extract structured claims from a biotech source text.

    - Returns a dict matching the extraction schema.
    - Missing required keys are filled with null / safe defaults.
    - confidence < 0.70 forces requires_human_review=True.
    - source_url from the function argument always overwrites the model value.
    - JSON parse failure returns a safe error dict with requires_human_review=True.
    """
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(text=text)
    if source_url:
        prompt += f"\n\nSource URL: {source_url}"

    raw = await call_ai(prompt)

    # Robustly extract JSON between first { and last }
    json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not json_match:
        return {
            "raw_response": raw,
            "parse_error": True,
            "requires_human_review": True,
            "method_status": "llm_assisted_claim_extraction_parse_failed",
            "source_url": source_url,
        }

    try:
        result = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {
            "raw_response": raw,
            "parse_error": True,
            "requires_human_review": True,
            "method_status": "llm_assisted_claim_extraction_parse_failed",
            "source_url": source_url,
        }

    # Fill any missing required keys with null / safe defaults
    for key in _REQUIRED_KEYS:
        if key not in result:
            result[key] = [] if key == "evidence_spans" else None

    # Clamp confidence to [0, 1] and enforce human-review flag
    confidence = max(0.0, min(1.0, float(result.get("confidence") or 0.0)))
    result["confidence"] = confidence
    if confidence < 0.70:
        result["requires_human_review"] = True

    # Always overwrite method_status and source_url from authoritative sources
    result["method_status"] = "llm_assisted_claim_extraction"
    result["source_url"] = source_url

    return result
