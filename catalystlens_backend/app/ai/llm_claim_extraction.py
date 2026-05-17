"""
LLM-assisted structured claim extraction from biotech filings.

GUARDRAIL: Extracted claims are for diligence support only. They must not
overwrite CatalystLens core model probabilities or investment conclusions.
All extractions with confidence < 0.70 are automatically flagged for human
review. Parse failures also return a safe error dict with requires_human_review=True.
"""

from __future__ import annotations

import json
import re

from app.ai.llm_client import call_ai

_EXTRACTION_PROMPT_TEMPLATE = """\
You are a structured data extraction assistant for biotech regulatory filings.

Extract the following fields from the provided source text and return ONLY valid JSON \
with no preamble or explanation:

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

Rules:
- Only extract claims directly supported by the source text.
- confidence is a float between 0.0 and 1.0.
- Set requires_human_review=true whenever confidence < 0.70.
- Do not invent claims not present in the source.
- Return valid JSON only — no markdown fences, no explanation.

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
    - confidence < 0.70 forces requires_human_review=True.
    - JSON parse failure returns a safe error dict with requires_human_review=True.
    - source_url is injected into the result if not returned by the model.
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
        }

    try:
        result = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {
            "raw_response": raw,
            "parse_error": True,
            "requires_human_review": True,
            "method_status": "llm_assisted_claim_extraction_parse_failed",
        }

    # Clamp confidence to [0, 1] and enforce human-review flag
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    result["confidence"] = confidence
    if confidence < 0.70:
        result["requires_human_review"] = True

    # Inject source_url if the model left it null
    if source_url and not result.get("source_url"):
        result["source_url"] = source_url

    result["method_status"] = "llm_assisted_claim_extraction"
    return result
