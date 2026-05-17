"""
Benchmark 9 — Mocked LLM extraction simulation.

Monkeypatches app.ai.llm_claim_extraction.call_ai so no real provider
is called. Tests both a valid JSON response and an invalid non-JSON
response. No network calls, no API key required.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.ai.llm_claim_extraction as lce


_VALID_JSON = json.dumps({
    "runway_claim": "cash runway into the first half of 2027",
    "normalized_runway_date": "H1 2027",
    "catalyst_claim": "topline Phase 2 results expected in Q4 2026",
    "normalized_catalyst_date": "Q4 2026",
    "financing_event_claim": "entered into an ATM facility",
    "financing_event_type": "atm",
    "program_discontinuation_claim": None,
    "safety_or_clinical_hold_claim": None,
    "evidence_spans": [
        "cash runway into the first half of 2027",
        "topline Phase 2 results expected in Q4 2026",
        "entered into an ATM facility",
    ],
    "confidence": 0.86,
    "requires_human_review": False,
    "method_status": "llm_assisted_claim_extraction",
    "source_url": None,
})

_INVALID_RESPONSE = "I cannot extract structured claims from this text. No JSON available."


def main() -> None:
    print("Running mocked LLM extraction smoke test...")

    # ------------------------------------------------------------------
    # Part 1: valid JSON response
    # ------------------------------------------------------------------
    print("\n--- Part 1: valid JSON response ---")

    original_call_ai = lce.call_ai

    async def _mock_valid(prompt, timeout=60.0):
        return _VALID_JSON

    lce.call_ai = _mock_valid
    try:
        result = asyncio.run(lce.llm_extract_claims(
            text="Cash runway into the first half of 2027. Topline Phase 2 results expected in Q4 2026.",
            source_url="https://example.com/filing.htm",
        ))
    finally:
        lce.call_ai = original_call_ai

    print(f"  method_status:          {result['method_status']}")
    print(f"  financing_event_type:   {result['financing_event_type']}")
    print(f"  confidence:             {result['confidence']}")
    print(f"  requires_human_review:  {result['requires_human_review']}")
    print(f"  source_url:             {result['source_url']}")

    assert result["method_status"] == "llm_assisted_claim_extraction", (
        f"Expected llm_assisted_claim_extraction, got {result['method_status']}"
    )
    assert result["financing_event_type"] == "atm", (
        f"Expected atm, got {result['financing_event_type']}"
    )
    assert abs(result["confidence"] - 0.86) < 1e-6, (
        f"Expected confidence 0.86, got {result['confidence']}"
    )
    assert result["requires_human_review"] is False, (
        "Expected requires_human_review=False for high-confidence extraction"
    )
    # source_url from function arg must always overwrite model value
    assert result["source_url"] == "https://example.com/filing.htm", (
        f"source_url not set from function arg: {result['source_url']}"
    )
    print("  PASSED")

    # ------------------------------------------------------------------
    # Part 2: invalid non-JSON response
    # ------------------------------------------------------------------
    print("\n--- Part 2: invalid non-JSON response ---")

    async def _mock_invalid(prompt, timeout=60.0):
        return _INVALID_RESPONSE

    lce.call_ai = _mock_invalid
    try:
        error_result = asyncio.run(lce.llm_extract_claims(
            text="Some opaque filing text.",
            source_url="https://example.com/other.htm",
        ))
    finally:
        lce.call_ai = original_call_ai

    print(f"  parse_error:            {error_result.get('parse_error')}")
    print(f"  requires_human_review:  {error_result['requires_human_review']}")
    print(f"  method_status:          {error_result['method_status']}")
    print(f"  source_url:             {error_result.get('source_url')}")

    assert error_result.get("parse_error") is True, "Expected parse_error=True"
    assert error_result["requires_human_review"] is True, "Expected requires_human_review=True on parse failure"
    assert error_result["method_status"] == "llm_assisted_claim_extraction_parse_failed", (
        f"Unexpected method_status: {error_result['method_status']}"
    )
    assert error_result.get("source_url") == "https://example.com/other.htm", (
        "source_url must be preserved in parse-failure result"
    )
    print("  PASSED")

    print("\nAll assertions passed.")
    print("smoke_llm_mocked.py: OK")


if __name__ == "__main__":
    main()
