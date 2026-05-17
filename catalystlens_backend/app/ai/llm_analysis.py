"""
LLM-assisted biotech context analysis for CatalystLens diligence support.

GUARDRAIL: This module provides source-grounded extraction and explanation only.
It must never overwrite CatalystLens core model probabilities, valuation,
posterior PoS, financing probabilities, or investment conclusions.
No investment advice is generated or implied.
"""

from __future__ import annotations

from app.ai.llm_client import call_ai

_DILIGENCE_PROMPT = """\
You are a biotech-finance diligence assistant for CatalystLens.

Your job is to analyze source text from SEC filings, press releases, \
clinical-trial updates, or company disclosures.

Rules:
- Extract only source-supported claims.
- Do not provide investment advice.
- Do not say buy, sell, hold, long, short, or trade.
- Do not change CatalystLens model probabilities.
- Do not claim access to nonpublic information.
- Clearly label uncertainty.
- Identify the diligence question that would most reduce uncertainty.
- Focus on capital-to-catalyst alignment, runway, financing pressure, \
clinical timing, disclosure consistency, and program discontinuation risk.

Return these sections:
1. Runway claims
2. Catalyst timing claims
3. Financing event claims
4. Program discontinuation or safety-risk claims
5. Narrative consistency concerns
6. Most important diligence question
7. Human-review notes\
"""


async def analyze_biotech_context(
    company_name: str,
    ticker: str,
    filing_or_press_text: str,
    question: str | None = None,
) -> dict:
    """
    Analyze a biotech SEC filing or press release for diligence-relevant claims.

    Returns source-grounded extraction only. Does not overwrite model
    probabilities or provide investment advice.

    Returns:
        {
            "analysis": str,
            "method_status": "llm_assisted_source_review",
            "probability_override": False,
            "investment_advice": False,
        }
    """
    question_section = (
        f"\n\nSpecific diligence question: {question}" if question else ""
    )
    prompt = (
        f"{_DILIGENCE_PROMPT}\n\n"
        f"Company: {company_name} ({ticker})\n\n"
        f"Source text:\n{filing_or_press_text}"
        f"{question_section}"
    )

    analysis = await call_ai(prompt)

    return {
        "analysis": analysis,
        "method_status": "llm_assisted_source_review",
        "probability_override": False,
        "investment_advice": False,
    }
