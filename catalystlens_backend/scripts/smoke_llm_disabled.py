"""
Benchmark 8 — LLM route smoke test without API keys.

Clears cloud API keys, then exercises /lambda-health and /audit with
use_llm_source_review=False. No cloud calls, no API key required.
"""

import json
import os
import sys

# Clear cloud keys before any imports that read os.environ at module level
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)
os.environ["GROQ_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app


def main() -> None:
    print("Running LLM-disabled smoke test (no API keys)...")

    with TestClient(app) as client:
        # ------------------------------------------------------------------
        # 1. /lambda-health — should report unavailable or local ollama
        # ------------------------------------------------------------------
        resp = client.get("/lambda-health")
        assert resp.status_code == 200, f"/lambda-health returned {resp.status_code}"
        health = resp.json()
        print(f"\n/lambda-health response: {health}")
        assert health["status"] in ("unavailable", "ok"), (
            f"Unexpected status: {health['status']}"
        )
        if health["status"] == "ok":
            assert health["provider"] == "ollama", (
                f"Expected ollama when no cloud keys set, got: {health['provider']}"
            )
        else:
            assert health["provider"] is None, (
                f"Expected provider=None when unavailable, got: {health['provider']}"
            )

        # No API key must appear anywhere in the response
        resp_text = json.dumps(health)
        for env_var in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
            key_val = os.environ.get(env_var, "")
            if key_val:
                assert key_val not in resp_text, f"API key leaked in /lambda-health response"

        # ------------------------------------------------------------------
        # 2. /audit with use_llm_source_review=False
        # ------------------------------------------------------------------
        audit_payload = {
            "financial": {
                "company_name": "LLMOffBio",
                "ticker": "LLMB",
                "cash_on_hand": 90_000_000,
                "marketable_securities": 5_000_000,
                "quarterly_operating_cash_burn": 14_000_000,
                "market_cap": 420_000_000,
                "debt": 0,
                "going_concern_flag": False,
                "biotech_market_condition_score": 6.0,
            },
            "clinical": {
                "asset_name": "LLMB-101",
                "indication": "Autoimmune",
                "trial_phase": "phase_2",
                "trial_status": "recruiting",
                "stated_months_to_catalyst": 16,
                "enrollment_target": 100,
                "enrollment_completed": 50,
                "enrollment_rate_per_month": 7,
                "number_of_sites": 10,
                "catalyst_type": "primary_readout",
            },
            "success_probability": {
                "trial_phase": "phase_2",
                "positive_signals": [],
                "negative_signals": [],
            },
            "valuation": {"asset_value_success": 500_000_000},
            "disclosure": {
                "company_narrative_distribution": {
                    "runway_strength": 0.7,
                    "clinical_timeline_confidence": 0.7,
                    "dilution_risk": 0.3,
                },
                "structured_audit_distribution": {
                    "runway_strength": 0.6,
                    "clinical_timeline_confidence": 0.6,
                    "dilution_risk": 0.4,
                },
            },
            "simulation": {
                "n_simulations": 300,
                "random_seed": 99,
                "use_llm_source_review": False,
            },
        }

        resp = client.post("/audit", json=audit_payload)
        assert resp.status_code == 200, f"/audit returned {resp.status_code}: {resp.text[:300]}"
        audit_data = resp.json()

        print(f"\n/audit company_name: {audit_data['company_name']}")
        print(f"/audit llm_source_review: {audit_data.get('llm_source_review')}")
        p = audit_data["capital_to_catalyst"]["probability_cashout_before_catalyst"]
        print(f"/audit probability_cashout_before_catalyst: {p:.4f}")

        assert audit_data.get("llm_source_review") is None, (
            "llm_source_review must be None when use_llm_source_review=False"
        )
        assert 0.0 <= p <= 1.0, f"probability out of range: {p}"

        # No API key in audit response
        audit_text = json.dumps(audit_data)
        for env_var in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
            key_val = os.environ.get(env_var, "")
            if key_val:
                assert key_val not in audit_text, f"API key leaked in /audit response"

    print("\nAll assertions passed.")
    print("smoke_llm_disabled.py: OK")


if __name__ == "__main__":
    main()
