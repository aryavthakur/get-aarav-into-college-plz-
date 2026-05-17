"""
Tests for the LLM integration layer: llm_client, llm_analysis, llm_claim_extraction,
and the three new FastAPI routes.

All tests use mocks. No real Groq, OpenRouter, or Ollama calls are made.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockResponse:
    """Minimal httpx.Response mock."""

    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data

    def json(self) -> dict:
        return self._data


def _openai_ok(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _make_mock_client(post_map: dict, get_map: dict | None = None):
    """
    Return a mock AsyncClient class whose post/get methods dispatch by
    URL substring.

    post_map / get_map: {url_substring: (status_code, response_dict)}
    """
    post_map = post_map or {}
    get_map = get_map or {}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url: str, **kwargs):
            for key, (code, data) in post_map.items():
                if key in url:
                    return _MockResponse(code, data)
            return _MockResponse(503, {})

        async def get(self, url: str, **kwargs):
            for key, (code, data) in get_map.items():
                if key in url:
                    return _MockResponse(code, data)
            return _MockResponse(503, {})

    return _Client


# ---------------------------------------------------------------------------
# TestAiHealth
# ---------------------------------------------------------------------------

class TestAiHealth:
    async def test_returns_groq_when_key_set(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "fake-groq-key")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")
        result = await lc.ai_health()
        assert result == {"status": "ok", "provider": "groq"}

    async def test_returns_openrouter_when_only_openrouter_set(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "fake-or-key")
        result = await lc.ai_health()
        assert result == {"status": "ok", "provider": "openrouter"}

    async def test_returns_unavailable_when_no_provider(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")
        # Mock Ollama as unreachable
        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(post_map={}, get_map={"/api/tags": (503, {})}),
        )
        result = await lc.ai_health()
        assert result == {"status": "unavailable"}

    async def test_returns_ollama_when_reachable(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")
        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(post_map={}, get_map={"/api/tags": (200, {"models": []})}),
        )
        result = await lc.ai_health()
        assert result == {"status": "ok", "provider": "ollama"}

    async def test_returns_unavailable_when_ollama_connection_error(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")

        class _ErrorClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, **kw):
                raise httpx.ConnectError("refused")

        monkeypatch.setattr(lc.httpx, "AsyncClient", _ErrorClient)
        result = await lc.ai_health()
        assert result == {"status": "unavailable"}


# ---------------------------------------------------------------------------
# TestCallAIFallback
# ---------------------------------------------------------------------------

class TestCallAIFallback:
    async def test_groq_429_falls_through_to_openrouter(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "fake-groq")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "fake-or")

        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(
                post_map={
                    "groq.com": (429, {}),
                    "openrouter.ai": (200, _openai_ok("OpenRouter result")),
                }
            ),
        )
        result = await lc.call_ai("test prompt")
        assert result == "OpenRouter result"

    async def test_openrouter_all_429_falls_to_ollama(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "fake-or")

        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(
                post_map={
                    "openrouter.ai": (429, {}),
                    "localhost:11434": (200, {"response": "Ollama result"}),
                }
            ),
        )
        result = await lc.call_ai("test prompt")
        assert result == "Ollama result"

    async def test_groq_200_returns_immediately(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "fake-groq")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")

        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(
                post_map={"groq.com": (200, _openai_ok("Groq result"))}
            ),
        )
        result = await lc.call_ai("test prompt")
        assert result == "Groq result"

    async def test_no_provider_raises_503(self, monkeypatch):
        import app.ai.llm_client as lc
        from fastapi import HTTPException

        monkeypatch.setattr(lc, "GROQ_API_KEY", "")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")

        class _UnreachableClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, **kw):
                raise httpx.ConnectError("refused")

            async def get(self, url, **kw):
                raise httpx.ConnectError("refused")

        monkeypatch.setattr(lc.httpx, "AsyncClient", _UnreachableClient)
        with pytest.raises(HTTPException) as exc_info:
            await lc.call_ai("test prompt")
        assert exc_info.value.status_code == 503

    async def test_groq_502_propagates(self, monkeypatch):
        import app.ai.llm_client as lc
        from fastapi import HTTPException

        monkeypatch.setattr(lc, "GROQ_API_KEY", "fake-groq")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")
        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(post_map={"groq.com": (500, {})}),
        )
        with pytest.raises(HTTPException) as exc_info:
            await lc.call_ai("test prompt")
        assert exc_info.value.status_code == 502

    async def test_no_api_key_in_error_messages(self, monkeypatch):
        """API keys must never appear in exception detail strings."""
        import app.ai.llm_client as lc
        from fastapi import HTTPException

        fake_key = "sk-supersecret-key-12345"
        monkeypatch.setattr(lc, "GROQ_API_KEY", fake_key)
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")
        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(post_map={"groq.com": (500, {})}),
        )
        with pytest.raises(HTTPException) as exc_info:
            await lc.call_ai("test prompt")
        assert fake_key not in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# TestLambdaHealthRoute
# ---------------------------------------------------------------------------

class TestLambdaHealthRoute:
    def test_lambda_health_returns_200(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "fake-groq")
        response = client.get("/lambda-health")
        assert response.status_code == 200

    def test_lambda_health_returns_provider(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "fake-groq")
        data = client.get("/lambda-health").json()
        assert data["status"] == "ok"
        assert data["provider"] == "groq"

    def test_lambda_health_unavailable_when_no_keys(self, monkeypatch):
        import app.ai.llm_client as lc
        monkeypatch.setattr(lc, "GROQ_API_KEY", "")
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")
        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(post_map={}, get_map={"/api/tags": (503, {})}),
        )
        data = client.get("/lambda-health").json()
        assert data["status"] == "unavailable"


# ---------------------------------------------------------------------------
# TestLambdaAnalyzeRoute
# ---------------------------------------------------------------------------

class TestLambdaAnalyzeRoute:
    def test_returns_method_status(self, monkeypatch):
        import app.ai.llm_analysis as la

        async def _mock_call_ai(prompt, timeout=60.0):
            return "Mock analysis output."

        monkeypatch.setattr(la, "call_ai", _mock_call_ai)
        response = client.post(
            "/lambda-analyze",
            json={
                "company_name": "Test Biotech",
                "ticker": "TBTC",
                "text": "Cash and cash equivalents: $42M.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method_status"] == "llm_assisted_source_review"

    def test_returns_investment_advice_false(self, monkeypatch):
        import app.ai.llm_analysis as la

        async def _mock_call_ai(prompt, timeout=60.0):
            return "Analysis text."

        monkeypatch.setattr(la, "call_ai", _mock_call_ai)
        data = client.post(
            "/lambda-analyze",
            json={"company_name": "Co", "ticker": "CO", "text": "Filing text."},
        ).json()
        assert data["investment_advice"] is False

    def test_returns_probability_override_false(self, monkeypatch):
        import app.ai.llm_analysis as la

        async def _mock_call_ai(prompt, timeout=60.0):
            return "Analysis."

        monkeypatch.setattr(la, "call_ai", _mock_call_ai)
        data = client.post(
            "/lambda-analyze",
            json={"company_name": "Co", "ticker": "CO", "text": "text"},
        ).json()
        assert data["probability_override"] is False

    def test_analysis_field_contains_mock_output(self, monkeypatch):
        import app.ai.llm_analysis as la

        async def _mock_call_ai(prompt, timeout=60.0):
            return "Section 1: Runway claims — sufficient to Q3 2026."

        monkeypatch.setattr(la, "call_ai", _mock_call_ai)
        data = client.post(
            "/lambda-analyze",
            json={"company_name": "Co", "ticker": "CO", "text": "text"},
        ).json()
        assert "Runway claims" in data["analysis"]


# ---------------------------------------------------------------------------
# TestExtractClaimsRoute
# ---------------------------------------------------------------------------

class TestExtractClaimsRoute:
    def _valid_extraction_json(self) -> str:
        return json.dumps({
            "runway_claim": "funded into Q3 2026",
            "normalized_runway_date": "Q3 2026",
            "catalyst_claim": "Phase 2 results expected H1 2025",
            "normalized_catalyst_date": "H1 2025",
            "financing_event_claim": None,
            "financing_event_type": None,
            "program_discontinuation_claim": None,
            "safety_or_clinical_hold_claim": None,
            "evidence_spans": ["funded into Q3 2026"],
            "confidence": 0.85,
            "requires_human_review": False,
            "method_status": "llm_assisted_claim_extraction",
            "source_url": None,
        })

    def test_extract_claims_parses_valid_json(self, monkeypatch):
        import app.ai.llm_claim_extraction as lce

        async def _mock_call_ai(prompt, timeout=60.0):
            return self._valid_extraction_json()

        monkeypatch.setattr(lce, "call_ai", _mock_call_ai)
        response = client.post(
            "/ai/extract-claims",
            json={"text": "We are funded into Q3 2026."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["method_status"] == "llm_assisted_claim_extraction"
        assert data["runway_claim"] == "funded into Q3 2026"
        assert data["confidence"] == pytest.approx(0.85)

    def test_extract_claims_handles_invalid_json_safely(self, monkeypatch):
        import app.ai.llm_claim_extraction as lce

        async def _mock_call_ai(prompt, timeout=60.0):
            return "Sorry, I cannot extract that. No JSON here."

        monkeypatch.setattr(lce, "call_ai", _mock_call_ai)
        response = client.post(
            "/ai/extract-claims",
            json={"text": "Some filing text."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["parse_error"] is True
        assert data["requires_human_review"] is True
        assert data["method_status"] == "llm_assisted_claim_extraction_parse_failed"

    def test_extract_claims_handles_malformed_json_safely(self, monkeypatch):
        import app.ai.llm_claim_extraction as lce

        async def _mock_call_ai(prompt, timeout=60.0):
            return '{"runway_claim": "Q3 2026", "confidence": 0.9'  # truncated JSON

        monkeypatch.setattr(lce, "call_ai", _mock_call_ai)
        data = client.post(
            "/ai/extract-claims",
            json={"text": "Filing text."},
        ).json()
        assert data["parse_error"] is True
        assert data["requires_human_review"] is True

    def test_source_url_injected_when_not_in_response(self, monkeypatch):
        import app.ai.llm_claim_extraction as lce
        payload = json.dumps({
            "runway_claim": None, "normalized_runway_date": None,
            "catalyst_claim": None, "normalized_catalyst_date": None,
            "financing_event_claim": None, "financing_event_type": None,
            "program_discontinuation_claim": None, "safety_or_clinical_hold_claim": None,
            "evidence_spans": [], "confidence": 0.75,
            "requires_human_review": False,
            "method_status": "llm_assisted_claim_extraction",
            "source_url": None,
        })

        async def _mock_call_ai(prompt, timeout=60.0):
            return payload

        monkeypatch.setattr(lce, "call_ai", _mock_call_ai)
        data = client.post(
            "/ai/extract-claims",
            json={"text": "text", "source_url": "https://example.com/filing.htm"},
        ).json()
        assert data["source_url"] == "https://example.com/filing.htm"


# ---------------------------------------------------------------------------
# TestLowConfidenceExtraction
# ---------------------------------------------------------------------------

class TestLowConfidenceExtraction:
    def test_low_confidence_forces_human_review(self, monkeypatch):
        import app.ai.llm_claim_extraction as lce

        low_conf = json.dumps({
            "runway_claim": "possibly Q4 2025",
            "normalized_runway_date": None,
            "catalyst_claim": None, "normalized_catalyst_date": None,
            "financing_event_claim": None, "financing_event_type": None,
            "program_discontinuation_claim": None, "safety_or_clinical_hold_claim": None,
            "evidence_spans": [],
            "confidence": 0.50,  # below 0.70 threshold
            "requires_human_review": False,  # model said false — must be overridden
            "method_status": "llm_assisted_claim_extraction",
            "source_url": None,
        })

        async def _mock_call_ai(prompt, timeout=60.0):
            return low_conf

        monkeypatch.setattr(lce, "call_ai", _mock_call_ai)
        data = client.post(
            "/ai/extract-claims",
            json={"text": "Ambiguous filing text."},
        ).json()
        assert data["confidence"] == pytest.approx(0.50)
        assert data["requires_human_review"] is True

    def test_confidence_clamped_above_one(self, monkeypatch):
        import app.ai.llm_claim_extraction as lce

        payload = json.dumps({
            "runway_claim": None, "normalized_runway_date": None,
            "catalyst_claim": None, "normalized_catalyst_date": None,
            "financing_event_claim": None, "financing_event_type": None,
            "program_discontinuation_claim": None, "safety_or_clinical_hold_claim": None,
            "evidence_spans": [],
            "confidence": 1.5,  # over 1.0 — must be clamped
            "requires_human_review": False,
            "method_status": "llm_assisted_claim_extraction",
            "source_url": None,
        })

        async def _mock_call_ai(prompt, timeout=60.0):
            return payload

        monkeypatch.setattr(lce, "call_ai", _mock_call_ai)
        data = client.post(
            "/ai/extract-claims", json={"text": "text"}
        ).json()
        assert data["confidence"] <= 1.0

    def test_confidence_clamped_below_zero(self, monkeypatch):
        import app.ai.llm_claim_extraction as lce

        payload = json.dumps({
            "runway_claim": None, "normalized_runway_date": None,
            "catalyst_claim": None, "normalized_catalyst_date": None,
            "financing_event_claim": None, "financing_event_type": None,
            "program_discontinuation_claim": None, "safety_or_clinical_hold_claim": None,
            "evidence_spans": [],
            "confidence": -0.5,
            "requires_human_review": False,
            "method_status": "llm_assisted_claim_extraction",
            "source_url": None,
        })

        async def _mock_call_ai(prompt, timeout=60.0):
            return payload

        monkeypatch.setattr(lce, "call_ai", _mock_call_ai)
        data = client.post(
            "/ai/extract-claims", json={"text": "text"}
        ).json()
        assert data["confidence"] >= 0.0
        assert data["requires_human_review"] is True


# ---------------------------------------------------------------------------
# TestAuditUnchangedWithoutLLM
# ---------------------------------------------------------------------------

class TestAuditUnchangedWithoutLLM:
    def test_audit_output_unchanged_when_llm_disabled(self):
        """use_llm_source_review=False must not affect audit output."""
        import json as _json
        import os

        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "example_company.json"
        )
        with open(data_path) as f:
            payload = _json.load(f)
        payload["simulation"]["n_simulations"] = 300
        payload["simulation"]["use_llm_source_review"] = False

        response = client.post("/audit", json=payload)
        assert response.status_code == 200
        data = response.json()
        # llm_source_review should be absent or null when disabled
        assert data.get("llm_source_review") is None

    def test_audit_llm_field_absent_by_default(self):
        import json as _json
        import os

        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "example_company.json"
        )
        with open(data_path) as f:
            payload = _json.load(f)
        payload["simulation"]["n_simulations"] = 300

        response = client.post("/audit", json=payload)
        assert response.status_code == 200
        data = response.json()
        # field exists in schema but should be null
        assert data.get("llm_source_review") is None


# ---------------------------------------------------------------------------
# TestNoAPIKeyInOutputs
# ---------------------------------------------------------------------------

class TestNoAPIKeyInOutputs:
    async def test_no_api_key_in_502_detail(self, monkeypatch):
        import app.ai.llm_client as lc
        from fastapi import HTTPException

        secret = "sk-very-secret-key-do-not-leak"
        monkeypatch.setattr(lc, "GROQ_API_KEY", secret)
        monkeypatch.setattr(lc, "OPENROUTER_API_KEY", "")
        monkeypatch.setattr(
            lc.httpx,
            "AsyncClient",
            _make_mock_client(post_map={"groq.com": (500, {})}),
        )
        with pytest.raises(HTTPException) as exc_info:
            await lc.call_ai("test")
        assert secret not in str(exc_info.value.detail)
        assert secret not in repr(exc_info.value)
