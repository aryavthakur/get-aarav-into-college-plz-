"""
Async LLM provider client with Groq → OpenRouter → Ollama fallback chain.

GUARDRAIL: This client is for source-grounded extraction and diligence support
only. It must never overwrite CatalystLens core model probabilities, valuation,
posterior PoS, financing probabilities, or investment conclusions.

Provider priority:
1. Groq  (GROQ_API_KEY)
2. OpenRouter  (OPENROUTER_API_KEY) — tries three free models in sequence
3. Ollama  (OLLAMA_URL, default http://localhost:11434)

Rate-limit (429) causes fallthrough to the next provider / next model.
Other non-200 HTTP responses raise HTTPException(502).
No provider available raises HTTPException(503).
API keys are never logged or returned in responses.
"""

from __future__ import annotations

import os

import httpx
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Configuration — populated from environment at import time
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")

_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"

_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _openai_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2048,
    }


def _extract_openai_text(data: dict) -> str:
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def call_ai(prompt: str, timeout: float = 60.0) -> str:
    """
    Call the first available LLM provider in order: Groq → OpenRouter → Ollama.

    - 429 from any provider causes fallthrough to the next provider/model.
    - Other non-200 responses raise HTTPException(502).
    - Connection errors cause fallthrough (provider unavailable).
    - If no provider succeeds, raises HTTPException(503).
    """
    async with httpx.AsyncClient(timeout=timeout) as client:

        # ------------------------------------------------------------------
        # 1. Groq
        # ------------------------------------------------------------------
        if GROQ_API_KEY:
            try:
                resp = await client.post(
                    _GROQ_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=_openai_payload(_GROQ_MODEL, prompt),
                )
                if resp.status_code == 200:
                    return _extract_openai_text(resp.json())
                if resp.status_code != 429:
                    raise HTTPException(502, f"Groq returned HTTP {resp.status_code}")
                # 429 → fall through to OpenRouter
            except HTTPException:
                raise
            except httpx.HTTPError:
                pass  # connection error — try next provider

        # ------------------------------------------------------------------
        # 2. OpenRouter — tries models in order; 429 tries next model
        # ------------------------------------------------------------------
        if OPENROUTER_API_KEY:
            for model in _OPENROUTER_MODELS:
                try:
                    resp = await client.post(
                        _OPENROUTER_ENDPOINT,
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=_openai_payload(model, prompt),
                    )
                    if resp.status_code == 200:
                        return _extract_openai_text(resp.json())
                    if resp.status_code == 429:
                        continue  # try next model
                    raise HTTPException(502, f"OpenRouter returned HTTP {resp.status_code}")
                except HTTPException:
                    raise
                except httpx.HTTPError:
                    break  # connection error — skip remaining OR models

        # ------------------------------------------------------------------
        # 3. Ollama (local)
        # ------------------------------------------------------------------
        try:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False},
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            raise HTTPException(502, f"Ollama returned HTTP {resp.status_code}")
        except HTTPException:
            raise
        except httpx.HTTPError:
            pass  # Ollama not reachable

    raise HTTPException(503, "No AI provider available.")


async def ai_health() -> dict:
    """
    Return the first configured / reachable AI provider.

    Returns:
        {"status": "ok", "provider": "groq"}        — if GROQ_API_KEY is set
        {"status": "ok", "provider": "openrouter"}   — if OPENROUTER_API_KEY is set
        {"status": "ok", "provider": "ollama"}       — if Ollama /api/tags responds 200
        {"status": "unavailable"}                    — nothing reachable
    """
    if GROQ_API_KEY:
        return {"status": "ok", "provider": "groq"}
    if OPENROUTER_API_KEY:
        return {"status": "ok", "provider": "openrouter"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            if resp.status_code == 200:
                return {"status": "ok", "provider": "ollama"}
    except httpx.HTTPError:
        pass
    return {"status": "unavailable"}
