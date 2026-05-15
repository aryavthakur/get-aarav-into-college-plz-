"""
SEC EDGAR API client helpers.

The SEC requires descriptive User-Agent headers and fair-access behavior. This
client intentionally exposes URL builders and bounded request configuration so
ingestion jobs can remain auditable and polite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


SEC_DATA_BASE_URL = "https://data.sec.gov"


@dataclass(frozen=True)
class SECClient:
    user_agent: str
    requests_per_second: int = 10
    base_url: str = SEC_DATA_BASE_URL

    def __post_init__(self) -> None:
        if not self.user_agent.strip():
            raise ValueError("SEC API requests require a descriptive User-Agent")
        object.__setattr__(self, "requests_per_second", min(max(1, self.requests_per_second), 10))

    @staticmethod
    def normalize_cik(cik: str | int) -> str:
        digits = "".join(ch for ch in str(cik) if ch.isdigit())
        return digits.zfill(10)

    def submissions_url(self, cik: str | int) -> str:
        return f"{self.base_url}/submissions/CIK{self.normalize_cik(cik)}.json"

    def companyfacts_url(self, cik: str | int) -> str:
        return f"{self.base_url}/api/xbrl/companyfacts/CIK{self.normalize_cik(cik)}.json"

    def get_json(self, url: str) -> dict[str, Any]:
        response = httpx.get(url, headers={"User-Agent": self.user_agent}, timeout=30.0)
        response.raise_for_status()
        return response.json()

    def get_submissions(self, cik: str | int) -> dict[str, Any]:
        return self.get_json(self.submissions_url(cik))

    def get_companyfacts(self, cik: str | int) -> dict[str, Any]:
        return self.get_json(self.companyfacts_url(cik))
