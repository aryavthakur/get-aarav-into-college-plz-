"""ClinicalTrials.gov API v2 client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


CTGOV_BASE_URL = "https://clinicaltrials.gov"


@dataclass(frozen=True)
class ClinicalTrialsClient:
    base_url: str = CTGOV_BASE_URL

    def study_url(self, nct_id: str) -> str:
        return f"{self.base_url}/api/v2/studies/{nct_id.upper()}"

    def get_study(self, nct_id: str) -> dict[str, Any]:
        response = httpx.get(self.study_url(nct_id), timeout=30.0)
        response.raise_for_status()
        return response.json()
