"""FRED API client helpers for market covariates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx


FRED_BASE_URL = "https://api.stlouisfed.org/fred"


@dataclass(frozen=True)
class FREDClient:
    api_key: str
    base_url: str = FRED_BASE_URL

    def series_observations_url(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> str:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        return f"{self.base_url}/series/observations?{urlencode(params)}"

    def get_series_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> dict[str, Any]:
        response = httpx.get(
            self.series_observations_url(series_id, observation_start, observation_end),
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
