"""Build point-in-time company-quarter feature rows from official-source payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompanyQuarterFeatureRow:
    cik: str
    ticker: str
    nct_id: str
    quarter_end: str
    cash_and_equivalents: float | None
    quarterly_operating_cash_burn: float | None
    trial_phase: str | None
    trial_status: str | None
    enrollment_target: int | None
    primary_completion_date: str | None
    high_yield_spread: float | None


def _extract_usd_fact(companyfacts: dict[str, Any], tag: str) -> list[dict[str, Any]]:
    return (
        companyfacts.get("facts", {})
        .get("us-gaap", {})
        .get(tag, {})
        .get("units", {})
        .get("USD", [])
    )


def _latest_quarterly_fact(companyfacts: dict[str, Any], tag: str) -> dict[str, Any] | None:
    facts = [
        fact
        for fact in _extract_usd_fact(companyfacts, tag)
        if str(fact.get("fp", "")).startswith("Q") and fact.get("end")
    ]
    if not facts:
        return None
    return sorted(facts, key=lambda f: f["end"])[-1]


def _phase_from_ctgov(phases: list[str] | None) -> str | None:
    if not phases:
        return None
    phase = phases[0].lower().replace("phase", "phase_")
    return phase.replace("__", "_")


def _status_from_ctgov(status: str | None) -> str | None:
    if status is None:
        return None
    return status.lower().replace(" ", "_")


def _market_value_on_or_before(
    observations_by_date: dict[str, float],
    target_date: str,
) -> float | None:
    valid_dates = [date for date in observations_by_date if date <= target_date]
    if not valid_dates:
        return None
    return observations_by_date[sorted(valid_dates)[-1]]


def build_company_quarter_panel(
    cik: str,
    ticker: str,
    nct_id: str,
    sec_companyfacts: dict[str, Any],
    ctgov_study: dict[str, Any],
    fred_observations: dict[str, dict[str, float]],
) -> list[CompanyQuarterFeatureRow]:
    cash_fact = _latest_quarterly_fact(
        sec_companyfacts,
        "CashAndCashEquivalentsAtCarryingValue",
    )
    burn_fact = _latest_quarterly_fact(
        sec_companyfacts,
        "NetCashProvidedByUsedInOperatingActivities",
    )
    if cash_fact is None:
        return []

    protocol = ctgov_study.get("protocolSection", {})
    status_module = protocol.get("statusModule", {})
    design_module = protocol.get("designModule", {})
    primary_completion = status_module.get("primaryCompletionDateStruct", {})

    quarter_end = cash_fact["end"]
    burn_value = burn_fact.get("val") if burn_fact else None
    spread_series = fred_observations.get("BAMLH0A0HYM2", {})

    return [
        CompanyQuarterFeatureRow(
            cik=cik,
            ticker=ticker,
            nct_id=nct_id,
            quarter_end=quarter_end,
            cash_and_equivalents=float(cash_fact["val"]) if cash_fact.get("val") is not None else None,
            quarterly_operating_cash_burn=abs(float(burn_value)) if burn_value is not None else None,
            trial_phase=_phase_from_ctgov(design_module.get("phases")),
            trial_status=_status_from_ctgov(status_module.get("overallStatus")),
            enrollment_target=design_module.get("enrollmentInfo", {}).get("count"),
            primary_completion_date=primary_completion.get("date"),
            high_yield_spread=_market_value_on_or_before(spread_series, quarter_end),
        )
    ]
