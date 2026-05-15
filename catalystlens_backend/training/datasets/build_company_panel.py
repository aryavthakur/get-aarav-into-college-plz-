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
    burn_source_period: str | None = None
    burn_derivation_warning: str | None = None


@dataclass(frozen=True)
class DerivedBurnFact:
    fiscal_year: int
    fiscal_period: str
    end: str
    quarterly_burn: float | None
    source_value: float
    warning: str | None = None


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


_QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3}


def derive_quarterly_operating_cash_burn(facts: list[dict[str, Any]]) -> list[DerivedBurnFact]:
    """Derive quarterly operating cash burn from SEC YTD operating cash flow facts."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for fact in facts:
        fp = fact.get("fp")
        fy = fact.get("fy")
        if fp not in _QUARTER_ORDER or fy is None or fact.get("val") is None:
            continue
        grouped.setdefault(int(fy), []).append(fact)

    derived: list[DerivedBurnFact] = []
    for fiscal_year, year_facts in grouped.items():
        by_period = {
            fact["fp"]: fact
            for fact in sorted(year_facts, key=lambda f: (f.get("end", ""), _QUARTER_ORDER[f["fp"]]))
        }
        previous_ytd: float | None = None
        for period in ("Q1", "Q2", "Q3"):
            fact = by_period.get(period)
            if fact is None:
                continue
            ytd_value = float(fact["val"])
            warning = None
            if period == "Q1":
                quarterly_value = ytd_value
            elif previous_ytd is None:
                quarterly_value = None
                warning = f"Missing prior quarter for {fiscal_year}-{period}; cannot derive quarterly burn."
            else:
                quarterly_value = ytd_value - previous_ytd
            previous_ytd = ytd_value
            derived.append(
                DerivedBurnFact(
                    fiscal_year=fiscal_year,
                    fiscal_period=period,
                    end=fact["end"],
                    quarterly_burn=abs(quarterly_value) if quarterly_value is not None else None,
                    source_value=ytd_value,
                    warning=warning,
                )
            )
    return sorted(derived, key=lambda item: item.end)


def _phase_from_ctgov(phases: list[str] | None) -> str | None:
    if not phases:
        return None
    phase = phases[0].lower().replace("phase", "phase_")
    return phase.replace("__", "_")


def _status_from_ctgov(status: str | None) -> str | None:
    if status is None:
        return None
    normalized = status.lower().replace(" ", "_")
    status_map = {
        "terminated": "terminated",
        "enrolling_by_invitation": "enrolling_by_invitation",
        "unknown": "unknown",
        "not_yet_recruiting": "not_yet_recruiting",
        "recruiting": "recruiting",
        "active_not_recruiting": "active_not_recruiting",
        "completed": "completed",
        "suspended": "suspended",
        "withdrawn": "withdrawn",
        "available": "available",
        "no_longer_available": "no_longer_available",
        "temporarily_not_available": "temporarily_not_available",
        "approved_for_marketing": "approved_for_marketing",
        "withheld": "withheld",
    }
    return status_map.get(normalized, "unknown")


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
    burn_facts = _extract_usd_fact(sec_companyfacts, "NetCashProvidedByUsedInOperatingActivities")
    derived_burns = derive_quarterly_operating_cash_burn(burn_facts)
    if cash_fact is None:
        return []

    protocol = ctgov_study.get("protocolSection", {})
    status_module = protocol.get("statusModule", {})
    design_module = protocol.get("designModule", {})
    primary_completion = status_module.get("primaryCompletionDateStruct", {})

    quarter_end = cash_fact["end"]
    matching_burns = [item for item in derived_burns if item.end <= quarter_end]
    burn_fact = matching_burns[-1] if matching_burns else None
    spread_series = fred_observations.get("BAMLH0A0HYM2", {})

    return [
        CompanyQuarterFeatureRow(
            cik=cik,
            ticker=ticker,
            nct_id=nct_id,
            quarter_end=quarter_end,
            cash_and_equivalents=float(cash_fact["val"]) if cash_fact.get("val") is not None else None,
            quarterly_operating_cash_burn=burn_fact.quarterly_burn if burn_fact else None,
            trial_phase=_phase_from_ctgov(design_module.get("phases")),
            trial_status=_status_from_ctgov(status_module.get("overallStatus")),
            enrollment_target=design_module.get("enrollmentInfo", {}).get("count"),
            primary_completion_date=primary_completion.get("date"),
            high_yield_spread=_market_value_on_or_before(spread_series, quarter_end),
            burn_source_period=(
                f"{burn_fact.fiscal_year}-{burn_fact.fiscal_period}" if burn_fact else None
            ),
            burn_derivation_warning=burn_fact.warning if burn_fact else None,
        )
    ]
