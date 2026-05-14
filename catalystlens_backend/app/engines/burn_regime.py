"""
Burn Regime Engine — Change Point Detection.

Detects structural shifts in a biotech company's quarterly operating cash burn
using PELT (Pruned Exact Linear Time) change-point detection.

Falls back to a simple heuristic method if the `ruptures` library is not
installed.

Burn acceleration feeds directly into the Cox solvency risk multiplier.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from app.core.config import BurnRegimeThresholds, CatalystLensConfig, get_default_config
from app.models.schemas import BurnRegimeResult, CompanyFinancialInput


# ---------------------------------------------------------------------------
# Change-point detection
# ---------------------------------------------------------------------------

def detect_burn_changepoints(burn_series: List[float], min_size: int = 2) -> List[int]:
    """
    Detect change points in a burn series using PELT (ruptures library).

    Falls back to a simple variance-split method when ruptures is unavailable.

    Returns a list of 0-based indices *before* which a regime shift occurs.
    """
    n = len(burn_series)
    if n < 3:
        return []

    signal = np.array(burn_series, dtype=float).reshape(-1, 1)

    try:
        import ruptures as rpt
        model = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
        model.fit(signal)
        # ruptures returns 1-based break indices; last entry is n (end sentinel)
        breakpoints = model.predict(pen=10.0)
        # Convert to 0-based internal indices, excluding the end sentinel
        return [b - 1 for b in breakpoints if b < n]

    except ImportError:
        # Simple fallback: detect if the second half mean differs meaningfully from first half
        mid = n // 2
        first_mean = float(np.mean(burn_series[:mid]))
        second_mean = float(np.mean(burn_series[mid:]))
        if first_mean > 0 and abs(second_mean - first_mean) / first_mean > 0.15:
            return [mid]
        return []


def calculate_burn_acceleration(burn_series: List[float]) -> float:
    """
    Compute mean quarter-over-quarter proportional change in burn.

    Returns a signed rate: positive = accelerating, negative = decelerating.
    e.g. 0.30 means burns are rising 30% per quarter on average.
    """
    if len(burn_series) < 2:
        return 0.0

    pct_changes = []
    for i in range(1, len(burn_series)):
        prev = burn_series[i - 1]
        if prev > 0:
            pct_changes.append((burn_series[i] - prev) / prev)

    return float(np.mean(pct_changes)) if pct_changes else 0.0


def classify_burn_regime(
    burn_series: List[float],
    acceleration: float,
    thresholds: BurnRegimeThresholds | None = None,
) -> Tuple[str, str]:
    """
    Classify the burn regime based on recent trajectory.

    Returns (regime_label, interpretation).
    """
    if thresholds is None:
        thresholds = BurnRegimeThresholds()

    if len(burn_series) < 2:
        return (
            "insufficient data",
            "Fewer than two quarters of data. Regime classification requires at least two data points.",
        )

    if acceleration < -thresholds.stable_max_qoq:
        return (
            "decreasing burn",
            "Operating cash consumption is declining, suggesting cost controls, reduced trial activity, "
            "or program wind-down. Verify whether declining burn reflects voluntary efficiency or "
            "involuntary program reduction.",
        )
    elif abs(acceleration) <= thresholds.stable_max_qoq:
        return (
            "stable burn",
            "Burn rate is relatively stable across recent quarters. Simple runway and modeled runway "
            "estimates are broadly consistent.",
        )
    elif acceleration <= thresholds.accelerating_max_qoq:
        return (
            "accelerating burn",
            "Operating cash consumption is increasing moderately. This is common during trial expansion, "
            "cohort scaling, or site activation. Simple runway guidance may overstate remaining time.",
        )
    else:
        return (
            "sharply accelerating burn",
            "Operating cash consumption is increasing sharply. Stated runway guidance based on prior "
            "burn may substantially overstate remaining time. Heightened financing risk warranted.",
        )


def _compute_quarterly_pct_changes(burn_series: List[float]) -> List[Optional[float]]:
    """Return QoQ percent changes; None for the first entry."""
    result: List[Optional[float]] = [None]
    for i in range(1, len(burn_series)):
        prev = burn_series[i - 1]
        if prev > 0:
            result.append(round((burn_series[i] - prev) / prev, 4))
        else:
            result.append(None)
    return result


def run_burn_regime_analysis(
    financial: CompanyFinancialInput,
    config: CatalystLensConfig | None = None,
) -> BurnRegimeResult:
    """Run the full burn regime detection analysis."""
    if config is None:
        config = get_default_config()

    history = financial.quarterly_burn_history
    burn_series = [entry.operating_cash_burn for entry in history]
    quarters = [entry.quarter for entry in history]

    if not burn_series:
        burn_series = [financial.quarterly_operating_cash_burn]
        quarters = ["current"]

    changepoints = detect_burn_changepoints(burn_series)
    acceleration = calculate_burn_acceleration(burn_series)
    regime, interpretation = classify_burn_regime(
        burn_series, acceleration, config.burn_regime_thresholds
    )
    qoq_changes = _compute_quarterly_pct_changes(burn_series)

    return BurnRegimeResult(
        burn_series=[round(b, 2) for b in burn_series],
        quarters=quarters,
        quarterly_pct_changes=qoq_changes,
        changepoint_indices=changepoints,
        burn_acceleration=round(acceleration, 4),
        regime=regime,
        regime_interpretation=interpretation,
        model_assumptions=[
            "PELT change-point detection uses RBF cost function with penalty=10.0.",
            "Falls back to simple mean-split method if ruptures library is unavailable.",
            "Burn acceleration is the mean QoQ proportional change in operating cash burn.",
            "Regime classification thresholds are configurable in config.py.",
        ],
    )
