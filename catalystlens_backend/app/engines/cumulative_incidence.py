"""
Public API for cumulative incidence function (CIF) curves.

Thin wrapper over multistate.py exposing a clean function for the
valuation and report layers without taking a dependency on multistate internals.
"""

from __future__ import annotations

import numpy as np

from app.engines.multistate import (
    CAUSE_NAMES,
    DEFAULT_CAUSE_SCALES,
    build_cause_lp,
    cif_at_time,
    compute_cif_curves,
    compute_overall_survival,
)


def build_default_time_grid(
    horizon_months: float = 48.0,
    n_points: int = 200,
) -> np.ndarray:
    """Uniform time grid from 0 to horizon_months."""
    return np.linspace(0.0, horizon_months, n_points)


def named_cif_at_time(
    t: float,
    aggregate_lp: float = 0.0,
    cause_scales: dict[int, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """
    CIF_{cause_name}(t) for each cause at time t.

    aggregate_lp: the aggregate Cox linear predictor; distributed across
    causes via build_cause_lp.
    """
    scales = cause_scales if cause_scales is not None else DEFAULT_CAUSE_SCALES
    cause_lp = build_cause_lp(aggregate_lp, list(scales.keys()))
    raw = cif_at_time(t, scales, cause_lp)
    return {CAUSE_NAMES[cid]: v for cid, v in raw.items()}


def survival_at_catalyst(
    catalyst_month: float,
    aggregate_lp: float = 0.0,
    cause_scales: dict[int, tuple[float, float]] | None = None,
) -> float:
    """P(still operating at catalyst_month) = S(catalyst_month)."""
    scales = cause_scales if cause_scales is not None else DEFAULT_CAUSE_SCALES
    cause_lp = build_cause_lp(aggregate_lp, list(scales.keys()))
    t = np.array([catalyst_month])
    s = compute_overall_survival(t, scales, cause_lp)
    return float(s[0])
