"""
Multi-state competing-risk survival engine.

Eight states:
  0  operating              (transient — still running)
  1  funded                 (absorbing — clean equity round)
  2  clean_refinancing      (absorbing — ATM / small PIPE)
  3  distressed_refinancing (absorbing — convertible / heavy dilution)
  4  partnership            (absorbing — licensing / co-development deal)
  5  debt_or_royalty        (absorbing — debt / royalty financing)
  6  program_discontinuation(absorbing — trial stopped, pivot)
  7  cash_exhaustion        (absorbing — literal cash out)

Cause-specific hazard: h_j(t|X) = h_0j(t) · exp(X^T β_j)
Baseline: Weibull, h_0j(t) = (k_j / scale_j) · (t / scale_j)^(k_j - 1)

Sampling: per-cause inverse-CDF (exact for independent Weibulls with differing shapes),
then take argmin to get the winning cause.
"""

from __future__ import annotations

import numpy as np

CAUSE_NAMES: dict[int, str] = {
    1: "funded",
    2: "clean_refinancing",
    3: "distressed_refinancing",
    4: "partnership",
    5: "debt_or_royalty",
    6: "program_discontinuation",
    7: "cash_exhaustion",
}

# Maps competing-risk cause → valuation state used by valuation.py's four-state model.
# States: 0=funded, 1=clean refi, 2=distressed, 3=discontinuation/cashout
CAUSE_TO_VALUATION_STATE: dict[int, int] = {
    1: 0,  # funded → full value
    2: 1,  # clean_refinancing → clean dilution
    3: 2,  # distressed_refinancing → heavy dilution
    4: 1,  # partnership → treated as clean refi (deal proceeds offset dilution)
    5: 2,  # debt_or_royalty → small haircut like distressed
    6: 3,  # program_discontinuation → downside
    7: 3,  # cash_exhaustion → downside
}

# Default per-cause Weibull scale parameters (months) and shape k.
# Calibrated so that the aggregate baseline median ≈ 22 months, matching the
# existing Cox-Weibull aggregate: S0(t)=exp(-(0.035t)^1.30), median≈21.6 months.
# Values are MVP assumptions; replace with fitted parameters once historical data available.
DEFAULT_CAUSE_SCALES: dict[int, tuple[float, float]] = {
    # cause_id: (k, scale_months)
    1: (1.30, 80.0),    # funded — most common outcome, moderate-hazard
    2: (1.10, 70.0),    # clean_refinancing — slightly front-loaded
    3: (1.50, 150.0),   # distressed — rare early, accelerates with time
    4: (1.20, 185.0),   # partnership — uncommon, roughly flat hazard
    5: (1.20, 215.0),   # debt_or_royalty — rare
    6: (1.80, 265.0),   # discontinuation — very rare, but picks up late
    7: (2.00, 125.0),   # cash_exhaustion — concentrated near cash-out event
}


def sample_competing_risk(
    cause_scales: dict[int, tuple[float, float]],
    cause_lp: dict[int, float],
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """
    Sample (time, cause) pairs for n paths.

    cause_scales: {cause_id: (k, scale_months)}
    cause_lp:     {cause_id: risk_multiplier} — exp(X^T β_j), default 1.0

    Cox accelerated-time: effective scale = scale_j / lp_j^(1/k_j)
    Inverse-CDF: T_j = (-log U)^(1/k_j) * scale_j_eff
    Winner = argmin(T_j) across causes.

    Returns ndarray of shape (n, 2): column 0 = time (float), column 1 = cause (int).
    """
    causes = sorted(cause_scales)
    n_causes = len(causes)
    times = np.empty((n, n_causes), dtype=np.float64)

    u = rng.uniform(0.0, 1.0, size=(n, n_causes))
    neg_log_u = -np.log(np.clip(u, 1e-15, 1.0))

    for col, cid in enumerate(causes):
        k, scale = cause_scales[cid]
        lp = float(cause_lp.get(cid, 1.0))
        lp = max(lp, 1e-9)
        scale_eff = scale / (lp ** (1.0 / k))
        times[:, col] = neg_log_u[:, col] ** (1.0 / k) * scale_eff

    winner_col = np.argmin(times, axis=1)
    winner_time = times[np.arange(n), winner_col]
    winner_cause = np.array([causes[c] for c in winner_col], dtype=np.int32)

    result = np.empty((n, 2), dtype=np.float64)
    result[:, 0] = np.maximum(winner_time, 0.1)
    result[:, 1] = winner_cause.astype(np.float64)
    return result


def build_cause_lp(
    aggregate_lp: float,
    cause_ids: list[int] | None = None,
) -> dict[int, float]:
    """
    Distribute the aggregate Cox linear predictor across cause-specific hazards.

    Strategy: the aggregate LP modulates causes differentially —
      cash_exhaustion (+full) is strongly correlated with financial distress;
      distressed_refinancing (+partial) is somewhat correlated;
      funded / clean_refi / partnership are *negatively* correlated
      (when a company is distressed, clean financing becomes harder).

    The LP here is the raw value (before exp), so we exponentiate per cause.

    Aggregate LP = 0 → all cause LPs = 1.0 → baseline hazards → expected behavior.
    """
    cause_weights: dict[int, float] = {
        1: -0.25,   # funded: harder to raise in distress → negative correlation
        2: -0.20,   # clean_refinancing: same
        3:  0.55,   # distressed_refinancing: positively correlated
        4: -0.15,   # partnership: weakly negative (partners shy from distressed companies)
        5:  0.10,   # debt_or_royalty: mildly positive
        6:  0.35,   # program_discontinuation: positively correlated with distress
        7:  1.00,   # cash_exhaustion: direct causal link to financial distress
    }
    ids = cause_ids if cause_ids is not None else list(DEFAULT_CAUSE_SCALES.keys())
    return {cid: float(np.exp(aggregate_lp * cause_weights.get(cid, 0.0))) for cid in ids}


def compute_overall_survival(
    times: np.ndarray,
    cause_scales: dict[int, tuple[float, float]],
    cause_lp: dict[int, float],
) -> np.ndarray:
    """
    Compute S(t) = exp(-∑_j (t / scale_j_eff)^k_j) for each t in times array.

    This is the probability of remaining in the operating state (cause 0)
    at each time point — i.e., no absorbing transition has occurred yet.
    """
    log_s = np.zeros(len(times), dtype=np.float64)
    for cid, (k, scale) in cause_scales.items():
        lp = max(float(cause_lp.get(cid, 1.0)), 1e-9)
        scale_eff = scale / (lp ** (1.0 / k))
        log_s -= (times / scale_eff) ** k
    return np.exp(log_s)


def compute_cause_hazard(
    times: np.ndarray,
    k: float,
    scale: float,
    lp: float,
) -> np.ndarray:
    """Cause-specific hazard h_j(t) = (k/scale_eff) * (t/scale_eff)^(k-1)."""
    scale_eff = scale / max(lp, 1e-9) ** (1.0 / k)
    return (k / scale_eff) * (times / scale_eff) ** (k - 1.0)


def compute_cif_curves(
    time_grid: np.ndarray,
    cause_scales: dict[int, tuple[float, float]],
    cause_lp: dict[int, float],
) -> dict[int, np.ndarray]:
    """
    Compute the cumulative incidence function (CIF) for each cause on a time grid.

    CIF_j(t) = ∫₀ᵗ h_j(s) · S(s) ds

    Uses trapezoidal integration on the provided grid.
    Returns {cause_id: CIF_j(t) array} of shape len(time_grid).

    Note: sum_j CIF_j(t) = 1 − S(t) ≤ 1 for all t.
    """
    s_values = compute_overall_survival(time_grid, cause_scales, cause_lp)

    cif: dict[int, np.ndarray] = {}
    for cid, (k, scale) in cause_scales.items():
        lp = max(float(cause_lp.get(cid, 1.0)), 1e-9)
        h_j = compute_cause_hazard(time_grid, k, scale, lp)
        integrand = h_j * s_values
        cif_j = np.zeros(len(time_grid), dtype=np.float64)
        for i in range(1, len(time_grid)):
            dt = time_grid[i] - time_grid[i - 1]
            cif_j[i] = cif_j[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * dt
        cif[cid] = cif_j

    return cif


def cif_at_time(
    t: float,
    cause_scales: dict[int, tuple[float, float]],
    cause_lp: dict[int, float],
    grid_points: int = 200,
) -> dict[int, float]:
    """CIF for each cause evaluated at a single time t."""
    grid = np.linspace(0.0, t, grid_points)
    curves = compute_cif_curves(grid, cause_scales, cause_lp)
    return {cid: float(curves[cid][-1]) for cid in curves}
