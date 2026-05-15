"""
Gaussian copula model for dependent clinical and financing risks.

Independence assumption in the base model:
  T_fin ~ Cox-Weibull(financial covariates)
  T_sci ~ Gamma(clinical covariates)
  These are sampled independently.

Reality: T_fin and T_sci can be correlated:
  - Enrollment delays (longer T_sci) often coincide with burn acceleration (shorter T_fin)
    because slow trials reflect site activation problems that also increase burn.
  - Strong interim data (catalyst for positive PoS) can unlock favorable financing,
    shortening T_fin (negative correlation in some settings).

Gaussian copula model:
  U_fin = F_fin(T_fin), U_sci = F_sci(T_sci)
  (U_fin, U_sci) ~ Gaussian copula with correlation rho.

Sampling via Cholesky decomposition:
  (Z_fin, Z_sci) ~ N(0, Sigma) where Sigma = [[1, rho], [rho, 1]]
  U_fin = Phi(Z_fin),  U_sci = Phi(Z_sci)
  T_fin = F_fin^{-1}(U_fin),  T_sci = F_sci^{-1}(U_sci)

Two scenarios:
  rho = +0.30: enrollment delays correlate with financing stress
    (typical for a poorly-managed trial that also has high burn)
  rho = -0.20: strong clinical progress unlocks financing
    (positive interim data triggers partnership or equity raise)

The copula model allows us to quantify:
  - How much cashout probability changes with tail dependence
  - Whether correlation increases or decreases the gap-to-catalyst risk
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri


@dataclass
class CopulaResult:
    rho: float
    copula_cashout_prob: float
    independent_cashout_prob: float
    dependence_effect: float
    copula_median_gap: float
    interpretation: str
    methodology_note: str = (
        "Gaussian copula with correlation rho between financing and scientific timing ranks. "
        "Marginal distributions unchanged; only rank dependence is modified. "
        "Correlation rho > 0: delayed trials coincide with shorter financing runway."
    )


def _gaussian_copula_samples(
    rng: np.random.Generator,
    n: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample two uniform marginals (U1, U2) from a Gaussian copula with correlation rho.

    Returns (U1, U2) in (0, 1)^2.
    """
    rho = float(np.clip(rho, -0.999, 0.999))
    # Cholesky decomp of [[1, rho], [rho, 1]]
    L11 = 1.0
    L21 = rho
    L22 = float(np.sqrt(max(1.0 - rho * rho, 1e-12)))

    Z1 = rng.standard_normal(n)
    Z2 = rng.standard_normal(n)

    W1 = L11 * Z1
    W2 = L21 * Z1 + L22 * Z2

    # Convert to uniform via Gaussian CDF
    U1 = ndtr(W1).clip(1e-9, 1 - 1e-9)
    U2 = ndtr(W2).clip(1e-9, 1 - 1e-9)
    return U1, U2


def simulate_with_copula(
    t_fin_sorted: np.ndarray,
    t_sci_sorted: np.ndarray,
    rng: np.random.Generator,
    rho: float,
    base_cashout_prob: float | None = None,
) -> CopulaResult:
    """
    Resample t_fin and t_sci with Gaussian copula rank correlation rho.

    The marginal distributions are preserved exactly (same empirical CDF).
    Only the joint rank structure is changed.

    t_fin_sorted: original financing times sorted ascending (shape n,)
    t_sci_sorted: original science times sorted ascending (shape n,)
    rho: correlation in the Gaussian copula
    base_cashout_prob: if provided, use as the independence baseline cashout probability
        (computed from original paired arrays) to avoid the false independence baseline
        from comparing independently-sorted arrays.
    """
    n = len(t_fin_sorted)
    assert len(t_sci_sorted) == n, "t_fin and t_sci must have equal length"

    # Use provided base cashout prob (computed on original paired arrays) to avoid
    # the false independence baseline from comparing independently-sorted arrays.
    if base_cashout_prob is not None:
        independent_cashout = float(base_cashout_prob)
    else:
        independent_cashout = float(np.mean(t_fin_sorted < t_sci_sorted))

    # Copula resampling: draw (U_fin, U_sci) from copula, then
    # map back to empirical quantiles of the original sorted arrays.
    U_fin, U_sci = _gaussian_copula_samples(rng, n, rho)

    # Rank-based lookup into empirical marginals
    rank_fin = np.clip((U_fin * n).astype(int), 0, n - 1)
    rank_sci = np.clip((U_sci * n).astype(int), 0, n - 1)

    t_fin_cop = t_fin_sorted[rank_fin]
    t_sci_cop = t_sci_sorted[rank_sci]

    copula_cashout = float(np.mean(t_fin_cop < t_sci_cop))
    gap = t_sci_cop - t_fin_cop
    copula_median_gap = float(np.median(gap))

    effect = copula_cashout - independent_cashout

    if abs(effect) < 0.01:
        interp = f"Copula (rho={rho:+.2f}) has negligible effect on cashout probability."
    elif effect > 0:
        interp = (
            f"Positive copula (rho={rho:+.2f}): enrollment delays coincide with financing stress, "
            f"raising cashout probability by {effect:+.1%} vs independence assumption."
        )
    else:
        interp = (
            f"Negative copula (rho={rho:+.2f}): clinical progress unlocks financing, "
            f"reducing cashout probability by {abs(effect):.1%} vs independence assumption."
        )

    return CopulaResult(
        rho=rho,
        copula_cashout_prob=round(copula_cashout, 4),
        independent_cashout_prob=round(independent_cashout, 4),
        dependence_effect=round(effect, 4),
        copula_median_gap=round(copula_median_gap, 2),
        interpretation=interp,
    )


@dataclass
class DependenceAnalysisResult:
    positive_rho: CopulaResult
    negative_rho: float
    negative_copula_cashout_prob: float
    negative_dependence_effect: float
    negative_interpretation: str
    base_cashout_prob: float
    methodology_note: str = (
        "Gaussian copula over {T_fin, T_sci} marginals with rho=+0.30 (trial delays ~ financing stress) "
        "and rho=-0.20 (clinical progress ~ financing unlock). "
        "Marginal distributions are held fixed; only tail dependence is varied."
    )


def run_dependence_analysis(
    t_fin: np.ndarray,
    t_sci: np.ndarray,
    rng: np.random.Generator,
) -> DependenceAnalysisResult:
    """
    Run Gaussian copula analysis under two canonical correlation scenarios.
    """
    t_fin_s = np.sort(t_fin)
    t_sci_s = np.sort(t_sci)
    base_cashout = float(np.mean(t_fin < t_sci))  # paired comparison on originals

    pos_result = simulate_with_copula(t_fin_s, t_sci_s, rng, rho=+0.30, base_cashout_prob=base_cashout)
    neg_result = simulate_with_copula(t_fin_s, t_sci_s, rng, rho=-0.20, base_cashout_prob=base_cashout)

    return DependenceAnalysisResult(
        positive_rho=pos_result,
        negative_rho=neg_result.rho,
        negative_copula_cashout_prob=neg_result.copula_cashout_prob,
        negative_dependence_effect=neg_result.dependence_effect,
        negative_interpretation=neg_result.interpretation,
        base_cashout_prob=round(base_cashout, 4),
    )
