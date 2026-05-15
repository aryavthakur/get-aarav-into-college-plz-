"""
Distributionally Robust Optimization (DRO) for CatalystLens.

Computes Wasserstein-ball worst-case bounds on cashout probability and EV.
Instead of assuming the model's parameter distributions are exactly correct,
DRO hedges against nearby distributions within an epsilon-ball.

Method: Variance-based worst-case bounds (tractable closed form).

For a random variable X with mean mu and variance sigma^2, the worst-case
expectation of a convex functional within a Wasserstein epsilon-ball is:

  E_worst[f(X)] ≤ f(mu) + epsilon * Lip(f) + sqrt(Var(X)) * phi(epsilon)

For a monotone 0/1 indicator (cashout event), the worst-case probability
under distributional uncertainty is bounded by:

  P_worst(cashout) ≤ P_nominal(cashout) + epsilon * sqrt(Var(1[cashout]))
                   = P_nominal + epsilon * sqrt(P(1-P))

This is the Cantelli-DRO bound. It quantifies how much the cashout probability
could deviate from the model estimate if the true distribution is within epsilon
of the fitted model in 2-Wasserstein distance.

Practical interpretation:
  epsilon=0.05: minor distributional misspecification (parameter uncertainty)
  epsilon=0.10: moderate misspecification (model uncertainty)
  epsilon=0.20: substantial misspecification (regime shift / tail risk)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustnessResult:
    nominal_cashout_prob: float
    nominal_ev: float

    # Wasserstein worst-case bounds
    worst_case_cashout_prob_e05: float
    worst_case_cashout_prob_e10: float
    worst_case_cashout_prob_e20: float

    worst_case_ev_e05: float
    worst_case_ev_e10: float
    worst_case_ev_e20: float

    # Best-case (distributional optimism)
    best_case_cashout_prob_e10: float
    best_case_ev_e10: float

    robustness_interpretation: str
    methodology_note: str = (
        "Variance-scaled distributional sensitivity bounds via Cantelli inequality. "
        "Epsilon = 2-Wasserstein radius as fraction of distributional spread. "
        "Worst-case EV uses variance-scaled bound: E_worst = E_nominal - epsilon * std(V)."
    )


def compute_robustness_bounds(
    t_fin: np.ndarray,
    t_sci: np.ndarray,
    pos_samples: np.ndarray,
    nominal_cashout_prob: float,
    nominal_ev: float,
    ev_samples: np.ndarray | None = None,
) -> RobustnessResult:
    """
    Compute distributional-robustness bounds for cashout probability and EV.

    Parameters
    ----------
    t_fin : financing / cash-out times (n,)
    t_sci : scientific milestone times (n,)
    pos_samples : per-path PoS draws (n,)
    nominal_cashout_prob : E[1(cashout before catalyst)]
    nominal_ev : financing-adjusted rNPV
    ev_samples : per-path EV values (optional; computed from distributions if None)
    """
    # Use the passed nominal as the anchor probability so bounds are symmetric around it.
    p = float(np.clip(nominal_cashout_prob, 1e-6, 1 - 1e-6))

    # Variance of the cashout indicator (Bernoulli)
    var_cashout = p * (1.0 - p)
    std_cashout = float(np.sqrt(var_cashout))

    # If EV samples not supplied, use a simple approximation from PoS and timing
    if ev_samples is None:
        # Very rough proxy: positive PoS contribution weighted by funded probability
        funded_fraction = float(np.mean(t_fin >= t_sci))
        ev_proxy = pos_samples * funded_fraction
        ev_std = float(np.std(ev_proxy)) * abs(nominal_ev) / max(abs(float(np.mean(ev_proxy))), 1e-6)
    else:
        ev_std = float(np.std(ev_samples))

    def _worst_cashout(eps: float) -> float:
        return float(np.clip(p + eps * std_cashout, 0.0, 1.0))

    def _best_cashout(eps: float) -> float:
        return float(np.clip(p - eps * std_cashout, 0.0, 1.0))

    def _worst_ev(eps: float) -> float:
        # Worst-case EV = nominal minus epsilon * std (distributional shift toward lower EV)
        return nominal_ev - eps * ev_std

    def _best_ev(eps: float) -> float:
        return nominal_ev + eps * ev_std

    wc05 = _worst_cashout(0.05)
    wc10 = _worst_cashout(0.10)
    wc20 = _worst_cashout(0.20)

    we05 = _worst_ev(0.05)
    we10 = _worst_ev(0.10)
    we20 = _worst_ev(0.20)

    bc10 = _best_cashout(0.10)
    be10 = _best_ev(0.10)

    # Interpretation
    spread_10 = wc10 - bc10
    if spread_10 < 0.05:
        interp = (
            "Results are robust: even under 10% distributional perturbation, "
            "cashout probability shifts by only {:.1%}. Conclusions stable.".format(spread_10)
        )
    elif spread_10 < 0.15:
        interp = (
            "Moderate sensitivity to distributional misspecification. "
            "10% perturbation spans {:.1%} cashout probability range.".format(spread_10)
        )
    else:
        interp = (
            "High sensitivity to distributional assumptions. "
            "10% perturbation spans {:.1%} cashout probability range — "
            "conclusions depend materially on distribution choice.".format(spread_10)
        )

    return RobustnessResult(
        nominal_cashout_prob=round(nominal_cashout_prob, 4),
        nominal_ev=round(nominal_ev, 2),
        worst_case_cashout_prob_e05=round(wc05, 4),
        worst_case_cashout_prob_e10=round(wc10, 4),
        worst_case_cashout_prob_e20=round(wc20, 4),
        worst_case_ev_e05=round(we05, 2),
        worst_case_ev_e10=round(we10, 2),
        worst_case_ev_e20=round(we20, 2),
        best_case_cashout_prob_e10=round(bc10, 4),
        best_case_ev_e10=round(be10, 2),
        robustness_interpretation=interp,
    )
