"""
Bayesian Model Averaging (BMA) for CatalystLens.

Rather than committing to a single parameter setting, BMA computes a
posterior-weighted average over a discrete set of plausible models.

Model set: 3 × 3 = 9 candidate models spanning Weibull shape × scale combinations
that represent different views on biotech financing hazard dynamics:

  Shapes (k): 1.0 (memoryless), 1.30 (baseline), 1.60 (accelerating)
  Scales (λ): 0.025 (long runway), 0.035 (baseline), 0.050 (short runway)

Each model M_i has:
  - Prior weight π_i (uniform by default)
  - Likelihood given observed company characteristics: L(data | M_i)
  - Posterior weight: w_i ∝ π_i * L(data | M_i)

Likelihood: proxy via how well the model's predicted survival at simple_runway
matches the prior expectation S(runway) ≈ 0.50 (median survival expected at
the model's own median). Penalise models that predict very extreme survival
at the company's current runway.

BMA outputs:
  - Model-averaged cashout probability
  - Model-averaged EV
  - Posterior model weights (which model the data supports most)
  - Effective number of models (diversity measure)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeibullModel:
    k: float
    lambda_: float
    prior_weight: float = 1.0 / 9.0

    def survival(self, t: float, risk_multiplier: float = 1.0) -> float:
        """S(t | rm) = exp(-(lambda * rm * t)^k)."""
        return math.exp(-((self.lambda_ * risk_multiplier * t) ** self.k))

    def log_likelihood(self, simple_runway: float, risk_multiplier: float) -> float:
        """
        Proxy log-likelihood: penalise models where S(runway) is far from 0.50.

        Rationale: a company at simple runway should have ~50% survival probability
        according to the model (definition of median). Models far from this are poorly
        calibrated for this company's situation.
        """
        s_runway = self.survival(simple_runway, risk_multiplier)
        # Gaussian log-likelihood with target 0.50 and sigma 0.20
        residual = s_runway - 0.50
        return -0.5 * (residual / 0.20) ** 2


_CANDIDATE_MODELS = [
    WeibullModel(k=1.00, lambda_=0.025),
    WeibullModel(k=1.00, lambda_=0.035),
    WeibullModel(k=1.00, lambda_=0.050),
    WeibullModel(k=1.30, lambda_=0.025),
    WeibullModel(k=1.30, lambda_=0.035),
    WeibullModel(k=1.30, lambda_=0.050),
    WeibullModel(k=1.60, lambda_=0.025),
    WeibullModel(k=1.60, lambda_=0.035),
    WeibullModel(k=1.60, lambda_=0.050),
]


@dataclass
class ModelWeight:
    k: float
    lambda_: float
    posterior_weight: float
    model_cashout_prob: float
    model_ev: float


@dataclass
class BMAResult:
    bma_cashout_prob: float
    bma_ev: float
    model_weights: list[ModelWeight]
    effective_n_models: float
    highest_weight_model_k: float
    highest_weight_model_lambda: float
    methodology_note: str = (
        "Proxy Bayesian-style model averaging over hand-specified Weibull candidates "
        "(3 shapes × 3 scales). Posterior weights from proxy log-likelihood "
        "based on calibration at simple runway. BMA output is the "
        "posterior-weighted average across all model candidates. "
        "Note: proxy likelihood uses calibration heuristic, not true marginal likelihood."
    )


def compute_bma(
    simple_runway: float,
    risk_multiplier: float,
    base_cashout_prob: float,
    base_ev: float,
    models: list[WeibullModel] | None = None,
) -> BMAResult:
    """
    Compute BMA-averaged cashout probability and EV.

    Approximation: each model's cashout_prob and EV are scaled relative to
    the baseline model (k=1.30, lambda=0.035) using the ratio of their
    survival functions at the relevant time horizon.

    This is a first-order approximation; exact BMA would re-run the full
    simulation for each model.
    """
    candidates = models if models is not None else _CANDIDATE_MODELS

    # Compute log-likelihoods and normalise
    log_liks = np.array([m.log_likelihood(simple_runway, risk_multiplier) for m in candidates])
    log_prior = np.array([math.log(max(m.prior_weight, 1e-15)) for m in candidates])
    log_post = log_liks + log_prior
    log_post -= np.max(log_post)  # numerical stability
    post_weights = np.exp(log_post)
    post_weights /= post_weights.sum()

    # Baseline model for relative scaling
    baseline = WeibullModel(k=1.30, lambda_=0.035)
    baseline_s = baseline.survival(simple_runway, risk_multiplier)

    weighted_cp = 0.0
    weighted_ev = 0.0
    mw_list: list[ModelWeight] = []

    for m, w in zip(candidates, post_weights):
        s_m = m.survival(simple_runway, risk_multiplier)
        # Scale: lower survival at runway → higher cashout prob
        if baseline_s > 1e-9:
            cp_ratio = (1.0 - s_m) / max(1.0 - baseline_s, 1e-9)
        else:
            cp_ratio = 1.0
        model_cp = float(np.clip(base_cashout_prob * cp_ratio, 0.0, 1.0))
        model_ev = base_ev * (s_m / max(baseline_s, 1e-9))  # higher survival → higher EV

        weighted_cp += float(w) * model_cp
        weighted_ev += float(w) * model_ev
        mw_list.append(ModelWeight(
            k=m.k, lambda_=m.lambda_,
            posterior_weight=round(float(w), 4),
            model_cashout_prob=round(model_cp, 4),
            model_ev=round(model_ev, 2),
        ))

    mw_list.sort(key=lambda x: x.posterior_weight, reverse=True)
    top = mw_list[0]

    # Effective number of models (entropy-based)
    weights = np.array([x.posterior_weight for x in mw_list])
    weights = weights / weights.sum()
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-15))))
    eff_n = math.exp(entropy)

    return BMAResult(
        bma_cashout_prob=round(float(np.clip(weighted_cp, 0.0, 1.0)), 4),
        bma_ev=round(weighted_ev, 2),
        model_weights=mw_list,
        effective_n_models=round(eff_n, 2),
        highest_weight_model_k=top.k,
        highest_weight_model_lambda=top.lambda_,
    )
