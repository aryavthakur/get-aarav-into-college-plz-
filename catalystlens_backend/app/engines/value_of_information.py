"""
Bayesian Expected Value of Information for diligence prioritization.

Two quantities are computed:
  EVPI — Expected Value of Perfect Information about the clinical outcome.
         Answers: "How much would it be worth to learn whether the trial will
         succeed before you commit capital?"
         Positive when the current investment is borderline (EV ≈ 0) and
         negative signal could change the hold/exit decision.

  EVSI — Expected Value of Sample Information for each diligence signal type.
         Models observing one weighted piece of evidence (equivalent to
         signal_weight effective Bernoulli observations from the Beta-Binomial
         preposterior) and computing the resulting decision-value improvement.
         Signals are ranked to prioritise diligence spend.

Methodology (Bayesian preposterior decision model):
  Current belief: PoS ~ Beta(α, β), posterior_mean = α/(α+β).
  invest_value = posterior_mean * upside_value - capital_required.
  pass_value = 0.
  decision_value = max(invest_value, pass_value).
  Observation X ∈ {positive, negative}:
    P(X=+) = α/(α+β) [positive draw more likely if drug works]
    Posterior(+): Beta(α + w, β)
    Posterior(−): Beta(α,     β + w)
  EVSI_w = P(+)·decision_value(Posterior+) + P(−)·decision_value(Posterior−)
           − decision_value(current)

This is nonzero precisely when the current EV is near the decision threshold and
a negative signal could flip the investment decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SignalEVSI:
    signal_name: str
    description: str
    category: str
    signal_weight: float
    evsi_dollars: float
    ev_if_positive: float
    ev_if_negative: float
    p_positive: float


@dataclass
class ValueOfInformationResult:
    evpi_dollars: float
    evpi_pct_of_ev: float
    evpi_interpretation: str
    per_signal_evsi: List[SignalEVSI]
    top_diligence_priority: str
    total_observable_evsi: float
    method_status: str = "heuristic"
    methodology_note: str = (
        "EVPI / EVSI computed under a Beta-Binomial preposterior decision-threshold model. "
        "The decision is a simplified binary invest/pass decision: "
        "invest_value = posterior_pos * upside_value - capital_required; pass_value = 0. "
        "It is not position sizing, hedging, entry-price optimization, or portfolio construction. "
        "Values are modeled estimates from uncalibrated signal weights, not validated investment accuracy."
    )


def _posterior_mean(alpha: float, beta: float) -> float:
    return alpha / max(alpha + beta, 1e-9)


def _infer_decision_inputs(
    alpha: float,
    beta: float,
    financing_adjusted_rnpv: float,
    upside_value: float | None,
    capital_required: float | None,
) -> tuple[float, float]:
    p = _posterior_mean(alpha, beta)
    if upside_value is None:
        if capital_required is None:
            capital_required = 0.0
        upside_value = (financing_adjusted_rnpv + capital_required) / max(p, 1e-9)
    if capital_required is None:
        capital_required = max(float(upside_value) * p - financing_adjusted_rnpv, 0.0)
    return max(float(upside_value), 0.0), max(float(capital_required), 0.0)


def _invest_value(alpha: float, beta: float, upside_value: float, capital_required: float) -> float:
    return _posterior_mean(alpha, beta) * upside_value - capital_required


def _decision_value(alpha: float, beta: float, upside_value: float, capital_required: float) -> float:
    return max(_invest_value(alpha, beta, upside_value, capital_required), 0.0)


def compute_evpi(
    alpha_posterior: float,
    beta_posterior: float,
    financing_adjusted_rnpv: float,
    upside_value: float | None = None,
    capital_required: float | None = None,
) -> float:
    """
    Value of learning the trial outcome (success/failure) before committing.

    EVPI = E_outcome[max(value if outcome known, pass)] - current decision value.

    When EV is strongly positive, the investor invests regardless → EVPI = 0.
    When EV is near zero (borderline), EVPI > 0 because the signal could
    prevent a loss-making investment.
    """
    p_pos = _posterior_mean(alpha_posterior, beta_posterior)
    upside, capital = _infer_decision_inputs(
        alpha_posterior, beta_posterior, financing_adjusted_rnpv, upside_value, capital_required
    )

    current_decision = _decision_value(alpha_posterior, beta_posterior, upside, capital)
    perfect_info_value = p_pos * max(upside - capital, 0.0)
    return max(0.0, perfect_info_value - current_decision)


def compute_signal_evsi(
    alpha_posterior: float,
    beta_posterior: float,
    signal_weight: float,
    financing_adjusted_rnpv: float,
    upside_value: float | None = None,
    capital_required: float | None = None,
) -> tuple[float, float, float]:
    """
    EVSI for a diligence signal with effective weight signal_weight.

    Returns (evsi_dollars, ev_if_positive, ev_if_negative).
    """
    p_pos = _posterior_mean(alpha_posterior, beta_posterior)
    upside, capital = _infer_decision_inputs(
        alpha_posterior, beta_posterior, financing_adjusted_rnpv, upside_value, capital_required
    )

    ev_after_pos = _invest_value(alpha_posterior + signal_weight, beta_posterior, upside, capital)
    ev_after_neg = _invest_value(alpha_posterior, beta_posterior + signal_weight, upside, capital)

    preposterior = p_pos * max(0.0, ev_after_pos) + (1.0 - p_pos) * max(0.0, ev_after_neg)
    current_decision = _decision_value(alpha_posterior, beta_posterior, upside, capital)
    evsi = max(0.0, preposterior - current_decision)
    return evsi, ev_after_pos, ev_after_neg


# Signal catalogue: name → (description, category, weight_proxy)
_SIGNAL_CATALOGUE: Dict[str, tuple[str, str, float]] = {
    "dose_response_observed": (
        "Dose-response relationship observed in prior cohort",
        "clinical", 2.0,
    ),
    "biomarker_correlation": (
        "Target engagement / pharmacodynamic biomarker data available",
        "clinical", 1.5,
    ),
    "prior_phase_positive": (
        "Positive readout in same indication in prior phase",
        "clinical", 1.5,
    ),
    "enrollment_on_track": (
        "Enrollment pace consistent with stated timeline",
        "operational", 0.5,
    ),
    "site_visit_completed": (
        "Clinical site visit: protocol adherence and patient retention confirmed",
        "operational", 0.5,
    ),
    "key_opinion_leader_endorsement": (
        "KOL endorsement of clinical design and endpoint choice",
        "clinical", 0.75,
    ),
    "regulatory_interaction_positive": (
        "Positive FDA/EMA meeting or written feedback on trial design",
        "regulatory", 1.0,
    ),
    "competitor_positive_data": (
        "Competitor in same mechanism/indication shows positive Phase 2+ data",
        "clinical", 1.0,
    ),
    "chemistry_cmc_de_risked": (
        "CMC / manufacturing scalability de-risked for commercial-scale production",
        "operational", 0.4,
    ),
    "independent_data_monitoring": (
        "Independent Data Monitoring Committee has not halted for futility",
        "clinical", 0.75,
    ),
}


def run_value_of_information_analysis(
    alpha_posterior: float,
    beta_posterior: float,
    financing_adjusted_rnpv: float,
    config_signal_weights: Dict[str, float] | None = None,
    upside_value: float | None = None,
    capital_required: float | None = None,
) -> ValueOfInformationResult:
    """
    Compute EVPI and per-signal EVSI for diligence prioritization.

    config_signal_weights: if supplied, override the default signal catalogue weights.
    """
    evpi = compute_evpi(
        alpha_posterior,
        beta_posterior,
        financing_adjusted_rnpv,
        upside_value=upside_value,
        capital_required=capital_required,
    )
    ev_abs = abs(financing_adjusted_rnpv) if financing_adjusted_rnpv != 0 else 1.0
    evpi_pct = evpi / ev_abs * 100.0

    if evpi > financing_adjusted_rnpv * 0.05:
        evpi_interp = (
            f"EVPI of ${evpi:,.0f} is material relative to current EV — the investment "
            f"thesis is near the decision threshold; resolving PoS uncertainty quickly has "
            f"meaningful option value."
        )
    elif evpi > 0:
        evpi_interp = (
            f"EVPI of ${evpi:,.0f} is small relative to current EV — investment thesis is "
            f"robust to clinical uncertainty at current posterior PoS."
        )
    else:
        evpi_interp = (
            "EVPI = $0 — under current posterior, the expected value is sufficiently "
            "positive that the hold/invest decision would not change regardless of clinical "
            "outcome. Diligence shifts the posterior but not the decision direction."
        )

    signal_evsis: List[SignalEVSI] = []
    for sig_name, (desc, cat, default_w) in _SIGNAL_CATALOGUE.items():
        w = float(config_signal_weights.get(sig_name, default_w)) if config_signal_weights else default_w
        evsi_d, ev_pos, ev_neg = compute_signal_evsi(
            alpha_posterior,
            beta_posterior,
            w,
            financing_adjusted_rnpv,
            upside_value=upside_value,
            capital_required=capital_required,
        )
        signal_evsis.append(SignalEVSI(
            signal_name=sig_name,
            description=desc,
            category=cat,
            signal_weight=w,
            evsi_dollars=round(evsi_d, 2),
            ev_if_positive=round(ev_pos, 2),
            ev_if_negative=round(ev_neg, 2),
            p_positive=round(alpha_posterior / (alpha_posterior + beta_posterior), 4),
        ))

    signal_evsis.sort(key=lambda s: s.evsi_dollars, reverse=True)

    top_priority = (
        signal_evsis[0].description if signal_evsis else "No signals available for analysis."
    )
    total_evsi = sum(s.evsi_dollars for s in signal_evsis)

    return ValueOfInformationResult(
        evpi_dollars=round(evpi, 2),
        evpi_pct_of_ev=round(evpi_pct, 2),
        evpi_interpretation=evpi_interp,
        per_signal_evsi=signal_evsis,
        top_diligence_priority=top_priority,
        total_observable_evsi=round(total_evsi, 2),
        method_status="heuristic",
    )
