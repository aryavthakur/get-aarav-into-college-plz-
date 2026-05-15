"""
Bayesian Expected Value of Information for diligence prioritization.

Two quantities are computed:
  EVPI — Expected Value of Perfect Information about the clinical outcome.
         Answers: "How much would it be worth to learn whether the trial will
         succeed before you commit capital?"
         EVPI > 0 precisely when a negative clinical signal could flip the
         investment decision from invest to pass.

  EVSI — Expected Value of Sample Information for each diligence signal type.
         Models observing one weighted piece of evidence (equivalent to
         signal_weight effective Bernoulli observations from the Beta-Binomial
         preposterior) and computing the resulting decision-value improvement.
         Signals are ranked to prioritise diligence spend.

Methodology (explicit decision model):
  Decision: invest capital_required for the right to upside_value on trial success, or pass (value=0).
  Current belief: PoS ~ Beta(α, β), posterior_mean = α/(α+β).
  invest_value = pos_mean * upside_value - capital_required
  pass_value   = 0
  decision_value = max(invest_value, pass_value)

  Under perfect information:
    P(success) paths → max(upside_value - capital_required, 0) (invest only if profitable)
    P(failure)  paths → 0 (always pass on known failure)
  EVPI = pos_mean * max(upside_value - capital_required, 0) - decision_value

  For EVSI (partial signal of weight w updating the Beta posterior):
    Posterior after positive: pos_after_pos = (α+w)/(α+β+w)
    Posterior after negative: pos_after_neg = α/(α+β+w)
    EVSI_w = P(+)·max(0, pos_after_pos * upside - capital) +
             P(−)·max(0, pos_after_neg * upside - capital) − decision_value

  EVSI is nonzero when the updated posterior could flip the investment decision.
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
    methodology_note: str = (
        "EVPI / EVSI computed via explicit decision model: "
        "invest_value = pos_mean * upside_value - capital_required; pass_value = 0. "
        "EVPI = E[value under perfect clinical outcome info] - decision_value. "
        "EVSI per signal = preposterior expected value after Beta update - decision_value. "
        "Values are positive when a negative signal could flip the invest/pass decision."
    )


def compute_evpi(
    alpha_posterior: float,
    beta_posterior: float,
    upside_value: float,
    capital_required: float,
) -> float:
    """
    EVPI under explicit decision model.

    invest_value = pos_mean * upside_value - capital_required
    Under perfect info: investor invests only on known success (if upside > capital),
    passes on known failure.
    EVPI = pos_mean * max(upside_value - capital_required, 0) - max(invest_value, 0)
    """
    pos_mean = alpha_posterior / (alpha_posterior + beta_posterior)
    invest_value = pos_mean * upside_value - capital_required
    decision_value = max(invest_value, 0.0)
    ev_perfect_info = pos_mean * max(upside_value - capital_required, 0.0)
    return max(0.0, ev_perfect_info - decision_value)


def compute_signal_evsi(
    alpha_posterior: float,
    beta_posterior: float,
    signal_weight: float,
    upside_value: float,
    capital_required: float,
) -> tuple[float, float, float]:
    """
    EVSI for a diligence signal with effective weight signal_weight.

    Returns (evsi_dollars, invest_value_if_positive, invest_value_if_negative).
    """
    total = alpha_posterior + beta_posterior
    p_pos = alpha_posterior / total

    pos_after_pos = (alpha_posterior + signal_weight) / (total + signal_weight)
    pos_after_neg = alpha_posterior / (total + signal_weight)

    invest_current = p_pos * upside_value - capital_required
    decision_current = max(invest_current, 0.0)

    iv_pos = pos_after_pos * upside_value - capital_required
    iv_neg = pos_after_neg * upside_value - capital_required

    preposterior = p_pos * max(iv_pos, 0.0) + (1.0 - p_pos) * max(iv_neg, 0.0)
    evsi = max(0.0, preposterior - decision_current)
    return evsi, iv_pos, iv_neg


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
    upside_value: float,
    capital_required: float,
    config_signal_weights: Dict[str, float] | None = None,
) -> ValueOfInformationResult:
    """
    Compute EVPI and per-signal EVSI for diligence prioritization.

    upside_value: expected discounted asset value conditional on trial success
    capital_required: capital the investor must commit to reach the catalyst (e.g. cash gap)
    config_signal_weights: if supplied, override the default signal catalogue weights.
    """
    pos_mean = alpha_posterior / (alpha_posterior + beta_posterior)
    evpi = compute_evpi(alpha_posterior, beta_posterior, upside_value, capital_required)

    invest_value = pos_mean * upside_value - capital_required
    ev_abs = abs(invest_value) if invest_value != 0 else max(abs(upside_value), 1.0)
    evpi_pct = evpi / ev_abs * 100.0 if ev_abs > 0 else 0.0

    decision_value = max(invest_value, 0.0)
    if evpi > decision_value * 0.05 and decision_value > 0:
        evpi_interp = (
            f"EVPI of ${evpi:,.0f} is material relative to current decision value — the investment "
            f"thesis is near the decision threshold; resolving PoS uncertainty quickly has "
            f"meaningful option value."
        )
    elif evpi > 0:
        evpi_interp = (
            f"EVPI of ${evpi:,.0f} — knowing the clinical outcome before committing would "
            f"prevent capital loss on the fraction of failure paths."
        )
    else:
        evpi_interp = (
            "EVPI = $0 — under current posterior, the investment decision would not change "
            "regardless of clinical outcome (either never profitable or the pass decision dominates)."
        )

    signal_evsis: List[SignalEVSI] = []
    for sig_name, (desc, cat, default_w) in _SIGNAL_CATALOGUE.items():
        w = float(config_signal_weights.get(sig_name, default_w)) if config_signal_weights else default_w
        evsi_d, ev_pos, ev_neg = compute_signal_evsi(
            alpha_posterior, beta_posterior, w, upside_value, capital_required
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
    )
