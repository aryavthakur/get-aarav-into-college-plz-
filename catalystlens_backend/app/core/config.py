"""
CatalystLens configuration.

All coefficients and thresholds are labeled as MVP defaults.
They are calibrated from general biotech financing intuition, NOT from
a trained model on historical outcomes. Replace with fitted parameters
once historical data is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PhasePosPrior:
    """Beta-distribution prior for probability of technical success by phase."""
    alpha: float
    beta: float
    description: str

    @property
    def prior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class SignalWeights:
    """
    Additive weights for Bayesian posterior update.

    Positive signals increase alpha (success evidence).
    Negative signals increase beta (failure evidence).

    Weights are MVP assumptions, not trained on clinical data.
    """
    positive: Dict[str, float] = field(default_factory=lambda: {
        "validated_biomarker": 2.0,
        "strong_preclinical_package": 1.5,
        "prior_human_signal": 2.0,
        "fast_enrollment": 0.5,
        "clinically_meaningful_endpoint": 1.5,
        "randomized_controlled_design": 2.0,
        "strong_mechanistic_rationale": 1.5,
    })
    negative: Dict[str, float] = field(default_factory=lambda: {
        "weak_endpoint": 2.0,
        "small_sample_size": 1.5,
        "open_label_design": 1.5,
        "slow_enrollment": 0.5,
        "safety_concerns": 2.5,
        "prior_failed_trials": 2.0,
        "unclear_mechanism": 1.5,
        "high_competition": 1.0,
        "poor_translatability": 2.0,
    })


@dataclass
class CoxCoefficients:
    """
    Cox-style log-hazard coefficients for biotech financing failure.

    Positive coefficient = increases log-hazard (increases failure risk).
    Negative coefficient = decreases log-hazard (decreases failure risk).

    These are UNTRAINED MVP ASSUMPTIONS. Replace with coefficients fit
    to historical biotech financing failure data once available.

    Reference point: a company with 15 months runway, 12 months since
    last raise, going-concern-free, moderate pipeline concentration, and
    biotech market score of 5 produces LP ≈ 0 (baseline hazard).
    """
    # Per-month runway below the 15-month reference (increases risk)
    cash_runway_per_month_vs_reference: float = 0.04
    reference_runway_months: float = 15.0

    # Burn acceleration rate (QoQ fraction, e.g. 0.3 = 30% acceleration)
    burn_acceleration: float = 0.80

    # Boolean: goes-concern language in filings
    going_concern_flag: float = 1.50

    # debt / total_liquidity
    debt_to_cash_ratio: float = 0.60

    # log(market_cap / total_liquidity) centered at log(3)
    log_market_cap_to_cash_centered: float = -0.30

    # Per month since last capital raise vs 12-month reference
    months_since_last_raise_vs_reference: float = 0.03
    reference_months_since_raise: float = 12.0

    # Boolean: completed a raise in the past 6 months
    recent_financing_flag: float = -0.50

    # Biotech market condition score (1–10), centered at 5
    biotech_market_condition_per_point: float = -0.15

    # Pipeline concentration score (0–1)
    pipeline_concentration: float = 0.30

    # Trial-phase risk mapping coefficient
    trial_phase_risk: float = 0.20


@dataclass
class WeibullParams:
    """
    Baseline Weibull survival curve parameters.

    S0(t) = exp(-( lambda_ * t )^k)

    With lambda_=0.035 and k=1.3, the baseline average biotech has:
      S0(12)  ≈ 70.5%
      S0(24)  ≈ 44.7%
      S0(36)  ≈ 25.7%

    These are UNTRAINED MVP ASSUMPTIONS.
    """
    lambda_: float = 0.035   # baseline hazard scale (rate)
    k: float = 1.30          # Weibull shape (>1 = increasing hazard)


@dataclass
class MilestoneTimingParams:
    """Parameters controlling Gamma-distributed milestone delay modeling."""
    base_delay_factor: float = 1.20          # 20% optimism-bias correction
    base_cv_by_phase: Dict[str, float] = field(default_factory=lambda: {
        "preclinical": 0.55,
        "phase_1": 0.45,
        "phase_2": 0.38,
        "phase_3": 0.30,
        "filed": 0.20,
        "approved": 0.10,
    })
    enrollment_delay_weight: float = 0.50    # up to +50% delay if 0% enrolled
    complexity_delay_weight: float = 0.20    # per complexity score (0–1)
    complexity_cv_weight: float = 0.08       # CV increase per complexity unit
    min_enrollment_remaining_buffer: float = 1.50  # minimum time = enrollment_remaining * 1.5


@dataclass
class RiskThresholds:
    """Classification thresholds for capital-to-catalyst risk."""
    low_cashout_max: float = 0.25       # < 25% cashout risk → Low
    moderate_cashout_max: float = 0.50  # 25–50% → Moderate
    high_cashout_max: float = 0.75      # 50–75% → High
    # > 75% → Critical


@dataclass
class DisclosureThresholds:
    """Jensen-Shannon divergence thresholds for disclosure gap classification."""
    aligned_jsd_max: float = 0.05
    mild_jsd_max: float = 0.15
    material_jsd_max: float = 0.30
    # > 0.30 → Severe


@dataclass
class BurnRegimeThresholds:
    """Thresholds for burn-rate regime classification."""
    stable_max_qoq: float = 0.10          # < 10% QoQ change
    accelerating_max_qoq: float = 0.30    # 10–30% → accelerating
    # > 30% → sharply accelerating


@dataclass
class CatalystLensConfig:
    """Master configuration object for all CatalystLens engines."""
    phase_priors: Dict[str, PhasePosPrior] = field(default_factory=lambda: {
        "preclinical": PhasePosPrior(1.0, 9.0, "~10% prior PoS (1-in-10 preclinical compounds)"),
        "phase_1":     PhasePosPrior(2.0, 8.0, "~20% prior PoS (Phase 1 to approval)"),
        "phase_2":     PhasePosPrior(3.0, 7.0, "~30% prior PoS (Phase 2 to approval)"),
        "phase_3":     PhasePosPrior(5.0, 5.0, "~50% prior PoS (Phase 3 to approval)"),
        "filed":       PhasePosPrior(8.0, 2.0, "~80% prior PoS (NDA/BLA filed)"),
        "approved":    PhasePosPrior(19.0, 1.0, "~95% prior PoS (approved, label expansion)"),
    })
    signal_weights: SignalWeights = field(default_factory=SignalWeights)
    cox_coefficients: CoxCoefficients = field(default_factory=CoxCoefficients)
    weibull_params: WeibullParams = field(default_factory=WeibullParams)
    milestone_timing: MilestoneTimingParams = field(default_factory=MilestoneTimingParams)
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    disclosure_thresholds: DisclosureThresholds = field(default_factory=DisclosureThresholds)
    burn_regime_thresholds: BurnRegimeThresholds = field(default_factory=BurnRegimeThresholds)

    default_n_simulations: int = 10_000
    sensitivity_n_simulations: int = 1_000
    scenario_n_simulations: int = 2_000
    default_random_seed: int = 42
    default_monthly_horizon: int = 48

    # Phase-risk encoding for Cox model (higher = more long-term funding risk)
    trial_phase_risk_map: Dict[str, float] = field(default_factory=lambda: {
        "preclinical": 1.0,
        "phase_1": 0.8,
        "phase_2": 0.4,
        "phase_3": 0.1,
        "filed": -0.1,
        "approved": -0.3,
    })


_DEFAULT_CONFIG: CatalystLensConfig | None = None


def get_default_config() -> CatalystLensConfig:
    """Return a singleton default configuration."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = CatalystLensConfig()
    return _DEFAULT_CONFIG
