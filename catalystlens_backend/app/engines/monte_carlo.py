"""
Central Monte Carlo Engine.

Integrates all CatalystLens sub-engines into a single coherent simulation:

  1. Burn regime detection → burn_acceleration
  2. Solvency model → risk_multiplier, survival curve
  3. Milestone timing → Gamma parameters
  4. Bayesian PoS → Beta posterior parameters
  5. Vectorised Monte Carlo (n_simulations):
       T_fin ~ Cox-Weibull(risk_multiplier)
       T_sci ~ Gamma(alpha, beta_rate)
       PoS   ~ Beta(alpha_post, beta_post)
  6. Capital-to-catalyst gap statistics
  7. Valuation distribution
  8. Scenario analysis (5 scenarios × reduced simulations)
  9. Sensitivity analysis (8 variables × 3 levels × reduced simulations)
  10. Disclosure consistency
  11. Report generation
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import CatalystLensConfig, get_default_config
from app.engines.bayesian_success import (
    build_success_prior_by_phase,
    run_success_probability_analysis,
    sample_success_probability,
    update_beta_posterior,
)
from app.engines.burn_regime import run_burn_regime_analysis
from app.engines.capital_to_catalyst import run_capital_to_catalyst_analysis
from app.engines.cash_path import simulate_cash_path
from app.engines.disclosure_consistency import run_disclosure_consistency_analysis
from app.engines.milestone_timing import (
    estimate_gamma_parameters,
    run_milestone_timing_analysis,
    sample_scientific_milestone_time,
)
from app.engines.report_generator import generate_full_report
from app.engines.multistate import (
    CAUSE_NAMES,
    CAUSE_TO_VALUATION_STATE,
    DEFAULT_CAUSE_SCALES,
    build_cause_lp,
    cif_at_time,
    compute_overall_survival,
    sample_competing_risk,
)
from app.engines.solvency import (
    calculate_monthly_burn,
    calculate_risk_multiplier,
    calculate_simple_runway_months,
    compute_total_liquidity,
    run_solvency_analysis,
    sample_financial_failure_time,
    _compute_linear_predictor,
)
from app.engines.valuation import run_valuation_simulation
from app.engines.dependence import run_dependence_analysis
from app.engines.state_space import StateSpaceParams, run_state_space_analysis
from app.engines.model_averaging import compute_bma
from app.engines.real_options import RealOptionsInput, simulate_real_options_value
from app.engines.risk_attribution import compute_shapley_attribution
from app.engines.robustness import compute_robustness_bounds
from app.engines.value_of_information import run_value_of_information_analysis
from app.models.schemas import (
    AuditRequest,
    AuditResponse,
    CashPathInput,
    CashPathResult,
    ClinicalCatalystInput,
    CompanyFinancialInput,
    DataQualityResult,
    FinancingEventInput,
    FinalSummaryResult,
    ModelVersionInfo,
    BMAResult,
    DependenceAnalysisResult,
    ModelWeightSchema,
    MultiStateResult,
    StateSpaceResult,
    ProvenanceBundle,
    ProvenanceItem,
    RealOptionsResult,
    RobustnessResult,
    RiskAttributionResult,
    ScenarioResult,
    SensitivityPoint,
    ShapleyComponentSchema,
    SignalEVSISchema,
    SimulationConfig,
    SuccessProbabilityInput,
    ValuationInput,
    ValidationSnapshot,
    ValueOfInformationResult,
)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

_SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "Bull Case",
        "description": "Faster enrollment, lower burn, favourable market conditions",
        "enrollment_rate_mult": 1.30,
        "burn_mult": 0.85,
        "market_condition_offset": 2.0,
        "stated_months_mult": 0.90,
        "pos_delta": 0.05,
        "financing_need": "Unlikely",
    },
    {
        "name": "Base Case",
        "description": "Current model inputs, no adjustment",
        "enrollment_rate_mult": 1.00,
        "burn_mult": 1.00,
        "market_condition_offset": 0.0,
        "stated_months_mult": 1.00,
        "pos_delta": 0.00,
        "financing_need": "Possible",
    },
    {
        "name": "Bear Case",
        "description": "Slower enrollment, accelerating burn, weaker market",
        "enrollment_rate_mult": 0.70,
        "burn_mult": 1.20,
        "market_condition_offset": -2.0,
        "stated_months_mult": 1.30,
        "pos_delta": -0.05,
        "financing_need": "Likely",
    },
    {
        "name": "Financing Stress",
        "description": "High burn acceleration plus poor capital markets",
        "enrollment_rate_mult": 0.85,
        "burn_mult": 1.40,
        "market_condition_offset": -3.0,
        "stated_months_mult": 1.15,
        "pos_delta": 0.00,
        "financing_need": "High probability",
    },
    {
        "name": "Clinical Delay",
        "description": "Significant trial delay, current burn trajectory",
        "enrollment_rate_mult": 0.50,
        "burn_mult": 1.05,
        "market_condition_offset": 0.0,
        "stated_months_mult": 1.60,
        "pos_delta": -0.03,
        "financing_need": "Likely",
    },
]


@dataclasses.dataclass(frozen=True)
class RNGStreams:
    """Independent, reproducible random streams for simulation components."""
    cash: np.random.Generator
    financing: np.random.Generator
    science: np.random.Generator
    valuation: np.random.Generator


def spawn_streams(seed: int) -> RNGStreams:
    """
    Spawn deterministic non-overlapping streams from one seed.

    SeedSequence gives exact reproducibility while avoiding accidental coupling
    between cash paths, financing hazards, science timing, and valuation draws.
    """
    seed_sequence = np.random.SeedSequence(seed)
    cash, financing, science, valuation = seed_sequence.spawn(4)
    return RNGStreams(
        cash=np.random.default_rng(cash),
        financing=np.random.default_rng(financing),
        science=np.random.default_rng(science),
        valuation=np.random.default_rng(valuation),
    )


def apply_cash_path_cap(
    t_fin: np.ndarray,
    financial: CompanyFinancialInput,
    sim_cfg: SimulationConfig,
    rng: np.random.Generator,
    planned_financing_events: list[FinancingEventInput] | None = None,
    catalyst_month: float | None = None,
) -> tuple[np.ndarray, CashPathResult]:
    """Apply mechanical cash exhaustion as an upper bound on financial survival samples."""
    cash_path = simulate_cash_path(
        CashPathInput(
            starting_cash=compute_total_liquidity(financial),
            monthly_burn=calculate_monthly_burn(financial.quarterly_operating_cash_burn),
            horizon_months=sim_cfg.monthly_horizon,
            financing_events=planned_financing_events or financial.planned_financing_events,
            catalyst_month=catalyst_month,
        ),
        rng=rng,
    )
    if cash_path.cash_exhaustion_month is None:
        return t_fin, cash_path
    return np.minimum(t_fin, float(cash_path.cash_exhaustion_month)), cash_path


# ---------------------------------------------------------------------------
# Sensitivity variable definitions
# ---------------------------------------------------------------------------

_SENSITIVITY_VARS: List[Dict[str, Any]] = [
    {
        "variable": "monthly_burn",
        "description": "Monthly operating cash burn",
        "low_mult": 0.75,
        "high_mult": 1.35,
        "field": "burn_mult",
    },
    {
        "variable": "stated_months_to_catalyst",
        "description": "Management-stated catalyst timeline",
        "low_mult": 0.80,
        "high_mult": 1.40,
        "field": "stated_months_mult",
    },
    {
        "variable": "enrollment_rate",
        "description": "Monthly trial enrollment rate",
        "low_mult": 0.60,
        "high_mult": 1.50,
        "field": "enrollment_rate_mult",
    },
    {
        "variable": "posterior_pos",
        "description": "Bayesian posterior probability of technical success",
        "low_mult": 0.70,
        "high_mult": 1.30,
        "field": "pos_delta_mult",  # special handling
    },
    {
        "variable": "annual_discount_rate",
        "description": "Risk-adjusted discount rate",
        "low_mult": 0.67,
        "high_mult": 1.50,
        "field": "discount_mult",
    },
    {
        "variable": "dilution_if_refinanced",
        "description": "Expected dilution if company must refinance",
        "low_mult": 0.50,
        "high_mult": 2.00,
        "field": "dilution_mult",
    },
    {
        "variable": "asset_value_success",
        "description": "Asset value on clinical/regulatory success",
        "low_mult": 0.50,
        "high_mult": 2.00,
        "field": "asset_value_mult",
    },
    {
        "variable": "market_condition_score",
        "description": "Biotech financing market condition (1–10)",
        "low_mult": None,
        "high_mult": None,
        "low_val": -2.0,  # offset from base
        "high_val": 2.0,
        "field": "market_offset",
    },
]


# ---------------------------------------------------------------------------
# Internal simulation helpers
# ---------------------------------------------------------------------------

def _run_core_simulation(
    financial: CompanyFinancialInput,
    clinical: ClinicalCatalystInput,
    pos_input: SuccessProbabilityInput,
    valuation: ValuationInput,
    burn_acceleration: float,
    n: int,
    rng: np.random.Generator,
    config: CatalystLensConfig,
    pos_alpha_override: Optional[float] = None,
    pos_beta_override: Optional[float] = None,
    min_sci_months: float = 0.1,
    use_multistate: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run vectorised simulation core.

    pos_alpha_override / pos_beta_override allow the sensitivity engine to inject
    modified Beta parameters without re-running the full signal update pipeline.
    min_sci_months floors t_sci samples at the minimum feasible public readout
    timeline (enrollment + follow-up + cleaning + analysis + disclosure lag).

    use_multistate=False (default): returns (t_fin, t_sci, pos) — backwards compatible.
    use_multistate=True: returns (t_fin, t_sci, pos, cause_array) where cause_array
      is shape (n,) with integer cause IDs 1–7.
    """
    wp = config.weibull_params
    monthly_burn = calculate_monthly_burn(financial.quarterly_operating_cash_burn)
    total_liquidity = compute_total_liquidity(financial)

    lp, _ = _compute_linear_predictor(
        monthly_burn=monthly_burn,
        total_liquidity=total_liquidity,
        burn_acceleration=burn_acceleration,
        market_cap=financial.market_cap,
        debt=financial.debt,
        going_concern_flag=financial.going_concern_flag,
        recent_financing_flag=financial.recent_financing_flag,
        months_since_last_raise=financial.months_since_last_raise,
        biotech_market_condition_score=financial.biotech_market_condition_score,
        pipeline_concentration_score=financial.pipeline_concentration_score,
        trial_phase=clinical.trial_phase,
        coeff=config.cox_coefficients,
        phase_risk_map=config.trial_phase_risk_map,
    )
    rm = calculate_risk_multiplier(lp)

    if pos_alpha_override is not None and pos_beta_override is not None:
        alpha_post, beta_post = pos_alpha_override, pos_beta_override
    else:
        # Use the same hierarchical prior path as run_success_probability_analysis so that
        # the posterior used in valuation exactly matches what the report displays.
        _pos_result = run_success_probability_analysis(pos_input, config)
        alpha_post = _pos_result.alpha_posterior
        beta_post = _pos_result.beta_posterior

    gamma_alpha, gamma_beta_rate, _, _ = estimate_gamma_parameters(clinical, config)

    t_sci = sample_scientific_milestone_time(rng, gamma_alpha, gamma_beta_rate, n, min_months=min_sci_months)
    pos = sample_success_probability(rng, alpha_post, beta_post, n)

    if use_multistate:
        # Build cause-specific linear predictors from the aggregate LP
        cause_lp = build_cause_lp(lp, list(DEFAULT_CAUSE_SCALES.keys()))
        samples = sample_competing_risk(DEFAULT_CAUSE_SCALES, cause_lp, rng, n)
        t_fin = samples[:, 0]
        cause_array = samples[:, 1].astype(np.int32)
        return t_fin, t_sci, pos, cause_array
    else:
        t_fin = sample_financial_failure_time(rng, rm, wp, n)
        return t_fin, t_sci, pos


def _scenario_label_for_ev(ev: float, base_ev: float) -> str:
    if base_ev <= 0:
        return "Cannot compare to base case with non-positive base EV."
    delta_pct = (ev - base_ev) / abs(base_ev) * 100
    if delta_pct > 5:
        return f"EV {delta_pct:+.0f}% vs base; improved financing headroom."
    elif delta_pct < -5:
        return f"EV {delta_pct:+.0f}% vs base; increased financing pressure."
    return "EV broadly in line with base case."


def _run_scenario(
    scenario: Dict[str, Any],
    financial: CompanyFinancialInput,
    clinical: ClinicalCatalystInput,
    pos_input: SuccessProbabilityInput,
    valuation: ValuationInput,
    burn_acceleration: float,
    base_ev: float,
    n: int,
    rng: np.random.Generator,
    config: CatalystLensConfig,
    sim_cfg: SimulationConfig,
) -> ScenarioResult:
    """Run one scenario with modified inputs and return ScenarioResult."""
    fin_mod = copy.deepcopy(financial)
    clin_mod = copy.deepcopy(clinical)
    val_mod = copy.deepcopy(valuation)
    pos_mod = copy.deepcopy(pos_input)

    burn_mult = scenario.get("burn_mult", 1.0)
    stated_mult = scenario.get("stated_months_mult", 1.0)
    enroll_mult = scenario.get("enrollment_rate_mult", 1.0)
    market_offset = scenario.get("market_condition_offset", 0.0)
    pos_delta = scenario.get("pos_delta", 0.0)

    # Mutate financial inputs via dict construction (Pydantic model)
    fin_data = fin_mod.model_dump()
    fin_data["quarterly_operating_cash_burn"] = (
        financial.quarterly_operating_cash_burn * burn_mult
    )
    fin_data["biotech_market_condition_score"] = float(np.clip(
        financial.biotech_market_condition_score + market_offset, 1.0, 10.0
    ))
    fin_mod = CompanyFinancialInput(**fin_data)

    clin_data = clin_mod.model_dump()
    clin_data["stated_months_to_catalyst"] = (
        clinical.stated_months_to_catalyst * stated_mult
    )
    clin_data["enrollment_rate_per_month"] = max(
        0.1, clinical.enrollment_rate_per_month * enroll_mult
    )
    clin_mod = ClinicalCatalystInput(**clin_data)

    # Adjust PoS by clamping posterior mean
    base_alpha, base_beta = build_success_prior_by_phase(
        pos_input.trial_phase, config,
        pos_input.custom_alpha_prior, pos_input.custom_beta_prior,
    )
    a_post, b_post, _, _ = update_beta_posterior(
        base_alpha, base_beta,
        pos_input.positive_signals, pos_input.negative_signals, config,
    )
    base_pos_mean = a_post / (a_post + b_post)
    new_pos_mean = float(np.clip(base_pos_mean + pos_delta, 0.01, 0.99))

    t_fin, t_sci, pos_samples = _run_core_simulation(
        fin_mod, clin_mod, pos_mod, val_mod,
        burn_acceleration * burn_mult, n, rng, config,
    )
    t_fin, _ = apply_cash_path_cap(t_fin, fin_mod, sim_cfg, rng)
    # Override pos_samples with scenario-adjusted mean (rescale beta)
    pos_samples = np.clip(pos_samples * (new_pos_mean / max(base_pos_mean, 0.01)), 0.01, 0.99)

    val_result = run_valuation_simulation(
        t_sci, t_fin, pos_samples, val_mod, rng,
        market_condition_score=float(fin_mod.biotech_market_condition_score),
        config=config,
    )
    ctc_result = run_capital_to_catalyst_analysis(t_sci, t_fin, config)

    return ScenarioResult(
        scenario_name=scenario["name"],
        description=scenario["description"],
        catalyst_timing_months=round(float(np.median(t_sci)), 1),
        burn_assumption=f"{burn_mult:.0%} of base burn",
        pos_assumption=round(new_pos_mean, 3),
        financing_need=scenario.get("financing_need", "Unknown"),
        expected_value=round(val_result.financing_adjusted_rnpv, 2),
        probability_cashout_before_catalyst=round(
            ctc_result.probability_cashout_before_catalyst, 4
        ),
        interpretation=_scenario_label_for_ev(val_result.financing_adjusted_rnpv, base_ev),
    )


def _run_sensitivity(
    variable_def: Dict[str, Any],
    financial: CompanyFinancialInput,
    clinical: ClinicalCatalystInput,
    pos_input: SuccessProbabilityInput,
    valuation: ValuationInput,
    burn_acceleration: float,
    base_cashout_prob: float,
    base_ev: float,
    n: int,
    rng: np.random.Generator,
    config: CatalystLensConfig,
    sim_cfg: SimulationConfig,
    var_seed: int = 0,
) -> SensitivityPoint:
    """
    Compute sensitivity of cashout probability and EV to one variable.

    Common Random Numbers (CRN): all three levels (low/base/high) use the same
    underlying random draws so directional comparisons are not confounded by
    Monte Carlo noise. Each variable gets its own independent seeded sub-stream.
    """
    # Spawn a fresh, deterministic RNG for this variable — independent from the
    # shared simulation rng so callers remain reproducible.
    var_rng = np.random.default_rng(var_seed)

    def _sim(fin: CompanyFinancialInput, clin: ClinicalCatalystInput,
             val: ValuationInput, ba: float,
             pos_alpha_override: Optional[float] = None,
             pos_beta_override: Optional[float] = None) -> Tuple[float, float]:
        # Save and restore RNG state so every level uses the same random draws (CRN).
        saved_state = var_rng.bit_generator.state
        t_fin, t_sci, pos = _run_core_simulation(
            fin, clin, pos_input, val, ba, n, var_rng, config,
            pos_alpha_override=pos_alpha_override,
            pos_beta_override=pos_beta_override,
        )
        t_fin, _ = apply_cash_path_cap(t_fin, fin, sim_cfg, var_rng)
        ctc = run_capital_to_catalyst_analysis(t_sci, t_fin, config)
        val_r = run_valuation_simulation(
            t_sci, t_fin, pos, val, var_rng,
            market_condition_score=float(fin.biotech_market_condition_score),
            config=config,
        )
        # Restore so the next level starts from the same draws.
        var_rng.bit_generator.state = saved_state
        return ctc.probability_cashout_before_catalyst, val_r.financing_adjusted_rnpv

    field = variable_def["field"]
    var = variable_def["variable"]

    low_label = high_label = base_label = ""
    results: List[Tuple[float, float]] = []

    for level, label_sfx in [("low", "Low"), ("base", "Base"), ("high", "High")]:
        fin_mod = financial
        clin_mod = clinical
        val_mod = valuation
        ba_mod = burn_acceleration

        if field == "burn_mult":
            mult = variable_def.get("low_mult" if level == "low" else "high_mult", 1.0)
            if level == "base":
                mult = 1.0
            fin_data = fin_mod.model_dump()
            fin_data["quarterly_operating_cash_burn"] = financial.quarterly_operating_cash_burn * mult
            fin_mod = CompanyFinancialInput(**fin_data)
            ba_mod = burn_acceleration * mult
            lbl = f"{mult:.0%} burn"

        elif field == "stated_months_mult":
            mult = variable_def.get("low_mult" if level == "low" else "high_mult", 1.0)
            if level == "base":
                mult = 1.0
            clin_data = clin_mod.model_dump()
            clin_data["stated_months_to_catalyst"] = clinical.stated_months_to_catalyst * mult
            clin_mod = ClinicalCatalystInput(**clin_data)
            lbl = f"{clinical.stated_months_to_catalyst * mult:.0f}mo timeline"

        elif field == "enrollment_rate_mult":
            mult = variable_def.get("low_mult" if level == "low" else "high_mult", 1.0)
            if level == "base":
                mult = 1.0
            clin_data = clin_mod.model_dump()
            clin_data["enrollment_rate_per_month"] = max(0.1, clinical.enrollment_rate_per_month * mult)
            clin_mod = ClinicalCatalystInput(**clin_data)
            lbl = f"{mult:.0%} enroll rate"

        elif field == "pos_delta_mult":
            # Use hierarchical posterior (same path as run_success_probability_analysis).
            _base_pos_result = run_success_probability_analysis(pos_input, config)
            a_po = _base_pos_result.alpha_posterior
            b_po = _base_pos_result.beta_posterior
            base_pos = a_po / (a_po + b_po)
            total_concentration = a_po + b_po
            mult = variable_def.get("low_mult" if level == "low" else "high_mult", 1.0)
            if level == "base":
                mult = 1.0
            new_pos_mean = float(np.clip(base_pos * mult, 0.01, 0.99))
            # Derive shifted alpha/beta preserving total concentration (same certainty, shifted mean)
            new_alpha = new_pos_mean * total_concentration
            new_beta = (1.0 - new_pos_mean) * total_concentration
            lbl = f"{new_pos_mean:.0%} PoS"
            cp, ev = _sim(fin_mod, clin_mod, val_mod, ba_mod,
                          pos_alpha_override=new_alpha, pos_beta_override=new_beta)
            results.append((cp, ev))
            if level == "low":
                low_label = lbl
            elif level == "base":
                base_label = lbl
            else:
                high_label = lbl
            continue

        elif field == "discount_mult":
            mult = variable_def.get("low_mult" if level == "low" else "high_mult", 1.0)
            if level == "base":
                mult = 1.0
            val_data = val_mod.model_dump()
            val_data["annual_discount_rate"] = float(np.clip(
                valuation.annual_discount_rate * mult, 0.01, 0.99
            ))
            val_mod = ValuationInput(**val_data)
            lbl = f"{val_data['annual_discount_rate']:.0%} discount"

        elif field == "dilution_mult":
            mult = variable_def.get("low_mult" if level == "low" else "high_mult", 1.0)
            if level == "base":
                mult = 1.0
            val_data = val_mod.model_dump()
            val_data["expected_dilution_if_refinanced"] = float(np.clip(
                valuation.expected_dilution_if_refinanced * mult, 0.01, 0.95
            ))
            val_mod = ValuationInput(**val_data)
            lbl = f"{val_data['expected_dilution_if_refinanced']:.0%} dilution"

        elif field == "asset_value_mult":
            mult = variable_def.get("low_mult" if level == "low" else "high_mult", 1.0)
            if level == "base":
                mult = 1.0
            val_data = val_mod.model_dump()
            val_data["asset_value_success"] = valuation.asset_value_success * mult
            val_mod = ValuationInput(**val_data)
            lbl = f"${val_data['asset_value_success']/1e6:.0f}M asset value"

        elif field == "market_offset":
            offset = variable_def.get("low_val" if level == "low" else "high_val", 0.0)
            if level == "base":
                offset = 0.0
            fin_data = fin_mod.model_dump()
            fin_data["biotech_market_condition_score"] = float(np.clip(
                financial.biotech_market_condition_score + offset, 1.0, 10.0
            ))
            fin_mod = CompanyFinancialInput(**fin_data)
            lbl = f"market={fin_data['biotech_market_condition_score']:.0f}/10"

        else:
            lbl = level
            results.append((base_cashout_prob, base_ev))
            if level == "low":
                low_label = lbl
            elif level == "base":
                base_label = lbl
            else:
                high_label = lbl
            continue

        cp, ev = _sim(fin_mod, clin_mod, val_mod, ba_mod)
        results.append((cp, ev))
        if level == "low":
            low_label = lbl
        elif level == "base":
            base_label = lbl
        else:
            high_label = lbl

    return SensitivityPoint(
        variable=var,
        low_label=low_label,
        base_label=base_label,
        high_label=high_label,
        low_cashout_prob=round(results[0][0], 4),
        base_cashout_prob=round(results[1][0], 4),
        high_cashout_prob=round(results[2][0], 4),
        low_expected_value=round(results[0][1], 2),
        base_expected_value=round(results[1][1], 2),
        high_expected_value=round(results[2][1], 2),
    )


# ---------------------------------------------------------------------------
# Diligence questions generator
# ---------------------------------------------------------------------------

def _generate_diligence_questions(
    cashout_prob: float,
    burn_regime: str,
    enrollment_fraction: float,
    risk_class: str,
    pos: float,
    simple_runway: float,
    stated_months: float,
    jsd: float,
) -> List[str]:
    questions = [
        "Does management's stated cash runway incorporate all planned trial expansion costs, including site activation and patient recruitment expenses?",
        "What financing options (ATM, PIPE, royalty monetisation, partnership milestone) are available if capital markets conditions deteriorate?",
        "Is the primary endpoint powered for statistically meaningful efficacy detection, or only exploratory signal generation?",
        "Has the company disclosed any interim analysis triggers, futility boundaries, or protocol amendments that could alter the catalyst timeline?",
    ]

    if cashout_prob > 0.40:
        questions.append(
            "Given elevated modelled cashout risk, what specific triggers would prompt management to pursue bridge financing or a strategic transaction?"
        )
        questions.append(
            "Does the company have access to non-dilutive capital sources (grants, BARDA, licensing revenue) that could extend the runway?"
        )

    if burn_regime in ("accelerating burn", "sharply accelerating burn"):
        questions.append(
            "What drove recent burn acceleration? Does the current burn trajectory reflect trial expansion costs, or ongoing Phase 1→2 dose escalation transition costs?"
        )
        questions.append(
            "Is the stated cash runway guidance based on the most recent quarterly burn, or on a prior quarter's burn rate?"
        )

    if enrollment_fraction < 0.50:
        questions.append(
            "What enrollment rate is required to hit the stated readout window, and is that rate consistent with current site activation?"
        )
        questions.append(
            "Are there protocol amendments, site additions, or patient population changes pending that could affect enrollment pace?"
        )

    if pos < 0.30:
        questions.append(
            "Given the limited positive signal base, what specific data from the current trial could materially de-risk technical failure before the next capital raise?"
        )

    if simple_runway < stated_months:
        questions.append(
            "Simple runway is shorter than the stated catalyst timeline. What financing assumptions does management embed in its stated runway guidance?"
        )

    if jsd > 0.15:
        questions.append(
            "There is a material divergence between management's narrative framing and the structured audit. What specific evidence supports management's stated confidence levels?"
        )

    questions.append(
        "Does the company have cash to reach data release (public disclosure), not merely primary completion (last patient last visit)?"
    )
    questions.append(
        "Are there material CMC, manufacturing, or regulatory preparedness costs anticipated between primary completion and NDA/BLA submission?"
    )

    return questions


# ---------------------------------------------------------------------------
# Data quality scoring
# ---------------------------------------------------------------------------

_EXPECTED_DISCLOSURE_CATEGORIES = {
    "runway_strength", "clinical_timeline_confidence", "dilution_risk",
    "trial_maturity", "endpoint_strength", "pipeline_diversification",
}


def _compute_data_quality(request: AuditRequest) -> DataQualityResult:
    limitations: List[str] = []

    # Financial completeness
    fin_score = 1.0
    if not request.financial.quarterly_burn_history:
        fin_score -= 0.25
        limitations.append("No quarterly burn history supplied; burn regime detection is limited.")
    if request.financial.going_concern_flag is False and request.financial.months_since_last_raise == 12.0:
        fin_score -= 0.05
        limitations.append("months_since_last_raise at default (12); verify against actual raise date.")
    if request.financial.cash_on_hand + request.financial.marketable_securities == 0:
        fin_score -= 0.30
        limitations.append("Total liquidity is zero; model results are not meaningful.")
    if request.financial.market_cap < (request.financial.cash_on_hand + request.financial.marketable_securities):
        limitations.append("Market cap is below reported cash — negative enterprise value. Verify inputs.")

    # Clinical completeness
    clin_score = 1.0
    enroll_frac = request.clinical.enrollment_completed / request.clinical.enrollment_target
    if enroll_frac == 0:
        clin_score -= 0.15
        limitations.append("Zero enrollment completed; trial may be pre-recruiting or very early stage.")
    if request.clinical.trial_status in ("suspended", "withdrawn"):
        clin_score -= 0.30
        limitations.append(f"Trial status is '{request.clinical.trial_status}'; catalyst timing is highly uncertain.")
    if request.clinical.enrollment_rate_per_month < 1.0:
        clin_score -= 0.10
        limitations.append("Enrollment rate < 1 patient/month; verify against ClinicalTrials.gov accrual data.")

    # Disclosure completeness
    narrative_cats = set(request.disclosure.company_narrative_distribution.keys())
    audit_cats = set(request.disclosure.structured_audit_distribution.keys())
    missing_from_narrative = _EXPECTED_DISCLOSURE_CATEGORIES - narrative_cats
    missing_from_audit = _EXPECTED_DISCLOSURE_CATEGORIES - audit_cats
    missing_total = len(missing_from_narrative) + len(missing_from_audit)
    max_missing = len(_EXPECTED_DISCLOSURE_CATEGORIES) * 2
    disc_score = max(0.0, 1.0 - missing_total / max_missing)
    if missing_from_narrative:
        limitations.append(
            f"Missing company narrative disclosure categories: {', '.join(sorted(missing_from_narrative))}."
        )
    if missing_from_audit:
        limitations.append(
            f"Missing structured audit disclosure categories: {', '.join(sorted(missing_from_audit))}."
        )

    total_liquidity = request.financial.cash_on_hand + request.financial.marketable_securities
    force_overall_cap: float | None = None
    if total_liquidity == 0:
        fin_score = min(fin_score, 0.25)
        force_overall_cap = 0.50
    if not request.financial.quarterly_burn_history:
        fin_score = min(fin_score, 0.80)

    overall = (fin_score + clin_score + disc_score) / 3.0
    component_min = min(fin_score, clin_score, disc_score)
    if component_min <= 0.25:
        overall = min(overall, 0.55)
    elif component_min <= 0.50:
        overall = min(overall, 0.70)
    if force_overall_cap is not None:
        overall = min(overall, force_overall_cap)
    overall = float(max(0.0, min(1.0, overall)))
    fin_score = float(max(0.0, min(1.0, fin_score)))
    clin_score = float(max(0.0, min(1.0, clin_score)))
    disc_score = float(max(0.0, min(1.0, disc_score)))

    quality_label: str
    if overall >= 0.80:
        quality_label = "high"
    elif overall >= 0.55:
        quality_label = "moderate"
    else:
        quality_label = "low"

    # Evidence quality is separate from completeness.
    # Manual inputs can be 100% complete but still have low evidence quality.
    # Source-traced provenance (SEC, ClinicalTrials.gov) raises evidence quality.
    # For now, all inputs are manual; source ETL integration will upgrade this.
    evidence_quality: str = "low"
    evidence_note = "All inputs are manual; no SEC/ClinicalTrials.gov source tracing applied."

    return DataQualityResult(
        financial_data_completeness=round(fin_score, 2),
        clinical_data_completeness=round(clin_score, 2),
        disclosure_data_completeness=round(disc_score, 2),
        overall_completeness=round(overall, 2),
        primary_limitations=limitations,
        data_quality_score=quality_label,
        evidence_quality_score=evidence_quality,
        evidence_quality_note=evidence_note,
    )


def _build_model_version(config: CatalystLensConfig, sim_cfg: SimulationConfig) -> ModelVersionInfo:
    config_payload = {
        "config": dataclasses.asdict(config),
        "simulation": sim_cfg.model_dump(),
    }
    config_repr = json.dumps(config_payload, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(config_repr.encode()).hexdigest()[:12]
    return ModelVersionInfo(
        backend_version="0.1.0",
        name="catalystlens-backend",
        semver="0.1.0",
        artifact_id="mvp_assumption_engine",
        coefficient_set="mvp_untrained_v1",
        n_simulations=sim_cfg.n_simulations,
        random_seed=sim_cfg.random_seed,
        config_hash=config_hash,
        training_cutoff_date=None,
        data_snapshot_ids=[],
    )


def _build_manual_provenance(request: AuditRequest) -> ProvenanceBundle:
    """Build explicit manual-input provenance placeholders until source ETL exists."""
    financial_items = [
        ProvenanceItem(field="cash_on_hand", value=request.financial.cash_on_hand),
        ProvenanceItem(field="marketable_securities", value=request.financial.marketable_securities),
        ProvenanceItem(
            field="quarterly_operating_cash_burn",
            value=request.financial.quarterly_operating_cash_burn,
        ),
        ProvenanceItem(field="market_cap", value=request.financial.market_cap),
    ]
    clinical_items = [
        ProvenanceItem(field="trial_phase", value=request.clinical.trial_phase),
        ProvenanceItem(field="trial_status", value=request.clinical.trial_status),
        ProvenanceItem(
            field="stated_months_to_catalyst",
            value=request.clinical.stated_months_to_catalyst,
        ),
        ProvenanceItem(field="enrollment_target", value=request.clinical.enrollment_target),
        ProvenanceItem(field="enrollment_completed", value=request.clinical.enrollment_completed),
    ]
    return ProvenanceBundle(
        financial_inputs=financial_items,
        clinical_inputs=clinical_items,
        claims=[],
        provenance_status="manual_inputs_unverified",
    )


def _build_validation_snapshot() -> ValidationSnapshot:
    return ValidationSnapshot(
        solvency_calibration_status="research_mode",
        pos_ppc_status="not_available",
        timing_interval_coverage_status="not_available",
        notes=[
            "No historical labeled training dataset supplied; calibration metrics are unavailable.",
            "Outputs remain assumption-based research-mode estimates, not validated predictions.",
        ],
    )


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def _classify_primary_risk(
    cashout_prob: float,
    pos: float,
    jsd: float,
    burn_regime: str,
) -> Tuple[str, str]:
    """Return (primary_risk_factor, secondary_risk_factor)."""
    scores = {
        "Capital / Financing": cashout_prob,
        "Technical / Scientific": 1.0 - pos,
        "Disclosure / Narrative": min(jsd * 3, 1.0),
        "Burn Trajectory": (
            0.8 if "sharply" in burn_regime else
            0.5 if "accelerating" in burn_regime else
            0.1
        ),
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[0][0], ranked[1][0]


# ---------------------------------------------------------------------------
# Main engine entry point
# ---------------------------------------------------------------------------

def run_full_audit(
    request: AuditRequest,
    config: CatalystLensConfig | None = None,
) -> AuditResponse:
    """
    Execute the complete CatalystLens audit.

    This is the primary API integration point.
    """
    # Always deepcopy to avoid mutating the global singleton across requests
    config = copy.deepcopy(get_default_config() if config is None else config)

    sim_cfg: SimulationConfig = request.simulation
    n = sim_cfg.n_simulations
    streams = spawn_streams(sim_cfg.random_seed)
    rng = streams.science

    # Override Weibull params from simulation config (safe: working on a copy)
    config.weibull_params.lambda_ = sim_cfg.baseline_lambda
    config.weibull_params.k = sim_cfg.baseline_k

    financial = request.financial
    clinical = request.clinical
    pos_input = request.success_probability
    valuation = request.valuation
    disclosure = request.disclosure

    # ---- Step 1: Burn regime ----
    burn_result = run_burn_regime_analysis(financial, config)

    # ---- Step 2: Solvency ----
    solvency_result = run_solvency_analysis(
        financial,
        burn_acceleration=burn_result.burn_acceleration,
        trial_phase=clinical.trial_phase,
        config=config,
        monthly_horizon=sim_cfg.monthly_horizon,
    )

    # ---- Step 3: Milestone timing ----
    milestone_result = run_milestone_timing_analysis(clinical, config)

    # ---- Step 4: Bayesian PoS ----
    pos_result = run_success_probability_analysis(pos_input, config)

    # ---- Step 5: Disclosure consistency ----
    disclosure_result = run_disclosure_consistency_analysis(disclosure, config)

    # ---- Step 6: Monte Carlo core simulation ----
    use_ms = sim_cfg.use_multistate
    if use_ms:
        t_fin, t_sci, pos_samples, cause_array = _run_core_simulation(
            financial, clinical, pos_input, valuation,
            burn_result.burn_acceleration, n, streams.financing, config,
            min_sci_months=milestone_result.public_readout_months,
            use_multistate=True,
        )
    else:
        t_fin, t_sci, pos_samples = _run_core_simulation(
            financial, clinical, pos_input, valuation,
            burn_result.burn_acceleration, n, streams.financing, config,
            min_sci_months=milestone_result.public_readout_months,
        )
        cause_array = None
    t_fin, cash_path_result = apply_cash_path_cap(
        t_fin, financial, sim_cfg, streams.cash,
        catalyst_month=milestone_result.public_readout_months,
    )

    # ---- Step 7: Capital-to-catalyst ----
    ctc_result = run_capital_to_catalyst_analysis(t_sci, t_fin, config)

    # ---- Step 8: Valuation ----
    val_result = run_valuation_simulation(
        t_sci, t_fin, pos_samples, valuation, streams.valuation,
        market_condition_score=float(financial.biotech_market_condition_score),
        config=config,
    )

    # ---- Step 9: Scenario analysis ----
    n_scen = config.scenario_n_simulations
    base_ev = val_result.financing_adjusted_rnpv
    scenarios: List[ScenarioResult] = []
    for sc_def in _SCENARIOS:
        sc_result = _run_scenario(
            sc_def, financial, clinical, pos_input, valuation,
            burn_result.burn_acceleration, base_ev, n_scen, rng, config, sim_cfg,
        )
        scenarios.append(sc_result)

    # ---- Step 10: Sensitivity analysis ----
    n_sens = config.sensitivity_n_simulations
    base_cashout = ctc_result.probability_cashout_before_catalyst
    sensitivity: List[SensitivityPoint] = []
    for var_idx, var_def in enumerate(_SENSITIVITY_VARS):
        # Each variable gets a unique, reproducible seed for CRN across low/base/high levels.
        var_seed = sim_cfg.random_seed ^ (var_idx * 0x9E3779B9 & 0xFFFFFFFF)
        sp = _run_sensitivity(
            var_def, financial, clinical, pos_input, valuation,
            burn_result.burn_acceleration, base_cashout, base_ev,
            n_sens, rng, config, sim_cfg, var_seed=var_seed,
        )
        sensitivity.append(sp)

    # ---- Step 11: Final summary ----
    primary_risk, secondary_risk = _classify_primary_risk(
        cashout_prob=ctc_result.probability_cashout_before_catalyst,
        pos=pos_result.posterior_mean,
        jsd=disclosure_result.jsd_score,
        burn_regime=burn_result.regime,
    )

    diligence_questions = _generate_diligence_questions(
        cashout_prob=ctc_result.probability_cashout_before_catalyst,
        burn_regime=burn_result.regime,
        enrollment_fraction=milestone_result.enrollment_fraction,
        risk_class=ctc_result.risk_classification,
        pos=pos_result.posterior_mean,
        simple_runway=solvency_result.simple_runway_months,
        stated_months=clinical.stated_months_to_catalyst,
        jsd=disclosure_result.jsd_score,
    )

    key_finding = (
        f"Under current model assumptions, there is a "
        f"{ctc_result.probability_cashout_before_catalyst:.1%} modelled probability "
        f"that {financial.company_name}'s capital is exhausted before the stated "
        f"{clinical.catalyst_type.replace('_', ' ')} milestone. "
        f"The primary risk driver is {primary_risk.lower()}, not exclusively scientific."
    )

    final_summary = FinalSummaryResult(
        risk_classification=ctc_result.risk_classification,
        probability_cashout_before_catalyst=ctc_result.probability_cashout_before_catalyst,
        probability_reaches_catalyst=ctc_result.probability_reaches_catalyst,
        posterior_pos=pos_result.posterior_mean,
        expected_value=val_result.mean_value,
        financing_adjusted_rnpv=val_result.financing_adjusted_rnpv,
        primary_risk_factor=primary_risk,
        secondary_risk_factor=secondary_risk,
        key_finding=key_finding,
        scenarios=scenarios,
        sensitivity=sensitivity,
        diligence_questions=diligence_questions,
    )

    # ---- Step 12: Multi-state result (optional) ----
    multi_state_result: MultiStateResult | None = None
    if use_ms and cause_array is not None:
        # Compute aggregate LP for CIF
        monthly_burn_ms = calculate_monthly_burn(financial.quarterly_operating_cash_burn)
        total_liq_ms = compute_total_liquidity(financial)
        _lp_val, _ = _compute_linear_predictor(
            monthly_burn=monthly_burn_ms,
            total_liquidity=total_liq_ms,
            burn_acceleration=burn_result.burn_acceleration,
            market_cap=financial.market_cap,
            debt=financial.debt,
            going_concern_flag=financial.going_concern_flag,
            recent_financing_flag=financial.recent_financing_flag,
            months_since_last_raise=financial.months_since_last_raise,
            biotech_market_condition_score=financial.biotech_market_condition_score,
            pipeline_concentration_score=financial.pipeline_concentration_score,
            trial_phase=clinical.trial_phase,
            coeff=config.cox_coefficients,
            phase_risk_map=config.trial_phase_risk_map,
        )
        cause_lp_ms = build_cause_lp(_lp_val, list(DEFAULT_CAUSE_SCALES.keys()))
        # Absorbing state probabilities from simulation
        absorbed_mask = t_fin <= sim_cfg.monthly_horizon
        absorbed_causes = cause_array[absorbed_mask]
        abs_probs: dict[str, float] = {}
        for cid, name in CAUSE_NAMES.items():
            abs_probs[name] = round(float(np.sum(absorbed_causes == cid)) / n, 4)
        still_operating = round(float(np.sum(~absorbed_mask)) / n, 4)
        # Median transition time
        absorbed_times = t_fin[absorbed_mask]
        median_t = float(np.median(absorbed_times)) if len(absorbed_times) > 0 else None
        # CIF at catalyst month
        cat_month = milestone_result.public_readout_months
        cif_cat = cif_at_time(cat_month, DEFAULT_CAUSE_SCALES, cause_lp_ms)
        cif_cat_named = {CAUSE_NAMES[cid]: round(v, 4) for cid, v in cif_cat.items()}
        s_cat = float(compute_overall_survival(
            np.array([cat_month]), DEFAULT_CAUSE_SCALES, cause_lp_ms
        )[0])
        multi_state_result = MultiStateResult(
            absorbing_state_probs=abs_probs,
            overall_survival_at_horizon=still_operating,
            median_transition_time=round(median_t, 1) if median_t else None,
            cif_at_catalyst_month=cif_cat_named,
            overall_survival_at_catalyst_month=round(s_cat, 4),
            model_assumptions=[
                "Multi-state competing-risk Weibull model with 7 absorbing causes.",
                "Cause LPs derived from aggregate Cox LP via differential weighting.",
                "Scale parameters are MVP defaults; not fit to historical outcome data.",
            ],
            method_status="uncalibrated_assumption",
        )

    # ---- Step 13: Value of Information ----
    voi_raw = run_value_of_information_analysis(
        alpha_posterior=pos_result.alpha_posterior,
        beta_posterior=pos_result.beta_posterior,
        financing_adjusted_rnpv=val_result.financing_adjusted_rnpv,
        upside_value=float(valuation.asset_value_success),
        capital_required=max(float(cash_path_result.capital_needed_to_reach_catalyst or 0.0), 0.0),
    )
    voi_result = ValueOfInformationResult(
        evpi_dollars=voi_raw.evpi_dollars,
        evpi_pct_of_ev=voi_raw.evpi_pct_of_ev,
        evpi_interpretation=voi_raw.evpi_interpretation,
        per_signal_evsi=[
            SignalEVSISchema(
                signal_name=s.signal_name,
                description=s.description,
                category=s.category,
                signal_weight=s.signal_weight,
                evsi_dollars=s.evsi_dollars,
                ev_if_positive=s.ev_if_positive,
                ev_if_negative=s.ev_if_negative,
                p_positive=s.p_positive,
            )
            for s in voi_raw.per_signal_evsi
        ],
        top_diligence_priority=voi_raw.top_diligence_priority,
        total_observable_evsi=voi_raw.total_observable_evsi,
        methodology_note=voi_raw.methodology_note,
        method_status=voi_raw.method_status,
    )

    # ---- Step 14: Real-options valuation ----
    def _phase_aware_exercise_cost(phase: str, asset_value_success: float) -> float:
        # MVP defaults for next-stage investment required to exercise the development option.
        phase_costs = {
            "preclinical": 25_000_000.0,
            "phase_1": 50_000_000.0,
            "phase_2": 150_000_000.0,
            "phase_3": 75_000_000.0,
            "filed": 25_000_000.0,
            "approved": 0.0,
        }
        return min(phase_costs.get(phase, 75_000_000.0), max(asset_value_success * 0.75, 0.0))

    ro_rng = np.random.default_rng(sim_cfg.random_seed ^ 0xDEADBEEF)
    ro_exercise_cost = _phase_aware_exercise_cost(
        clinical.trial_phase,
        float(valuation.asset_value_success),
    )
    ro_input = RealOptionsInput(
        asset_value_success=float(valuation.asset_value_success),
        exercise_cost=ro_exercise_cost,
        asset_volatility=0.60,
        annual_discount_rate=float(valuation.annual_discount_rate),
        pos_mean=float(pos_result.posterior_mean),
        financing_state_probabilities={
            "funded": val_result.p_funded_through_catalyst,
            "clean_refinancing": val_result.p_refinancing_success,
            "distressed_refinancing": val_result.p_distressed_financing,
            "program_discontinuation": val_result.p_program_discontinuation,
        },
        clean_refinancing_dilution=float(valuation.expected_dilution_if_refinanced),
        distressed_refinancing_dilution=min(float(valuation.expected_dilution_if_refinanced) * 2.0, 1.0),
    )
    ro_raw = simulate_real_options_value(t_sci, pos_samples, ro_input, ro_rng)
    real_options_result = RealOptionsResult(
        rov_mean=ro_raw.rov_mean,
        rov_median=ro_raw.rov_median,
        rov_p5=ro_raw.rov_p5,
        rov_p95=ro_raw.rov_p95,
        rnpv_static=ro_raw.rnpv_static,
        real_options_premium=ro_raw.real_options_premium,
        real_options_premium_pct=ro_raw.real_options_premium_pct,
        abandonment_value=ro_raw.abandonment_value,
        financing_adjusted_rov=ro_raw.financing_adjusted_rov,
        exercise_cost=ro_raw.exercise_cost,
        asset_volatility=ro_raw.asset_volatility,
        model_assumptions=ro_raw.model_assumptions,
        method_status=ro_raw.method_status,
    )

    # ---- Step 15: Shapley risk attribution ----
    shapley_rng = np.random.default_rng(sim_cfg.random_seed ^ 0xCAFE0001)
    shapley_raw = compute_shapley_attribution(
        sensitivity_rows=sensitivity,
        total_cashout_prob=ctc_result.probability_cashout_before_catalyst,
        total_ev=val_result.financing_adjusted_rnpv,
        n_permutations=64,
        rng=shapley_rng,
    )
    risk_attribution_result = RiskAttributionResult(
        components=[
            ShapleyComponentSchema(
                driver=c.driver,
                description=c.description,
                cashout_prob_shapley=c.cashout_prob_shapley,
                ev_shapley=c.ev_shapley,
                rank=c.rank,
            )
            for c in shapley_raw.components
        ],
        total_cashout_prob=shapley_raw.total_cashout_prob,
        total_ev=shapley_raw.total_ev,
        explained_cashout_prob=shapley_raw.explained_cashout_prob,
        explained_ev=shapley_raw.explained_ev,
        methodology_note=shapley_raw.methodology_note,
        method_status=shapley_raw.method_status,
    )

    # ---- Step 16: Distributional robustness ----
    robustness_raw = compute_robustness_bounds(
        t_fin=t_fin,
        t_sci=t_sci,
        pos_samples=pos_samples,
        nominal_cashout_prob=ctc_result.probability_cashout_before_catalyst,
        nominal_ev=val_result.financing_adjusted_rnpv,
    )
    robustness_result = RobustnessResult(
        nominal_cashout_prob=robustness_raw.nominal_cashout_prob,
        nominal_ev=robustness_raw.nominal_ev,
        worst_case_cashout_prob_e05=robustness_raw.worst_case_cashout_prob_e05,
        worst_case_cashout_prob_e10=robustness_raw.worst_case_cashout_prob_e10,
        worst_case_cashout_prob_e20=robustness_raw.worst_case_cashout_prob_e20,
        worst_case_ev_e05=robustness_raw.worst_case_ev_e05,
        worst_case_ev_e10=robustness_raw.worst_case_ev_e10,
        worst_case_ev_e20=robustness_raw.worst_case_ev_e20,
        best_case_cashout_prob_e10=robustness_raw.best_case_cashout_prob_e10,
        best_case_ev_e10=robustness_raw.best_case_ev_e10,
        robustness_interpretation=robustness_raw.robustness_interpretation,
        methodology_note=robustness_raw.methodology_note,
        method_status=robustness_raw.method_status,
    )

    # ---- Step 17: Bayesian model averaging ----
    monthly_burn_bma = calculate_monthly_burn(financial.quarterly_operating_cash_burn)
    total_liq_bma = compute_total_liquidity(financial)
    simple_runway_bma = total_liq_bma / monthly_burn_bma if monthly_burn_bma > 0 else 12.0
    bma_raw = compute_bma(
        simple_runway=simple_runway_bma,
        risk_multiplier=float(solvency_result.risk_multiplier),
        base_cashout_prob=ctc_result.probability_cashout_before_catalyst,
        base_ev=val_result.financing_adjusted_rnpv,
    )
    bma_result = BMAResult(
        bma_cashout_prob=bma_raw.bma_cashout_prob,
        bma_ev=bma_raw.bma_ev,
        model_weights=[
            ModelWeightSchema(
                k=mw.k, lambda_=mw.lambda_,
                posterior_weight=mw.posterior_weight,
                model_cashout_prob=mw.model_cashout_prob,
                model_ev=mw.model_ev,
            )
            for mw in bma_raw.model_weights
        ],
        effective_n_models=bma_raw.effective_n_models,
        highest_weight_model_k=bma_raw.highest_weight_model_k,
        highest_weight_model_lambda=bma_raw.highest_weight_model_lambda,
        methodology_note=bma_raw.methodology_note,
        method_status=bma_raw.method_status,
    )

    # ---- Step 18: Copula dependence analysis ----
    dep_rng = np.random.default_rng(sim_cfg.random_seed ^ 0xFADED000)
    dep_raw = run_dependence_analysis(t_fin, t_sci, dep_rng)
    dependence_result = DependenceAnalysisResult(
        base_cashout_prob=dep_raw.base_cashout_prob,
        positive_rho_cashout_prob=dep_raw.positive_rho.copula_cashout_prob,
        positive_rho_dependence_effect=dep_raw.positive_rho.dependence_effect,
        positive_rho_interpretation=dep_raw.positive_rho.interpretation,
        negative_rho_cashout_prob=dep_raw.negative_copula_cashout_prob,
        negative_rho_dependence_effect=dep_raw.negative_dependence_effect,
        negative_rho_interpretation=dep_raw.negative_interpretation,
        methodology_note=dep_raw.methodology_note,
        method_status=dep_raw.method_status,
    )

    # ---- Step 19: Bayesian state-space model ----
    ss_rng = np.random.default_rng(sim_cfg.random_seed ^ 0xB5A550FF)
    ss_params = StateSpaceParams(n_particles=1000)
    monthly_burn_ss = calculate_monthly_burn(financial.quarterly_operating_cash_burn)
    total_liq_ss = compute_total_liquidity(financial)
    runway_ss = total_liq_ss / monthly_burn_ss if monthly_burn_ss > 0 else 12.0
    ss_raw = run_state_space_analysis(
        cash_months_runway=runway_ss,
        burn_acceleration=burn_result.burn_acceleration,
        enrollment_fraction=milestone_result.enrollment_fraction,
        biotech_market_score=float(financial.biotech_market_condition_score),
        rng=ss_rng,
        params=ss_params,
    )
    state_space_result = StateSpaceResult(
        cash_health_score=ss_raw.cash_health_score,
        burn_acceleration_signal=ss_raw.burn_acceleration_signal,
        clinical_progress_signal=ss_raw.clinical_progress_signal,
        market_condition_signal=ss_raw.market_condition_signal,
        anomaly_score=ss_raw.anomaly_score,
        current_state_posterior_mean=list(float(x) for x in ss_raw.current_state_estimate.posterior_mean),
        current_state_posterior_std=list(float(x) for x in ss_raw.current_state_estimate.posterior_std),
        predicted_state_posterior_mean=list(float(x) for x in ss_raw.predicted_state_estimate.posterior_mean),
        effective_sample_size=ss_raw.current_state_estimate.effective_sample_size,
        interpretation=ss_raw.interpretation,
        methodology_note=ss_raw.methodology_note,
        method_status=ss_raw.method_status,
    )

    # ---- Step 20: Report ----
    audit_response = AuditResponse(
        company_name=financial.company_name,
        ticker=financial.ticker,
        asset_name=clinical.asset_name,
        audit_timestamp=datetime.now(timezone.utc).isoformat(),
        model_version=_build_model_version(config, sim_cfg),
        provenance=_build_manual_provenance(request),
        validation_snapshot=_build_validation_snapshot(),
        data_quality=_compute_data_quality(request),
        cash_path=cash_path_result,
        solvency=solvency_result,
        success_probability=pos_result,
        milestone_timing=milestone_result,
        capital_to_catalyst=ctc_result,
        valuation=val_result,
        burn_regime=burn_result,
        disclosure_consistency=disclosure_result,
        final_summary=final_summary,
        multi_state=multi_state_result,
        value_of_information=voi_result,
        real_options=real_options_result,
        risk_attribution=risk_attribution_result,
        robustness=robustness_result,
        bma=bma_result,
        dependence=dependence_result,
        state_space=state_space_result,
        warnings=[
            "This is NOT investment advice.",
            "All model outputs are probabilistic ESTIMATES, not predictions.",
            "Cox coefficients are UNTRAINED MVP ASSUMPTIONS (see config.py).",
            "Public filings may lag real-world financing and clinical developments.",
            "ClinicalTrials.gov dates may differ from company-internal expectations.",
            "rNPV outputs are highly sensitive to user-supplied asset_value_success.",
        ],
        assumptions=[
            f"Weibull baseline: lambda={sim_cfg.baseline_lambda}, k={sim_cfg.baseline_k}.",
            f"Simulations: n={n}, seed={sim_cfg.random_seed}.",
            f"Gamma milestone delay factor: {milestone_result.delay_factor:.2f}x stated timeline.",
            f"Phase prior: Beta({pos_result.alpha_prior}, {pos_result.beta_prior}).",
            "Signal weights are configurable in config.py.",
        ],
        markdown_report="",  # populated below
    )

    report_md = generate_full_report(audit_response, request)
    audit_response = audit_response.model_copy(update={"markdown_report": report_md})

    return audit_response
