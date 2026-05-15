"""
Pydantic schemas for CatalystLens API — inputs, results, and audit response.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------

TrialPhase = Literal["preclinical", "phase_1", "phase_2", "phase_3", "filed", "approved"]
TrialStatus = Literal[
    "not_yet_recruiting", "recruiting", "active_not_recruiting",
    "completed", "suspended", "withdrawn", "terminated",
    "enrolling_by_invitation", "available", "no_longer_available",
    "temporarily_not_available", "approved_for_marketing", "withheld",
    "unknown",
]
CatalystType = Literal[
    "phase_completion", "interim_analysis", "primary_readout",
    "regulatory_submission", "approval_decision", "proof_of_concept",
]
FinancingEventKind = Literal["clean_refi", "distressed_refi", "partnership"]

FinancingState = Literal[
    "funded",
    "clean_refinancing",
    "distressed_refinancing",
    "partnership",
    "debt_or_royalty",
    "program_discontinuation",
    "cash_exhaustion",
]
CashPathState = Literal["continue", "cash_exhaustion", "horizon_reached"]
SourceType = Literal["sec_filing", "clinicaltrials", "deck", "press_release", "manual_input"]


# ---------------------------------------------------------------------------
# INPUT SCHEMAS
# ---------------------------------------------------------------------------

class QuarterlyBurnEntry(BaseModel):
    quarter: str = Field(..., examples=["2023-Q1"])
    operating_cash_burn: float = Field(..., gt=0, description="USD, positive value")


class CompanyFinancialInput(BaseModel):
    company_name: str
    ticker: str
    cash_on_hand: float = Field(..., ge=0, description="USD cash and equivalents")
    marketable_securities: float = Field(0.0, ge=0)
    quarterly_operating_cash_burn: float = Field(..., gt=0, description="Most recent quarterly burn (USD)")
    quarterly_burn_history: List[QuarterlyBurnEntry] = Field(
        default_factory=list,
        description="Chronological list of quarterly burn entries for regime detection",
    )
    market_cap: float = Field(..., gt=0)
    debt: float = Field(0.0, ge=0)
    going_concern_flag: bool = Field(False)
    recent_financing_flag: bool = Field(False, description="Capital raise within past 6 months")
    months_since_last_raise: float = Field(12.0, ge=0)
    biotech_market_condition_score: float = Field(
        5.0, ge=1.0, le=10.0,
        description="1=very poor, 5=neutral, 10=excellent financing market",
    )
    pipeline_concentration_score: float = Field(
        0.5, ge=0.0, le=1.0,
        description="0=diversified pipeline, 1=single-asset company",
    )
    planned_financing_events: List["FinancingEventInput"] = Field(
        default_factory=list,
        description="Planned or assumed financing inflows for mechanical cash-path simulation",
    )

    @field_validator("cash_on_hand", "marketable_securities", "quarterly_operating_cash_burn")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Financial values cannot be negative")
        return v


class ClinicalCatalystInput(BaseModel):
    asset_name: str
    indication: str
    trial_phase: TrialPhase
    trial_status: TrialStatus
    stated_months_to_catalyst: float = Field(..., gt=0)
    enrollment_target: int = Field(..., gt=0)
    enrollment_completed: int = Field(..., ge=0)
    enrollment_rate_per_month: float = Field(..., gt=0)
    number_of_sites: int = Field(1, ge=1)
    indication_complexity_score: float = Field(
        0.5, ge=0.0, le=1.0, description="0=simple, 1=complex/rare disease"
    )
    endpoint_complexity_score: float = Field(
        0.5, ge=0.0, le=1.0, description="0=simple biomarker, 1=complex survival endpoint"
    )
    regulatory_complexity_score: float = Field(
        0.5, ge=0.0, le=1.0, description="0=standard pathway, 1=novel/complex regulatory path"
    )
    catalyst_type: CatalystType = "primary_readout"
    followup_months_after_enrollment: float = Field(
        1.0, ge=0.0,
        description="Endpoint follow-up duration after enrollment completion before primary completion",
    )
    data_cleaning_months: float = Field(
        1.0, ge=0.0,
        description="Expected database lock / data cleaning duration after primary completion",
    )
    analysis_months: float = Field(
        1.0, ge=0.0,
        description="Expected statistical analysis duration after data cleaning",
    )
    disclosure_lag_months: float = Field(
        1.0, ge=0.0,
        description="Expected lag from analysis completion to public readout disclosure",
    )

    @field_validator("enrollment_completed")
    @classmethod
    def enrollment_cannot_exceed_target(cls, v: int, info) -> int:
        target = info.data.get("enrollment_target")
        if target is not None and v > target:
            raise ValueError("enrollment_completed cannot exceed enrollment_target")
        return v


class SuccessProbabilityInput(BaseModel):
    trial_phase: TrialPhase
    disease_area: Optional[str] = Field(
        None,
        description="Disease area stratum for future hierarchical PoS priors",
    )
    modality: Optional[str] = Field(
        None,
        description="Therapeutic modality stratum for future hierarchical PoS priors",
    )
    endpoint_family: Optional[str] = Field(
        None,
        description="Endpoint family stratum for future hierarchical PoS priors",
    )
    positive_signals: List[str] = Field(
        default_factory=list,
        description="List of positive signal keys present for this trial",
    )
    negative_signals: List[str] = Field(
        default_factory=list,
        description="List of negative signal keys present for this trial",
    )
    custom_alpha_prior: Optional[float] = Field(None, gt=0)
    custom_beta_prior: Optional[float] = Field(None, gt=0)


class ValuationInput(BaseModel):
    asset_value_success: float = Field(..., gt=0, description="USD asset value if approved/successful")
    downside_value: float = Field(0.0, ge=0, description="USD residual value on failure")
    annual_discount_rate: float = Field(0.12, gt=0, lt=1.0, description="WACC / risk-adjusted discount rate")
    expected_dilution_if_refinanced: float = Field(
        0.25, ge=0.0, le=0.95,
        description="Fractional dilution (e.g. 0.25 = 25% shareholder dilution) if company must refinance",
    )
    financing_penalty_strength: float = Field(
        0.6, ge=0.0, le=1.0,
        description="How strongly near-term financing need penalises value (0=no penalty, 1=full penalty)",
    )


class DisclosureInput(BaseModel):
    company_narrative_distribution: Dict[str, float] = Field(
        ...,
        description="Score 0–1 per category reflecting management narrative framing",
        examples=[{
            "runway_strength": 0.8,
            "clinical_timeline_confidence": 0.9,
            "dilution_risk": 0.1,
            "trial_maturity": 0.7,
            "endpoint_strength": 0.8,
            "pipeline_diversification": 0.3,
        }],
    )
    structured_audit_distribution: Dict[str, float] = Field(
        ...,
        description="Score 0–1 per category derived from quantitative model outputs",
        examples=[{
            "runway_strength": 0.4,
            "clinical_timeline_confidence": 0.5,
            "dilution_risk": 0.7,
            "trial_maturity": 0.4,
            "endpoint_strength": 0.5,
            "pipeline_diversification": 0.3,
        }],
    )

    @field_validator("company_narrative_distribution", "structured_audit_distribution")
    @classmethod
    def scores_must_be_unit_interval(cls, dist: Dict[str, float]) -> Dict[str, float]:
        for key, value in dist.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be between 0 and 1")
        return dist


class SimulationConfig(BaseModel):
    n_simulations: int = Field(10_000, ge=100, le=100_000)
    random_seed: int = 42
    monthly_horizon: int = Field(48, ge=12, le=120)
    baseline_lambda: float = Field(0.035, gt=0)
    baseline_k: float = Field(1.30, gt=0)
    use_multistate: bool = Field(
        False,
        description=(
            "Enable multi-state competing-risk engine (8 absorbing states). "
            "When False, uses the legacy binary Cox-Weibull sampler."
        ),
    )


class AuditRequest(BaseModel):
    financial: CompanyFinancialInput
    clinical: ClinicalCatalystInput
    success_probability: SuccessProbabilityInput
    valuation: ValuationInput
    disclosure: DisclosureInput
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    allow_phase_override: bool = Field(
        False,
        description="Set true only when intentionally using a PoS prior stage different from the clinical catalyst phase.",
    )

    @model_validator(mode="after")
    def phase_inputs_must_match(self) -> "AuditRequest":
        if (
            not self.allow_phase_override
            and self.clinical.trial_phase != self.success_probability.trial_phase
        ):
            raise ValueError(
                "success_probability.trial_phase must match clinical.trial_phase "
                "unless allow_phase_override is true"
            )
        return self


class EvidenceRef(BaseModel):
    source_type: SourceType = "manual_input"
    source_id: str = "manual"
    as_of_date: Optional[str] = None
    locator: Optional[str] = None
    sha256: Optional[str] = None


class ProvenanceItem(BaseModel):
    field: str
    value: Optional[float | str | int | bool] = None
    evidence: EvidenceRef = Field(default_factory=EvidenceRef)


class ProvenanceBundle(BaseModel):
    financial_inputs: List[ProvenanceItem] = Field(default_factory=list)
    clinical_inputs: List[ProvenanceItem] = Field(default_factory=list)
    claims: List[ProvenanceItem] = Field(default_factory=list)
    provenance_status: Literal["manual_inputs_unverified", "source_traced"] = "manual_inputs_unverified"


class FinancingEventInput(BaseModel):
    month: int = Field(..., ge=0)
    kind: FinancingEventKind
    gross_proceeds: float = Field(..., ge=0)
    transaction_cost_rate: float = Field(0.0, ge=0.0, le=0.95)

    @property
    def net_proceeds(self) -> float:
        return self.gross_proceeds * (1.0 - self.transaction_cost_rate)


class CashPathInput(BaseModel):
    starting_cash: float = Field(..., ge=0)
    monthly_burn: float = Field(..., gt=0)
    horizon_months: int = Field(48, ge=1, le=240)
    monthly_burn_volatility: float = Field(
        0.0, ge=0.0, le=2.0,
        description="Lognormal monthly burn volatility; 0 keeps burn deterministic.",
    )
    financing_events: List[FinancingEventInput] = Field(default_factory=list)
    catalyst_month: Optional[float] = Field(
        None, ge=0,
        description="Optional catalyst/public-readout month for capital-needed-to-catalyst calculation",
    )


class CashPathMonth(BaseModel):
    month: int
    starting_cash: float
    sampled_burn: float
    capital_inflow: float
    ending_cash: float
    state: CashPathState


class CashPathResult(BaseModel):
    cash_exhaustion_month: Optional[int]
    final_state: CashPathState
    minimum_cash_balance: float
    ending_cash: float
    total_burn: float
    total_capital_raised: float
    cash_shortfall_at_exhaustion: float = 0.0
    maximum_cash_deficit: float = 0.0
    capital_needed_to_survive_horizon: float = 0.0
    capital_needed_to_reach_catalyst: Optional[float] = None
    monthly_balances: List[CashPathMonth]


# ---------------------------------------------------------------------------
# RESULT SCHEMAS
# ---------------------------------------------------------------------------

class SurvivalPoint(BaseModel):
    month: int
    survival_probability: float
    implied_cashout_risk: float


class SolvencyResult(BaseModel):
    monthly_burn: float
    total_liquidity: float
    simple_runway_months: float
    risk_multiplier: float
    linear_predictor: float
    covariate_contributions: Dict[str, float]
    survival_curve: List[SurvivalPoint]
    median_failure_time: float
    p_survival_6m: float
    p_survival_12m: float
    p_survival_18m: float
    p_survival_24m: float
    model_assumptions: List[str]


class SuccessProbabilityResult(BaseModel):
    alpha_prior: float
    beta_prior: float
    prior_mean: float
    alpha_posterior: float
    beta_posterior: float
    posterior_mean: float
    credible_interval_lower: float
    credible_interval_upper: float
    credible_interval_pct: float
    applied_positive_weights: Dict[str, float]
    applied_negative_weights: Dict[str, float]
    prior_source: str = "mvp_phase_prior"
    prior_confidence: float = 0.35
    prior_fallback_level: str = "phase_only"
    model_assumptions: List[str]


class MilestoneTimingResult(BaseModel):
    gamma_alpha: float
    gamma_beta_rate: float
    stated_months: float
    adjusted_mean_months: float
    delay_factor: float
    cv: float
    enrollment_fraction: float
    enrollment_remaining_months: float
    enrollment_component_months: float = 0.0
    followup_component_months: float = 0.0
    data_cleaning_component_months: float = 0.0
    analysis_component_months: float = 0.0
    disclosure_lag_months: float = 0.0
    primary_completion_months: float = 0.0
    public_readout_lag_months: float = 0.0
    public_readout_months: float = 0.0
    p5_months: float
    p25_months: float
    p50_months: float
    p75_months: float
    p95_months: float
    model_assumptions: List[str]


class CapitalToCatalystResult(BaseModel):
    probability_reaches_catalyst: float
    probability_cashout_before_catalyst: float
    median_gap_months: float
    p5_gap_months: float
    p95_gap_months: float
    median_financial_failure_time: float
    median_catalyst_time: float
    risk_classification: str
    interpretation: str


class ValuationResult(BaseModel):
    mean_value: float
    median_value: float
    p5_value: float
    p95_value: float
    technical_risk_only_rnpv: float
    financing_adjusted_rnpv: float
    financing_risk_discount: float
    probability_downside: float
    probability_high_upside: float
    high_upside_threshold: float
    # Four-state financing model probabilities
    p_funded_through_catalyst: float = 0.0
    p_refinancing_success: float = 0.0
    p_distressed_financing: float = 0.0
    p_program_discontinuation: float = 0.0
    mean_value_if_funded: float = 0.0
    mean_value_if_refinanced: float = 0.0
    mean_value_if_distressed: float = 0.0
    model_assumptions: List[str]


class BurnRegimeResult(BaseModel):
    burn_series: List[float]
    quarters: List[str]
    quarterly_pct_changes: List[Optional[float]]
    changepoint_indices: List[int]
    burn_acceleration: float
    regime: str
    regime_interpretation: str
    model_assumptions: List[str]


class DisclosureConsistencyResult(BaseModel):
    jsd_score: float
    kl_narrative_vs_audit: float
    kl_audit_vs_narrative: float
    mean_absolute_gap: float
    optimism_bias: float
    max_category_gap: float
    combined_gap_score: float
    gap_classification: str
    category_gaps: Dict[str, float]
    narrative_normalized: Dict[str, float]
    audit_normalized: Dict[str, float]
    interpretation: str


class ScenarioResult(BaseModel):
    scenario_name: str
    description: str
    catalyst_timing_months: float
    burn_assumption: str
    pos_assumption: float
    financing_need: str
    expected_value: float
    probability_cashout_before_catalyst: float
    interpretation: str


class SensitivityPoint(BaseModel):
    variable: str
    low_label: str
    base_label: str
    high_label: str
    low_cashout_prob: float
    base_cashout_prob: float
    high_cashout_prob: float
    low_expected_value: float
    base_expected_value: float
    high_expected_value: float


class FinalSummaryResult(BaseModel):
    risk_classification: str
    probability_cashout_before_catalyst: float
    probability_reaches_catalyst: float
    posterior_pos: float
    expected_value: float
    financing_adjusted_rnpv: float
    primary_risk_factor: str
    secondary_risk_factor: str
    key_finding: str
    scenarios: List[ScenarioResult]
    sensitivity: List[SensitivityPoint]
    diligence_questions: List[str]


class RobustnessResult(BaseModel):
    """Wasserstein-ball DRO bounds on cashout probability and EV."""
    nominal_cashout_prob: float
    nominal_ev: float
    worst_case_cashout_prob_e05: float
    worst_case_cashout_prob_e10: float
    worst_case_cashout_prob_e20: float
    worst_case_ev_e05: float
    worst_case_ev_e10: float
    worst_case_ev_e20: float
    best_case_cashout_prob_e10: float
    best_case_ev_e10: float
    robustness_interpretation: str
    methodology_note: str


class ModelWeightSchema(BaseModel):
    k: float
    lambda_: float
    posterior_weight: float
    model_cashout_prob: float
    model_ev: float


class BMAResult(BaseModel):
    """Bayesian model averaging over Weibull model candidates."""
    bma_cashout_prob: float
    bma_ev: float
    model_weights: List[ModelWeightSchema]
    effective_n_models: float
    highest_weight_model_k: float
    highest_weight_model_lambda: float
    methodology_note: str


class DependenceAnalysisResult(BaseModel):
    """Gaussian copula analysis of T_fin / T_sci rank correlation."""
    base_cashout_prob: float
    positive_rho_cashout_prob: float
    positive_rho_dependence_effect: float
    positive_rho_interpretation: str
    negative_rho_cashout_prob: float
    negative_rho_dependence_effect: float
    negative_rho_interpretation: str
    methodology_note: str


class RealOptionsResult(BaseModel):
    """Real-options valuation output (compound option on clinical success)."""
    rov_mean: float
    rov_median: float
    rov_p5: float
    rov_p95: float
    rnpv_static: float
    real_options_premium: float
    real_options_premium_pct: float
    abandonment_value: float
    model_assumptions: List[str]


class ShapleyComponentSchema(BaseModel):
    driver: str
    description: str
    cashout_prob_shapley: float
    ev_shapley: float
    rank: int


class RiskAttributionResult(BaseModel):
    """Shapley-based decomposition of cashout probability and EV uncertainty."""
    components: List[ShapleyComponentSchema]
    total_cashout_prob: float
    total_ev: float
    explained_cashout_prob: float
    explained_ev: float
    methodology_note: str


class MultiStateResult(BaseModel):
    """Output of the multi-state competing-risk engine (8 absorbing states)."""
    absorbing_state_probs: Dict[str, float] = Field(
        description="Fraction of simulation paths absorbed into each state by horizon end",
    )
    overall_survival_at_horizon: float = Field(
        description="S(horizon): fraction of paths still in operating state at end of horizon",
    )
    median_transition_time: Optional[float] = Field(
        None, description="Median time to absorption across all causes (months)",
    )
    cif_at_catalyst_month: Dict[str, float] = Field(
        default_factory=dict,
        description="CIF_j(catalyst_month) for each cause; empty when catalyst_month is None",
    )
    overall_survival_at_catalyst_month: Optional[float] = Field(
        None, description="S(catalyst_month): probability of being in operating state at catalyst",
    )
    model_assumptions: List[str] = Field(default_factory=list)


class SignalEVSISchema(BaseModel):
    signal_name: str
    description: str
    category: str
    signal_weight: float
    evsi_dollars: float
    ev_if_positive: float
    ev_if_negative: float
    p_positive: float


class ValueOfInformationResult(BaseModel):
    """EVPI and per-signal EVSI for diligence prioritization."""
    evpi_dollars: float
    evpi_pct_of_ev: float
    evpi_interpretation: str
    per_signal_evsi: List[SignalEVSISchema]
    top_diligence_priority: str
    total_observable_evsi: float
    methodology_note: str


class DataQualityResult(BaseModel):
    financial_data_completeness: float = Field(ge=0.0, le=1.0)
    clinical_data_completeness: float = Field(ge=0.0, le=1.0)
    disclosure_data_completeness: float = Field(ge=0.0, le=1.0)
    overall_completeness: float = Field(ge=0.0, le=1.0)
    primary_limitations: List[str]
    data_quality_score: Literal["high", "moderate", "low"]
    evidence_quality_score: Literal["high", "moderate", "low"] = "low"
    evidence_quality_note: str = "All inputs are manual; no SEC/ClinicalTrials.gov source tracing applied."


class ValidationSnapshot(BaseModel):
    solvency_calibration_status: Literal["research_mode", "validated"] = "research_mode"
    solvency_c_index_ipcw: Optional[float] = None
    solvency_integrated_brier_score: Optional[float] = None
    solvency_ici_12m: Optional[float] = None
    pos_ppc_status: Literal["not_available", "pass", "fail"] = "not_available"
    timing_interval_coverage_status: Literal["not_available", "pass", "fail"] = "not_available"
    notes: List[str] = Field(default_factory=list)


class ModelVersionInfo(BaseModel):
    backend_version: str = "0.1.0"
    name: str = "catalystlens-backend"
    semver: str = "0.1.0"
    artifact_id: str = "mvp_assumption_engine"
    coefficient_set: str = "mvp_untrained_v1"
    n_simulations: int
    random_seed: int
    config_hash: str
    training_cutoff_date: Optional[str] = None
    data_snapshot_ids: List[str] = Field(default_factory=list)
    calibration_status: str = (
        "UNCALIBRATED — coefficients are configurable MVP assumptions, "
        "not fit to historical biotech financing outcome data"
    )


class AuditResponse(BaseModel):
    company_name: str
    ticker: str
    asset_name: str
    audit_timestamp: str
    model_version: ModelVersionInfo
    provenance: ProvenanceBundle = Field(default_factory=ProvenanceBundle)
    validation_snapshot: ValidationSnapshot = Field(default_factory=ValidationSnapshot)
    data_quality: DataQualityResult
    cash_path: CashPathResult
    solvency: SolvencyResult
    success_probability: SuccessProbabilityResult
    milestone_timing: MilestoneTimingResult
    capital_to_catalyst: CapitalToCatalystResult
    valuation: ValuationResult
    burn_regime: BurnRegimeResult
    disclosure_consistency: DisclosureConsistencyResult
    final_summary: FinalSummaryResult
    multi_state: Optional[MultiStateResult] = None
    value_of_information: Optional[ValueOfInformationResult] = None
    real_options: Optional[RealOptionsResult] = None
    risk_attribution: Optional[RiskAttributionResult] = None
    robustness: Optional[RobustnessResult] = None
    bma: Optional[BMAResult] = None
    dependence: Optional[DependenceAnalysisResult] = None
    warnings: List[str]
    assumptions: List[str]
    markdown_report: str
