"""
CatalystLens FastAPI route definitions.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.engines.bayesian_success import run_success_probability_analysis
from app.engines.burn_regime import run_burn_regime_analysis
from app.engines.disclosure_consistency import run_disclosure_consistency_analysis
from app.engines.milestone_timing import run_milestone_timing_analysis
from app.engines.monte_carlo import run_full_audit
from app.engines.solvency import run_solvency_analysis
from app.models.schemas import (
    AuditRequest,
    AuditResponse,
    BurnRegimeResult,
    ClinicalCatalystInput,
    CompanyFinancialInput,
    DisclosureConsistencyResult,
    DisclosureInput,
    MilestoneTimingResult,
    SimulationConfig,
    SolvencyResult,
    SuccessProbabilityInput,
    SuccessProbabilityResult,
)

router = APIRouter()


@router.get("/", tags=["status"])
def root():
    """Health check and API status."""
    return {
        "service": "CatalystLens",
        "version": "0.1.0",
        "status": "operational",
        "description": (
            "Probabilistic biotech capital-to-catalyst audit engine. "
            "All outputs are model estimates, not investment advice."
        ),
    }


@router.post("/audit", response_model=AuditResponse, tags=["audit"])
def full_audit(request: AuditRequest) -> AuditResponse:
    """
    Run a complete CatalystLens audit.

    Accepts a full company audit payload and returns:
    - Solvency / financial clock analysis
    - Bayesian probability of technical success
    - Milestone timing distribution
    - Capital-to-catalyst gap probability
    - Valuation / rNPV distribution
    - Burn regime detection
    - Disclosure consistency analysis
    - Institutional Markdown report
    """
    try:
        return run_full_audit(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit engine error: {exc}") from exc


@router.post("/solvency", response_model=SolvencyResult, tags=["engines"])
def solvency_only(financial: CompanyFinancialInput) -> SolvencyResult:
    """Run the financial solvency (survival) model only."""
    try:
        return run_solvency_analysis(financial)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/success-probability", response_model=SuccessProbabilityResult, tags=["engines"])
def success_probability_only(inputs: SuccessProbabilityInput) -> SuccessProbabilityResult:
    """Run the Bayesian probability of technical success model only."""
    try:
        return run_success_probability_analysis(inputs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/milestone-timing", response_model=MilestoneTimingResult, tags=["engines"])
def milestone_timing_only(clinical: ClinicalCatalystInput) -> MilestoneTimingResult:
    """Run the Gamma milestone timing model only."""
    try:
        return run_milestone_timing_analysis(clinical)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/burn-regime", response_model=BurnRegimeResult, tags=["engines"])
def burn_regime_only(financial: CompanyFinancialInput) -> BurnRegimeResult:
    """Run burn-rate change point detection only."""
    try:
        return run_burn_regime_analysis(financial)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/disclosure-consistency", response_model=DisclosureConsistencyResult, tags=["engines"])
def disclosure_consistency_only(inputs: DisclosureInput) -> DisclosureConsistencyResult:
    """Run disclosure consistency (Jensen-Shannon divergence) analysis only."""
    try:
        return run_disclosure_consistency_analysis(inputs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate", response_model=AuditResponse, tags=["audit"])
def simulate(request: AuditRequest) -> AuditResponse:
    """
    Run full Monte Carlo simulation.

    Equivalent to /audit — provided as a semantically explicit endpoint
    for callers that want to emphasise simulation rather than audit framing.
    """
    try:
        return run_full_audit(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation engine error: {exc}") from exc
