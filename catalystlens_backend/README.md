# CatalystLens Backend

Probabilistic biotech capital-to-catalyst audit engine.

**Core question:** *Will this company remain funded long enough to reach the scientific milestone its valuation depends on?*

> **Disclaimer:** All model outputs are probabilistic estimates, not predictions or investment recommendations. Cox coefficients, signal weights, and phase priors are configurable MVP assumptions that have not been trained on historical biotech outcome data. This system does not constitute investment advice.

---

## Architecture

```
catalystlens_backend/
├── app/
│   ├── main.py                      # FastAPI app entry point
│   ├── api/
│   │   └── routes.py                # All API endpoints
│   ├── core/
│   │   └── config.py                # All coefficients and thresholds
│   ├── data_sources/
│   │   ├── sec_client.py            # SEC EDGAR submissions/companyfacts client
│   │   ├── clinicaltrials_client.py # ClinicalTrials.gov API v2 client
│   │   ├── fred_client.py           # FRED market-covariate client
│   │   └── cache.py                 # Raw JSON payload cache with SHA-256 hashes
│   ├── labeling/
│   │   └── financing_events.py      # First-pass financing/discontinuation labels
│   ├── models/
│   │   └── schemas.py               # Pydantic input/output schemas
│   ├── registry/
│   │   └── model_registry.py        # Frozen model artifact metadata cards
│   └── engines/
│       ├── solvency.py              # Financial clock (Cox-Weibull survival)
│       ├── cash_path.py             # Mechanical monthly cash-balance path
│       ├── milestone_timing.py      # Scientific clock (Gamma distribution)
│       ├── bayesian_success.py      # Bayesian PoS (Beta posterior)
│       ├── capital_to_catalyst.py   # Gap probability P(T_sci < T_fin)
│       ├── valuation.py             # rNPV / Monte Carlo valuation
│       ├── burn_regime.py           # PELT change-point detection
│       ├── disclosure_consistency.py# Jensen-Shannon divergence
│       ├── monte_carlo.py           # Central simulation engine
│       └── report_generator.py      # Institutional Markdown report
├── data/
│   └── example_company.json         # Example: NovaCure Therapeutics (NCTX)
├── tests/
│   ├── test_solvency.py
│   ├── test_bayesian_success.py
│   ├── test_monte_carlo.py
│   └── test_api.py
├── training/
│   └── datasets/
│       └── build_company_panel.py   # Point-in-time company-quarter panel builder
├── conftest.py
└── requirements.txt
```

---

## Installation

```bash
cd catalystlens_backend
pip install -r requirements.txt
```

Requires Python 3.11+.

---

## Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: 127 tests, all passing.

---

## Calling `/audit` with the Example Company

```bash
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d @data/example_company.json
```

Or in Python:

```python
import json, httpx

with open("data/example_company.json") as f:
    payload = json.load(f)

r = httpx.post("http://localhost:8000/audit", json=payload)
result = r.json()

print(result["capital_to_catalyst"]["probability_cashout_before_catalyst"])
print(result["final_summary"]["risk_classification"])
print(result["markdown_report"][:2000])
```

The example company (NovaCure Therapeutics / NCTX) is designed to produce **moderate-to-high capital-to-catalyst risk** based on:
- ~11 months of simple runway vs 18-month stated catalyst
- Sharply accelerating burn (+66% QoQ over 4 quarters)
- 35% enrollment completion at current pace
- Single-asset pipeline concentration

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/audit` | Full CatalystLens audit |
| POST | `/simulate` | Monte Carlo simulation (alias for `/audit`) |
| POST | `/solvency` | Financial survival model only |
| POST | `/success-probability` | Bayesian PoS only |
| POST | `/milestone-timing` | Gamma milestone timing only |
| POST | `/burn-regime` | PELT burn change-point detection only |
| POST | `/disclosure-consistency` | Jensen-Shannon divergence analysis only |

---

## Where the Math Lives

| File | Mathematical Content |
|---|---|
| `engines/solvency.py` | Cox PH-style Weibull survival model, risk multiplier computation, inverse-CDF sampling |
| `engines/cash_path.py` | Mechanical cash balance path: cash[t+1] = cash[t] - burn[t] + financing inflows |
| `engines/milestone_timing.py` | Gamma(α, β) parameterization from stated timeline + complexity, delay factor calculation |
| `engines/bayesian_success.py` | Beta(α, β) prior/posterior, signal weight updates, credible intervals |
| `engines/capital_to_catalyst.py` | P(T_sci < T_fin) from Monte Carlo samples, gap statistics |
| `engines/valuation.py` | Discount factor, financing adjustment penalty, rNPV decomposition |
| `engines/burn_regime.py` | PELT change-point detection (ruptures), QoQ acceleration |
| `engines/disclosure_consistency.py` | Jensen-Shannon divergence, KL divergence, normalized distributions |
| `engines/monte_carlo.py` | Vectorized simulation loop, scenario engine, sensitivity engine |
| `core/config.py` | All configurable coefficients (Cox betas, Weibull params, phase priors, signal weights) |

---

## Institutional Data and Training Foundation

The backend now includes the foundation required to move from assumption-based MVP
outputs to trained artifacts:

| Layer | Files | Purpose |
|---|---|---|
| Official data clients | `app/data_sources/sec_client.py`, `clinicaltrials_client.py`, `fred_client.py` | Fetch SEC EDGAR, ClinicalTrials.gov v2, and FRED JSON payloads through explicit URL builders |
| Raw cache | `app/data_sources/cache.py` | Persist raw source payloads with SHA-256 hashes for replay and provenance |
| Provenance | `app/provenance.py`, `app/models/schemas.py` | Attach source type, source id, locator, date, and payload hash to inputs and claims |
| Point-in-time panel | `training/datasets/build_company_panel.py` | Convert SEC/ClinicalTrials/FRED payloads into company-quarter feature rows |
| Outcome labels | `app/labeling/financing_events.py` | First-pass labels for refinancing, distressed financing, partnership, and discontinuation events |
| Model registry | `app/registry/model_registry.py` | Store frozen artifact cards with training cutoff, snapshot IDs, metrics, and config hash |

Current `/audit` responses expose `provenance`, `validation_snapshot`, and
`model_version` fields. Until historical labels are assembled and trained
artifacts are registered, the service reports `research_mode` and uses the
manual assumption engine rather than claiming calibrated institutional accuracy.

---

## What Should Be Replaced with Real Models / APIs

| Component | Current State | What to Replace With |
|---|---|---|
| Cox coefficients | Untrained MVP assumptions | Coefficients fit to historical biotech financing failure data |
| Phase priors | Industry intuition-based Beta parameters | Calibrated to historical trial-to-approval outcome databases |
| Signal weights | Directional expert priors | Trained on clinical success/failure prediction literature |
| Weibull baseline | General biotech intuition | Fit to historical clinical-stage biotech financing outcomes |
| Clinical timing | Gamma parameterized from stated timeline | ClinicalTrials.gov API integration + historical delay modeling |
| Burn history | User-supplied JSON | SEC EDGAR API (XBRL cash flow data extraction) |
| Disclosure scores | User-supplied narrative vs audit | NLP analysis of 10-K/10-Q risk factors vs quantitative model |
| Asset value | User-supplied assumption | Comparable transaction / peak sales modeling integration |

---

## Model Assumptions (Summary)

1. **Solvency:** Weibull baseline S₀(t) = exp(-(λt)^k) with λ=0.035, k=1.3. Cox risk multiplier = exp(LP) where LP is a linear combination of 10 covariates.
2. **Milestone Timing:** T_sci ~ Gamma(α, β) where α = 1/CV², mean = stated_months × delay_factor. Minimum time floored at enrollment_remaining × buffer.
3. **Bayesian PoS:** Beta(α, β) updated with additive signal weights. Prior is phase-specific.
4. **Capital-to-Catalyst:** P(cashout) = P(T_fin < T_sci) estimated from N=10,000 Monte Carlo samples.
5. **Valuation:** rNPV = E[asset_value × discount × financing_adjustment × success_indicator].
6. **All probabilities are modelled estimates.** They are not ground truth and should be treated as probabilistic scenario analysis inputs, not output facts.
