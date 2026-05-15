"""
Tests for FastAPI endpoints.
"""

import json
import os
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _load_example_payload() -> dict:
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "example_company.json"
    )
    with open(data_path) as f:
        payload = json.load(f)
    # Use fewer simulations for test speed
    payload["simulation"]["n_simulations"] = 1000
    return payload


def _minimal_financial_payload() -> dict:
    return {
        "company_name": "TestCo",
        "ticker": "TST",
        "cash_on_hand": 60000000,
        "marketable_securities": 10000000,
        "quarterly_operating_cash_burn": 15000000,
        "market_cap": 200000000,
        "debt": 5000000,
        "going_concern_flag": False,
        "recent_financing_flag": False,
        "months_since_last_raise": 12.0,
        "biotech_market_condition_score": 5.0,
        "pipeline_concentration_score": 0.5,
    }


def _minimal_clinical_payload() -> dict:
    return {
        "asset_name": "TST-001",
        "indication": "Test Indication",
        "trial_phase": "phase_2",
        "trial_status": "recruiting",
        "stated_months_to_catalyst": 18,
        "enrollment_target": 120,
        "enrollment_completed": 60,
        "enrollment_rate_per_month": 8,
        "number_of_sites": 10,
        "indication_complexity_score": 0.5,
        "endpoint_complexity_score": 0.5,
        "regulatory_complexity_score": 0.5,
        "catalyst_type": "primary_readout",
    }


class TestRootEndpoint:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_service_name(self):
        response = client.get("/")
        data = response.json()
        assert data["service"] == "CatalystLens"
        assert data["status"] == "operational"

    def test_response_has_version(self):
        response = client.get("/")
        assert "version" in response.json()


class TestAuditEndpoint:
    def test_audit_returns_200_with_example_payload(self):
        payload = _load_example_payload()
        response = client.post("/audit", json=payload)
        assert response.status_code == 200, f"Audit failed: {response.text[:500]}"

    def test_audit_response_has_expected_top_level_keys(self):
        payload = _load_example_payload()
        response = client.post("/audit", json=payload)
        data = response.json()
        expected_keys = [
            "company_name", "ticker", "asset_name",
            "model_version", "data_quality",
            "solvency", "success_probability", "milestone_timing",
            "capital_to_catalyst", "valuation", "burn_regime",
            "disclosure_consistency", "final_summary",
            "warnings", "assumptions", "markdown_report",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_audit_model_version_fields(self):
        payload = _load_example_payload()
        response = client.post("/audit", json=payload)
        mv = response.json()["model_version"]
        assert mv["backend_version"] == "0.1.0"
        assert mv["coefficient_set"] == "mvp_untrained_v1"
        assert isinstance(mv["n_simulations"], int)
        assert isinstance(mv["config_hash"], str) and len(mv["config_hash"]) > 0

    def test_audit_data_quality_fields(self):
        payload = _load_example_payload()
        response = client.post("/audit", json=payload)
        dq = response.json()["data_quality"]
        assert 0.0 <= dq["financial_data_completeness"] <= 1.0
        assert 0.0 <= dq["clinical_data_completeness"] <= 1.0
        assert 0.0 <= dq["disclosure_data_completeness"] <= 1.0
        assert 0.0 <= dq["overall_completeness"] <= 1.0
        assert dq["data_quality_score"] in ("high", "moderate", "low")
        assert isinstance(dq["primary_limitations"], list)

    def test_audit_company_name_matches_input(self):
        payload = _load_example_payload()
        response = client.post("/audit", json=payload)
        data = response.json()
        assert data["company_name"] == "NovaCure Therapeutics"

    def test_audit_probability_in_range(self):
        payload = _load_example_payload()
        response = client.post("/audit", json=payload)
        data = response.json()
        p = data["capital_to_catalyst"]["probability_cashout_before_catalyst"]
        assert 0.0 <= p <= 1.0

    def test_audit_markdown_report_non_empty(self):
        payload = _load_example_payload()
        response = client.post("/audit", json=payload)
        data = response.json()
        assert len(data["markdown_report"]) > 500

    def test_audit_invalid_payload_returns_422(self):
        response = client.post("/audit", json={"invalid": "payload"})
        assert response.status_code == 422


class TestSolvencyEndpoint:
    def test_solvency_returns_200(self):
        response = client.post("/solvency", json=_minimal_financial_payload())
        assert response.status_code == 200

    def test_solvency_returns_monthly_burn(self):
        response = client.post("/solvency", json=_minimal_financial_payload())
        data = response.json()
        assert "monthly_burn" in data
        assert data["monthly_burn"] == pytest.approx(5_000_000, rel=1e-3)

    def test_solvency_survival_curve_non_empty(self):
        response = client.post("/solvency", json=_minimal_financial_payload())
        data = response.json()
        assert len(data["survival_curve"]) > 0

    def test_solvency_probabilities_in_range(self):
        response = client.post("/solvency", json=_minimal_financial_payload())
        data = response.json()
        for key in ["p_survival_6m", "p_survival_12m", "p_survival_18m", "p_survival_24m"]:
            assert 0.0 <= data[key] <= 1.0


class TestSuccessProbabilityEndpoint:
    def test_returns_200(self):
        payload = {
            "trial_phase": "phase_2",
            "positive_signals": ["validated_biomarker"],
            "negative_signals": ["small_sample_size"],
        }
        response = client.post("/success-probability", json=payload)
        assert response.status_code == 200

    def test_returns_posterior_mean(self):
        payload = {
            "trial_phase": "phase_2",
            "positive_signals": [],
            "negative_signals": [],
        }
        response = client.post("/success-probability", json=payload)
        data = response.json()
        assert "posterior_mean" in data
        assert 0.0 <= data["posterior_mean"] <= 1.0


class TestMilestoneTimingEndpoint:
    def test_returns_200(self):
        response = client.post("/milestone-timing", json=_minimal_clinical_payload())
        assert response.status_code == 200

    def test_returns_gamma_parameters(self):
        response = client.post("/milestone-timing", json=_minimal_clinical_payload())
        data = response.json()
        assert "gamma_alpha" in data
        assert "p50_months" in data
        assert data["p50_months"] > 0


class TestBurnRegimeEndpoint:
    def test_returns_200(self):
        payload = {
            **_minimal_financial_payload(),
            "quarterly_burn_history": [
                {"quarter": "2023-Q1", "operating_cash_burn": 12000000},
                {"quarter": "2023-Q2", "operating_cash_burn": 14000000},
                {"quarter": "2023-Q3", "operating_cash_burn": 17000000},
                {"quarter": "2023-Q4", "operating_cash_burn": 22000000},
            ],
        }
        response = client.post("/burn-regime", json=payload)
        assert response.status_code == 200

    def test_returns_regime_classification(self):
        payload = {
            **_minimal_financial_payload(),
            "quarterly_burn_history": [
                {"quarter": "2023-Q1", "operating_cash_burn": 10000000},
                {"quarter": "2023-Q2", "operating_cash_burn": 10500000},
                {"quarter": "2023-Q3", "operating_cash_burn": 11000000},
            ],
        }
        response = client.post("/burn-regime", json=payload)
        data = response.json()
        assert "regime" in data
        assert data["regime"] in [
            "stable burn", "accelerating burn", "sharply accelerating burn",
            "decreasing burn", "insufficient data",
        ]


class TestDisclosureConsistencyEndpoint:
    def test_returns_200(self):
        payload = {
            "company_narrative_distribution": {
                "runway_strength": 0.8,
                "clinical_timeline_confidence": 0.9,
                "dilution_risk": 0.1,
            },
            "structured_audit_distribution": {
                "runway_strength": 0.4,
                "clinical_timeline_confidence": 0.5,
                "dilution_risk": 0.7,
            },
        }
        response = client.post("/disclosure-consistency", json=payload)
        assert response.status_code == 200

    def test_returns_jsd_in_range(self):
        payload = {
            "company_narrative_distribution": {
                "runway_strength": 0.8,
                "dilution_risk": 0.1,
            },
            "structured_audit_distribution": {
                "runway_strength": 0.3,
                "dilution_risk": 0.8,
            },
        }
        response = client.post("/disclosure-consistency", json=payload)
        data = response.json()
        assert 0.0 <= data["jsd_score"] <= 1.0

    def test_identical_distributions_produce_near_zero_jsd(self):
        dist = {
            "runway_strength": 0.5,
            "clinical_timeline_confidence": 0.6,
            "dilution_risk": 0.3,
        }
        payload = {
            "company_narrative_distribution": dist,
            "structured_audit_distribution": dist,
        }
        response = client.post("/disclosure-consistency", json=payload)
        data = response.json()
        assert data["jsd_score"] == pytest.approx(0.0, abs=1e-4)
