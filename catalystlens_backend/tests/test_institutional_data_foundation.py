"""
Tests for institutional-grade data, provenance, and model-governance foundations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data_sources.cache import RawPayloadCache
from app.data_sources.clinicaltrials_client import ClinicalTrialsClient
from app.data_sources.fred_client import FREDClient
from app.data_sources.sec_client import SECClient
from app.labeling.financing_events import classify_financing_event, label_financing_events
from app.provenance import evidence_from_cached_payload
from app.registry.model_registry import ModelArtifactCard, ModelRegistry
from training.datasets.build_company_panel import build_company_quarter_panel


class TestOfficialDataClients:
    def test_sec_client_builds_documented_urls_and_normalizes_cik(self):
        client = SECClient(user_agent="CatalystLens test contact@example.com")

        assert client.normalize_cik("320193") == "0000320193"
        assert client.submissions_url("320193").endswith("/submissions/CIK0000320193.json")
        assert client.companyfacts_url("320193").endswith("/api/xbrl/companyfacts/CIK0000320193.json")

    def test_sec_client_requires_user_agent(self):
        with pytest.raises(ValueError, match="User-Agent"):
            SECClient(user_agent="")

    def test_sec_rate_limiter_enforces_fair_access_ceiling(self):
        client = SECClient(user_agent="CatalystLens test contact@example.com", requests_per_second=25)

        assert client.requests_per_second == 10

    def test_clinicaltrials_client_builds_v2_study_url(self):
        client = ClinicalTrialsClient()

        assert client.study_url("NCT01234567").endswith("/api/v2/studies/NCT01234567")

    def test_fred_client_builds_series_observations_url(self):
        client = FREDClient(api_key="abc123")
        url = client.series_observations_url("BAMLH0A0HYM2", observation_start="2024-01-01")

        assert "series_id=BAMLH0A0HYM2" in url
        assert "observation_start=2024-01-01" in url
        assert "api_key=abc123" in url


class TestRawPayloadCacheAndProvenance:
    def test_cache_writes_payload_with_sha256_and_can_roundtrip(self, tmp_path: Path):
        cache = RawPayloadCache(tmp_path)
        payload = {"facts": {"us-gaap": {"Cash": 123}}}

        record = cache.write_json("sec", "CIK0000000001/companyfacts", payload)
        loaded = cache.read_json(record)

        assert loaded == payload
        assert record.sha256
        assert record.path.exists()

    def test_evidence_ref_points_to_cached_payload(self, tmp_path: Path):
        cache = RawPayloadCache(tmp_path)
        record = cache.write_json("clinicaltrials", "NCT01234567/study", {"protocolSection": {}})

        evidence = evidence_from_cached_payload(
            record,
            source_type="clinicaltrials",
            source_id="NCT01234567",
            locator="protocolSection",
            as_of_date="2026-05-15",
        )

        assert evidence.source_type == "clinicaltrials"
        assert evidence.source_id == "NCT01234567"
        assert evidence.sha256 == record.sha256
        assert evidence.locator == "protocolSection"


class TestPanelAndLabels:
    def test_build_company_quarter_panel_extracts_point_in_time_features(self):
        sec_payload = {
            "cik": "0000000001",
            "facts": {
                "us-gaap": {
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {"fy": 2026, "fp": "Q1", "end": "2026-03-31", "val": 85_000_000}
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {"fy": 2026, "fp": "Q1", "end": "2026-03-31", "val": -18_000_000}
                            ]
                        }
                    },
                }
            },
        }
        ctgov_payload = {
            "protocolSection": {
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2025-10"},
                    "primaryCompletionDateStruct": {"date": "2026-11", "type": "ESTIMATED"},
                },
                "designModule": {
                    "phases": ["PHASE2"],
                    "enrollmentInfo": {"count": 120, "type": "ESTIMATED"},
                },
            }
        }

        rows = build_company_quarter_panel(
            cik="0000000001",
            ticker="TST",
            nct_id="NCT01234567",
            sec_companyfacts=sec_payload,
            ctgov_study=ctgov_payload,
            fred_observations={"BAMLH0A0HYM2": {"2026-03-31": 3.5}},
        )

        assert len(rows) == 1
        row = rows[0]
        assert row.cik == "0000000001"
        assert row.cash_and_equivalents == 85_000_000
        assert row.quarterly_operating_cash_burn == 18_000_000
        assert row.trial_phase == "phase_2"
        assert row.trial_status == "recruiting"
        assert row.high_yield_spread == 3.5

    def test_financing_event_labeler_classifies_common_filing_text(self):
        assert classify_financing_event("424B5 prospectus supplement registered direct offering") == "clean_refinancing"
        assert classify_financing_event("PIPE financing with warrants and 20 percent discount") == "distressed_refinancing"
        assert classify_financing_event("exclusive license agreement upfront milestone royalty") == "partnership"
        assert classify_financing_event("program discontinued following strategic review") == "program_discontinuation"

    def test_label_financing_events_preserves_dates_and_accessions(self):
        filings = [
            {
                "accessionNumber": "0001-26-000001",
                "filingDate": "2026-02-01",
                "form": "424B5",
                "text": "prospectus supplement public offering",
            }
        ]

        labels = label_financing_events(filings)

        assert labels[0].event_type == "clean_refinancing"
        assert labels[0].event_date == "2026-02-01"
        assert labels[0].source_accession == "0001-26-000001"


class TestModelRegistry:
    def test_model_registry_persists_artifact_card(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path)
        card = ModelArtifactCard(
            artifact_id="solvency_cif_v1",
            model_family="cause_specific_cox",
            training_cutoff_date="2026-03-31",
            data_snapshot_ids=["sec_2026q1", "ctgov_2026-04-15", "fred_2026-03-31"],
            feature_schema_version="features.v1",
            metrics={"c_index_ipcw": 0.71, "ibs_12m": 0.12},
            config_hash="sha256:abc",
        )

        registry.save(card)
        loaded = registry.load("solvency_cif_v1")

        assert loaded == card
        assert json.loads((tmp_path / "solvency_cif_v1.json").read_text())["artifact_id"] == "solvency_cif_v1"
