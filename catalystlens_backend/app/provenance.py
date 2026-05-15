"""Provenance helpers for source-traced audit inputs."""

from __future__ import annotations

from app.data_sources.cache import CachedPayloadRecord
from app.models.schemas import EvidenceRef, SourceType


def evidence_from_cached_payload(
    record: CachedPayloadRecord,
    source_type: SourceType,
    source_id: str,
    locator: str,
    as_of_date: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        source_type=source_type,
        source_id=source_id,
        as_of_date=as_of_date,
        locator=locator,
        sha256=record.sha256,
    )
