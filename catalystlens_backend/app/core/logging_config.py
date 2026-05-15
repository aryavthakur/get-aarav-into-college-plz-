"""
Structured logging configuration for CatalystLens.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger("catalystlens")


def audit_event(event: str, **fields: Any) -> None:
    """Emit a structured audit log line."""
    parts = " ".join(f"{k}={v!r}" for k, v in fields.items())
    logger.info("event=%r %s", event, parts)


@contextmanager
def timed_audit(event: str, **fields: Any):
    """Context manager that logs start/end of an audit step with elapsed time."""
    request_id = str(uuid.uuid4())[:8]
    audit_event(f"{event}_started", request_id=request_id, **fields)
    t0 = time.perf_counter()
    try:
        yield request_id
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        audit_event(f"{event}_completed", request_id=request_id, elapsed_ms=elapsed_ms, **fields)
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        audit_event(f"{event}_failed", request_id=request_id, elapsed_ms=elapsed_ms,
                    error=str(exc), **fields)
        raise
