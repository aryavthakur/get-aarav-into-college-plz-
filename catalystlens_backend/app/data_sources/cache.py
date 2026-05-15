"""
Raw payload cache with content hashing.

Training and audit replay need immutable raw inputs. This cache writes JSON
payloads under a deterministic namespace and records a SHA-256 content hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedPayloadRecord:
    namespace: str
    key: str
    path: Path
    sha256: str


class RawPayloadCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_json(self, namespace: str, key: str, payload: dict[str, Any]) -> CachedPayloadRecord:
        relative = Path(namespace) / f"{key}.json"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.write_bytes(encoded)
        return CachedPayloadRecord(
            namespace=namespace,
            key=key,
            path=path,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def read_json(self, record: CachedPayloadRecord) -> dict[str, Any]:
        return json.loads(record.path.read_text())
