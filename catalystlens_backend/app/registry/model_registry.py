"""Lightweight JSON model registry for frozen training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class ModelArtifactCard(BaseModel):
    artifact_id: str
    model_family: str
    training_cutoff_date: str
    data_snapshot_ids: List[str] = Field(default_factory=list)
    feature_schema_version: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    config_hash: str


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, card: ModelArtifactCard) -> Path:
        path = self.root / f"{card.artifact_id}.json"
        path.write_text(json.dumps(card.model_dump(), sort_keys=True, indent=2))
        return path

    def load(self, artifact_id: str) -> ModelArtifactCard:
        path = self.root / f"{artifact_id}.json"
        return ModelArtifactCard(**json.loads(path.read_text()))
