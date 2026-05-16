"""Lightweight JSON model registry for frozen training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


CalibrationStatus = Literal[
    "synthetic_test_only",
    "preliminary_backtest",
    "insufficient_data",
    "externally_validated",
]


class ModelArtifactCard(BaseModel):
    artifact_id: str
    model_family: str
    training_cutoff_date: str
    data_snapshot_ids: List[str] = Field(default_factory=list)
    feature_schema_version: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    config_hash: str
    training_dataset_id: Optional[str] = None
    validation_dataset_id: Optional[str] = None
    n_training_examples: int = 0
    n_validation_examples: int = 0
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    calibration_status: CalibrationStatus = "insufficient_data"
    validation_report_path: Optional[str] = None
    trained_artifact_path: Optional[str] = None

    @model_validator(mode="after")
    def validation_status_must_match_evidence(self) -> "ModelArtifactCard":
        if self.validation_dataset_id and "synthetic" in self.validation_dataset_id.lower():
            if self.calibration_status == "externally_validated":
                raise ValueError("synthetic validation cannot be marked externally_validated")
            if not self.validation_metrics and self.calibration_status != "synthetic_test_only":
                self.calibration_status = "synthetic_test_only"
        if not self.validation_metrics and self.calibration_status not in {"synthetic_test_only", "insufficient_data"}:
            self.calibration_status = "insufficient_data"
        return self


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

    def list_cards(self) -> List[ModelArtifactCard]:
        return [
            ModelArtifactCard(**json.loads(path.read_text()))
            for path in sorted(self.root.glob("*.json"))
        ]
