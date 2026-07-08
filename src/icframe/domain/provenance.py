from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .base import ICFrameModel, Scalar
from .evaluation import EvaluationResult


class RunProvenance(ICFrameModel):
    run_id: str
    scenario_name: str
    scenario_hash: str
    seed: int
    created_at: datetime
    package_version: str
    evaluation: EvaluationResult
    best_params: dict[str, Scalar] = Field(default_factory=dict)
    study_name: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
