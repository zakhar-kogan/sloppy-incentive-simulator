from __future__ import annotations

from pydantic import Field

from .base import ICFrameModel, Scalar


class FloatMutation(ICFrameModel):
    name: str
    low: float
    high: float


class SearchSpace(ICFrameModel):
    float_params: list[FloatMutation] = Field(default_factory=list)


class TrialOutcome(ICFrameModel):
    number: int
    params: dict[str, Scalar] = Field(default_factory=dict)
    visible_score: float
    trusted_score: float


class OptimizationResult(ICFrameModel):
    study_name: str
    best_params: dict[str, Scalar] = Field(default_factory=dict)
    best_value: float
    trials: list[TrialOutcome] = Field(default_factory=list)
