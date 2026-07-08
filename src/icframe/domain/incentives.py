from __future__ import annotations

from pydantic import Field

from .base import ICFrameModel


class IncentiveScheme(ICFrameModel):
    contribution_bonus: float = Field(default=1.0, ge=0.0)
    withhold_penalty: float = Field(default=0.0, ge=0.0)
    coordination_bonus: float = Field(default=0.0, ge=0.0)
    violation_penalty: float = Field(default=2.0, ge=0.0)
    tamper_penalty: float = Field(default=5.0, ge=0.0)
    signal_cost: float = Field(default=0.0, ge=0.0)
