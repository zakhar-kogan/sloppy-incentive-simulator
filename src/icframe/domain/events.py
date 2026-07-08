from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from .base import ICFrameModel, Scalar


class EventKind(StrEnum):
    CONTRIBUTE = "contribute"
    WITHHOLD = "withhold"
    SIGNAL = "signal"
    REWARD = "reward"
    SANCTION = "sanction"
    VIOLATION = "violation"
    TAMPER = "tamper"


class Event(ICFrameModel):
    step: int = Field(ge=0)
    actor: str
    kind: EventKind
    target: str | None = None
    amount: float = 0.0
    reward: float = 0.0
    tags: tuple[str, ...] = ()
    metadata: dict[str, Scalar] = Field(default_factory=dict)
