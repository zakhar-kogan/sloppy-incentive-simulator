from __future__ import annotations

from pydantic import Field

from .base import ICFrameModel
from .events import Event


class AgentSnapshot(ICFrameModel):
    name: str
    balance: float
    payoff: float = 0.0
    contributions: int = 0
    withholds: int = 0
    sent_messages: int = 0
    received_messages: int = 0
    violations: int = 0
    last_action: str | None = None


class WorldSnapshot(ICFrameModel):
    step: int = Field(ge=0)
    public_pool: float = 0.0
    allowed_actions: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    agents: list[AgentSnapshot] = Field(default_factory=list)


class InteractionEdge(ICFrameModel):
    source: str
    target: str
    weight: float = 0.0
    event_count: int = 0


class InteractionGraph(ICFrameModel):
    nodes: tuple[str, ...] = ()
    edges: list[InteractionEdge] = Field(default_factory=list)


class SimulationTrace(ICFrameModel):
    scenario_name: str
    seed: int = Field(ge=0)
    events: list[Event] = Field(default_factory=list)
    snapshots: list[WorldSnapshot] = Field(default_factory=list)
    graph: InteractionGraph | None = None

    @property
    def final_snapshot(self) -> WorldSnapshot:
        if not self.snapshots:
            raise ValueError("simulation trace is missing snapshots")
        return self.snapshots[-1]
