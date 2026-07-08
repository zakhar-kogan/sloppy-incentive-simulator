from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from .base import ICFrameModel
from .evaluation import ObjectiveWeights
from .incentives import IncentiveScheme
from .norms import LawProgram
from .topology import TopologyConfig


class AgentPolicy(StrEnum):
    COOPERATIVE = "cooperative"
    OPPORTUNISTIC = "opportunistic"
    SIGNALER = "signaler"
    TAMPERER = "tamperer"


class AgentConfig(ICFrameModel):
    name: str
    policy: AgentPolicy
    endowment: float = Field(default=10.0, ge=0.0)


class SimulationConfig(ICFrameModel):
    rounds: int = Field(default=5, ge=1)
    seed: int = Field(default=0, ge=0)
    contribution_amount: float = Field(default=2.0, gt=0.0)
    public_return_multiplier: float = Field(default=1.5, gt=0.0)
    message_bonus: float = Field(default=0.25, ge=0.0)
    tamper_probability: float = Field(default=0.0, ge=0.0, le=1.0)


class Scenario(ICFrameModel):
    name: str
    description: str
    agents: list[AgentConfig] = Field(min_length=1)
    laws: LawProgram
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    incentives: IncentiveScheme = Field(default_factory=IncentiveScheme)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    visible_objective: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
    trusted_objective: ObjectiveWeights = Field(
        default_factory=lambda: ObjectiveWeights(collusion_penalty=2.0, tamper_penalty=3.0)
    )
    baseline_visible_score: float = 0.0
    baseline_trusted_score: float = 0.0

    @property
    def agent_names(self) -> tuple[str, ...]:
        return tuple(agent.name for agent in self.agents)
