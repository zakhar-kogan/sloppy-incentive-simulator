from __future__ import annotations

from pydantic import Field

from .base import ICFrameModel, Scalar
from .evaluation import EvaluationMetrics, FailureDiagnostics
from .state import AgentSnapshot, InteractionEdge


class AgentSeriesPoint(ICFrameModel):
    step: int
    name: str
    balance: float
    payoff: float
    contributions: int
    withholds: int
    sent_messages: int
    received_messages: int
    violations: int
    last_action: str | None = None


class StepSummary(ICFrameModel):
    step: int
    public_pool: float
    total_balance: float
    total_payoff: float
    event_counts: dict[str, int] = Field(default_factory=dict)


class ExperimentSummary(ICFrameModel):
    run_id: str
    scenario_name: str
    seed: int
    visible_score: float
    trusted_score: float
    score_gap: float
    metrics: EvaluationMetrics
    diagnostics: FailureDiagnostics
    best_params: dict[str, Scalar] = Field(default_factory=dict)
    event_counts: dict[str, int] = Field(default_factory=dict)
    agent_outcomes: list[AgentSnapshot] = Field(default_factory=list)
    agent_series: list[AgentSeriesPoint] = Field(default_factory=list)
    step_summaries: list[StepSummary] = Field(default_factory=list)
    graph_edges: list[InteractionEdge] = Field(default_factory=list)
