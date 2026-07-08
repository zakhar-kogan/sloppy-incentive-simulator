from __future__ import annotations

from pydantic import Field

from .base import ICFrameModel


class ObjectiveWeights(ICFrameModel):
    efficiency: float = 1.0
    equality: float = 1.0
    compliance: float = 1.0
    throughput: float = 0.25
    collusion_penalty: float = 1.0
    tamper_penalty: float = 1.0
    reward_hacking_penalty: float = 1.0


class GraphMetrics(ICFrameModel):
    node_count: int = 0
    edge_count: int = 0
    reciprocity: float = 0.0
    max_pair_share: float = 0.0
    collusion_index: float = 0.0


class EvaluationMetrics(ICFrameModel):
    total_contributions: float
    total_payoff: float
    average_payoff: float
    gini: float
    violation_count: int
    reward_hacking_events: int
    tamper_events: int
    throughput: int
    signal_volume: int
    graph: GraphMetrics = Field(default_factory=GraphMetrics)


class FailureDiagnostics(ICFrameModel):
    goodhart_gaming: bool = False
    reward_hacking: bool = False
    collusion: bool = False
    system_hacking: bool = False
    notes: list[str] = Field(default_factory=list)


class EvaluationResult(ICFrameModel):
    visible_score: float
    trusted_score: float
    metrics: EvaluationMetrics
    diagnostics: FailureDiagnostics
