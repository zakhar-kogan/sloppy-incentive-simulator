from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from .base import ICFrameModel, Scalar
from .incentive_spec import RetentionProfile


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class StudyMode(StrEnum):
    SINGLE = "single"
    PARETO = "pareto"


class RunConfig(ICFrameModel):
    seed: int | None = None
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    retention: RetentionProfile | None = None
    sample_every_steps: int | None = Field(default=None, ge=1)
    artifact_root: Path = Path(".artifacts/icframe")
    run_id: str | None = None


class LiveLLMBudget(ICFrameModel):
    enabled: bool = False
    max_calls: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    concurrency: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def explicit_limits(self) -> LiveLLMBudget:
        if self.enabled and (self.max_calls is None or self.max_cost_usd is None):
            raise ValueError("live LLM studies require max_calls and max_cost_usd")
        if not self.enabled and (
            self.max_calls is not None or self.max_cost_usd is not None or self.concurrency != 1
        ):
            raise ValueError("LLM budgets require enabled=true")
        return self


class ParameterRange(ICFrameModel):
    minimum: float | int
    maximum: float | int

    @model_validator(mode="after")
    def ordered(self) -> ParameterRange:
        if self.minimum > self.maximum:
            raise ValueError("parameter range minimum exceeds maximum")
        return self


class StudyConfig(ICFrameModel):
    mode: StudyMode
    objectives: list[str] = Field(min_length=1)
    parameters: list[str] = Field(default_factory=list)
    parameter_ranges: dict[str, ParameterRange] = Field(default_factory=dict)
    trials: int = Field(default=20, ge=1)
    seeds: list[int] = Field(min_length=1)
    workers: int = Field(
        default_factory=lambda: min(4, os.cpu_count() or 1),
        ge=1,
    )
    artifact_root: Path = Path(".artifacts/icframe")
    study_id: str | None = None
    live_llm: LiveLLMBudget = Field(default_factory=LiveLLMBudget)

    @model_validator(mode="after")
    def mode_objectives(self) -> StudyConfig:
        if self.mode is StudyMode.SINGLE and len(self.objectives) != 1:
            raise ValueError("single-objective studies require exactly one objective")
        if self.mode is StudyMode.PARETO and len(self.objectives) < 2:
            raise ValueError("Pareto studies require at least two objectives")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("study seeds contain duplicates")
        if self.live_llm.enabled and self.workers != 1:
            raise ValueError("live LLM studies require workers=1")
        return self


class ConstraintResult(ICFrameModel):
    metric: str
    value: float
    threshold: float
    operator: str
    passed: bool


class Checkpoint(ICFrameModel):
    step: int
    metrics: dict[str, float] = Field(default_factory=dict)
    action_counts: dict[str, int] = Field(default_factory=dict)
    transition_counts: dict[str, int] = Field(default_factory=dict)
    tag_counts: dict[str, int] = Field(default_factory=dict)


class AgentStatistics(ICFrameModel):
    action_counts: dict[str, int] = Field(default_factory=dict)
    reward: float = 0.0
    failed_decisions: int = 0
    violations: int = 0
    detections: int = 0
    enforcement: int = 0


class LLMUsageBreakdown(ICFrameModel):
    provider: str
    model: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = 0.0


class LLMLatencyBucket(ICFrameModel):
    upper_ms: int | None
    count: int = 0


class LLMUsageSummary(ICFrameModel):
    attempted: int = 0
    completed: int = 0
    failed: int = 0
    malformed: int = 0
    invalid: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = 0.0
    retry_count: int = 0
    fallback_count: int = 0
    approximate_p50_ms: int | None = None
    approximate_p95_ms: int | None = None
    latency_buckets: list[LLMLatencyBucket] = Field(default_factory=list)
    breakdown: dict[str, LLMUsageBreakdown] = Field(default_factory=dict)


class AgentResult(ICFrameModel):
    id: str
    archetype: str
    role: str
    state: str
    resources: dict[str, float] = Field(default_factory=dict)
    policy: str
    policy_state: dict[str, object] = Field(default_factory=dict)
    statistics: AgentStatistics = Field(default_factory=AgentStatistics)


class RunSummary(ICFrameModel):
    run_id: str
    pack_id: str
    spec_name: str
    seed: int
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    status: RunStatus
    retention: RetentionProfile
    steps_planned: int
    steps_completed: int
    event_count: int
    metrics: dict[str, float] = Field(default_factory=dict)
    objectives: dict[str, float] = Field(default_factory=dict)
    constraints: list[ConstraintResult] = Field(default_factory=list)
    feasible: bool = True
    action_counts: dict[str, int] = Field(default_factory=dict)
    transition_counts: dict[str, int] = Field(default_factory=dict)
    tag_counts: dict[str, int] = Field(default_factory=dict)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    agents: list[AgentResult] = Field(default_factory=list)
    llm_calls: int = 0
    estimated_llm_cost_usd: float | None = 0.0
    llm_usage: LLMUsageSummary = Field(default_factory=LLMUsageSummary)
    replayable: bool = True
    replay_reason: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


class SeedResult(ICFrameModel):
    seed: int
    metrics: dict[str, float]
    objectives: dict[str, float]
    feasible: bool
    constraints: list[ConstraintResult] = Field(default_factory=list)
    llm_calls: int = 0
    estimated_llm_cost_usd: float | None = 0.0


class TrialRecord(ICFrameModel):
    number: int
    parameters: dict[str, Scalar]
    seeds: list[SeedResult]
    objective_values: dict[str, float]
    feasible: bool
    llm_calls: int = 0
    estimated_llm_cost_usd: float | None = 0.0
    runtime_hash: str = ""
    hook_hash: str = ""
    state: str = "complete"
    error: str | None = None


class StudySummary(ICFrameModel):
    study_id: str
    pack_id: str
    mode: StudyMode
    status: RunStatus
    objectives: list[str]
    parameters: list[str]
    seeds: list[int]
    trial_count: int
    trials: list[TrialRecord] = Field(default_factory=list)
    best_trial: int | None = None
    pareto_trials: list[int] = Field(default_factory=list)
    retained_run_ids: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
