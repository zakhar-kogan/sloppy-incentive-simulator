from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import (
    Availability,
    ConstraintOperator,
    EffectScope,
    GraphVisibility,
    MetricScope,
    MetricType,
    NormStatus,
    Operation,
    OutcomeVector,
    OutcomeVisibility,
    PolicyKind,
    ScheduleMode,
)


@dataclass(frozen=True, slots=True)
class CompiledEffect:
    scope: EffectScope
    population: str | None
    operation: Operation
    values: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class CompiledStateUpdate:
    scope: EffectScope
    population: str | None
    field: tuple[str, ...]
    operation: Operation
    value: Scalar


@dataclass(frozen=True, slots=True)
class CompiledEnforcement:
    audit_probability: float
    detection_probability: float
    false_positive_probability: float
    false_negative_probability: float
    enforcement_probability: float
    sanctions: tuple[CompiledEffect, ...]
    compliance_rewards: tuple[CompiledEffect, ...]
    remediation_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledTransition:
    id: str
    from_state: str
    action: str
    to_state: str
    availability: Availability
    norm_status: NormStatus
    requires_target: bool
    target_populations: frozenset[str]
    tags: tuple[str, ...]
    effects: tuple[CompiledEffect, ...]
    state_updates: tuple[CompiledStateUpdate, ...]
    enforcement: CompiledEnforcement | None
    prompt_label: str | None
    prompt_description: str | None
    explanation_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledVisibility:
    graph: GraphVisibility
    outcomes: OutcomeVisibility
    sanctions: OutcomeVisibility
    prompts: bool
    history_events: int


@dataclass(frozen=True, slots=True)
class CompiledMetric:
    name: str
    type: MetricType
    channel: str | None
    scope: MetricScope
    required_tags: frozenset[str]
    left: str | None
    right: str | None
    numerator: str | None
    denominator: str | None
    terms: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class CompiledConstraint:
    metric: str
    operator: ConstraintOperator
    threshold: float


@dataclass(slots=True)
class AgentState:
    id: str
    archetype: str
    population: str
    role: str
    state: str
    resources: dict[str, float]
    attributes: dict[str, Scalar]
    scalarizer: dict[str, float]
    policy_kind: PolicyKind
    visibility_profile: str
    history: deque[dict[str, Any]] = field(default_factory=deque)

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            id=self.id,
            archetype=self.archetype,
            population=self.population,
            role=self.role,
            state=self.state,
            resources=dict(self.resources),
            attributes=dict(self.attributes),
        )


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    id: str
    archetype: str
    population: str
    role: str
    state: str
    resources: dict[str, float]
    attributes: dict[str, Scalar]


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    step: int
    global_values: dict[str, Scalar]
    agents: dict[str, AgentSnapshot]


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    transition_id: str
    action: str
    target_id: str | None
    norm_status: NormStatus
    tags: tuple[str, ...]
    visible_outcomes: OutcomeVector
    visible_sanctions: OutcomeVector
    prompt_label: str | None
    prompt_description: str | None

    @property
    def key(self) -> str:
        return f"{self.action}@{self.target_id}" if self.target_id else self.action


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    run_id: str
    step: int
    agent_id: str
    state: str
    resources: dict[str, float]
    candidates: tuple[ActionCandidate, ...]
    visible_history: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class PolicyDecision:
    decision_id: str
    observation_id: str
    step: int
    agent_id: str
    policy: PolicyKind
    candidate_keys: tuple[str, ...]
    action: str | None = None
    target_id: str | None = None
    estimated_rewards: dict[str, float] = field(default_factory=dict)
    probability: float | None = None
    rationale: str | None = None
    failure: str | None = None
    state_delta: dict[str, Any] = field(default_factory=dict)
    llm_call: dict[str, Any] | None = None


@dataclass(slots=True)
class PolicyChoice:
    action: str | None = None
    target_id: str | None = None
    estimated_rewards: dict[str, float] = field(default_factory=dict)
    probability: float | None = None
    rationale: str | None = None
    failure: str | None = None
    llm_call: dict[str, Any] | None = None


@dataclass(slots=True)
class RuntimeEvent:
    event_id: str
    step: int
    actor_id: str
    target_id: str | None
    transition_id: str
    action: str
    from_state: str
    to_state: str
    availability: Availability
    norm_status: NormStatus
    tags: tuple[str, ...]
    outcomes_by_agent: dict[str, OutcomeVector] = field(default_factory=dict)
    global_outcome: OutcomeVector = field(default_factory=dict)
    scalar_rewards: dict[str, float] = field(default_factory=dict)
    audit_sampled: bool = False
    detected: bool = False
    enforced: bool = False
    explanation_reasons: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    remediation_actions: tuple[str, ...] = ()
    counts_as_action: bool = True


@dataclass(frozen=True, slots=True)
class PolicyFeedback:
    state: str
    action: str
    target_id: str | None
    reward: float
    next_state: str
    observation: Observation


@dataclass(slots=True)
class WorldState:
    step: int
    global_values: dict[str, Scalar]
    agents: dict[str, AgentState]

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            step=self.step,
            global_values=dict(self.global_values),
            agents={agent_id: agent.snapshot() for agent_id, agent in self.agents.items()},
        )


@dataclass(slots=True)
class StepResult:
    step: int
    events: list[RuntimeEvent]
    decisions: list[PolicyDecision]
    observations: list[Observation]
    rewards: dict[str, float]
    terminated: bool


@dataclass(frozen=True, slots=True)
class EngineConfig:
    run_id: str
    seed: int
    schedule: ScheduleMode
