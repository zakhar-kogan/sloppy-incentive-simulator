from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import ICFrameModel, Scalar

OutcomeVector = dict[str, float]


class IncentiveSpecModel(ICFrameModel):
    """Boundary model for TOML-backed IncentiveSpec data.

    The legacy ICFrame models stay fully strict. IncentiveSpec accepts string
    enum values from TOML, while still forbidding unknown fields and validating
    cross-references explicitly.
    """

    model_config = ConfigDict(strict=False, extra="forbid", validate_assignment=True)


class PolicyBackend(StrEnum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC_WEIGHTED = "stochastic_weighted"
    EPSILON_GREEDY_BANDIT = "epsilon_greedy_bandit"
    UCB_BANDIT = "ucb_bandit"
    Q_LEARNING_SIMPLE = "q_learning_simple"
    SCRIPTED = "scripted"
    LLM_POLICY = "llm_policy"


class Availability(StrEnum):
    HARD_AVAILABLE = "hard_available"
    HARD_BLOCKED = "hard_blocked"
    POSSIBLE_VIOLATION = "possible_violation"


class NormStatus(StrEnum):
    PERMITTED = "permitted"
    FORBIDDEN = "forbidden"
    OBLIGATORY = "obligatory"
    DISCOURAGED = "discouraged"
    UNKNOWN = "unknown"


class ScheduleMode(StrEnum):
    SEQUENTIAL_FIXED = "sequential_fixed"
    SEQUENTIAL_RANDOM = "sequential_random"
    PARALLEL_SIMULTANEOUS = "parallel_simultaneous"
    STAGED = "staged"


class GraphVisibility(StrEnum):
    FULL_GRAPH = "full_graph"
    LOCAL_GRAPH = "local_graph"
    DISCOVERED_GRAPH = "discovered_graph"
    PROMPT_ONLY = "prompt_only"
    BLACK_BOX = "black_box"
    NONE = "none"


class OutcomeVisibility(StrEnum):
    FULL_NUMERIC = "full_numeric"
    OWN_SCALAR = "own_scalar"
    SIGN_ONLY = "sign_only"
    ORDINAL = "ordinal"
    NOISY_NUMERIC = "noisy_numeric"
    LABEL_ONLY = "label_only"
    HIDDEN = "hidden"
    LEARNED = "learned"


class EffectOperation(StrEnum):
    ADD = "add"
    MULTIPLY = "multiply"
    SET = "set"


class MetricType(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    RATE = "rate"
    DIFFERENCE = "difference"
    RATIO = "ratio"
    ZSCORE_DIFFERENCE = "zscore_difference"
    ROLLING_MEAN = "rolling_mean"
    EVENT_COUNT = "event_count"
    EVENT_RATE = "event_rate"


class SpecHeader(IncentiveSpecModel):
    version: Literal["0.2"]
    name: str
    domain: str = "generic"


class ExperimentConfig(IncentiveSpecModel):
    steps: int = Field(default=1, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0])
    schedule: ScheduleMode = ScheduleMode.SEQUENTIAL_RANDOM


class OutcomeSpace(IncentiveSpecModel):
    channels: list[str] = Field(min_length=1)

    @field_validator("channels")
    @classmethod
    def channels_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("outcome_space.channels contains duplicates")
        return value


class StateSpace(IncentiveSpecModel):
    initial_global: str
    all: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def initial_state_is_declared(self) -> StateSpace:
        if self.initial_global not in self.all:
            raise ValueError("states.initial_global must be declared in states.all")
        return self


class ActionSpace(IncentiveSpecModel):
    all: list[str] = Field(min_length=1)


class VisibilityProfile(IncentiveSpecModel):
    graph: GraphVisibility
    observed_outcomes: OutcomeVisibility = OutcomeVisibility.HIDDEN
    latent_outcomes: OutcomeVisibility = OutcomeVisibility.HIDDEN
    governance_outcomes: OutcomeVisibility = OutcomeVisibility.HIDDEN
    sanctions: OutcomeVisibility = OutcomeVisibility.HIDDEN
    audit_probabilities: OutcomeVisibility = OutcomeVisibility.HIDDEN
    detection_probabilities: OutcomeVisibility = OutcomeVisibility.HIDDEN
    other_agents_outcomes: OutcomeVisibility = OutcomeVisibility.HIDDEN
    prompts: bool = False


class MemoryConfig(IncentiveSpecModel):
    enabled: bool = False
    mode: str = "discovered_graph"
    max_events: int = Field(default=200, ge=0)
    learn_transition_outcomes: bool = True
    learn_audit_probabilities: bool = True
    forgetting_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMConfig(IncentiveSpecModel):
    model: str
    temperature: float = Field(default=0.0, ge=0.0)
    max_context_events: int = Field(default=20, ge=0)
    include_action_descriptions: bool = True
    include_visible_graph: bool = True
    include_visible_rewards: bool = False
    require_json_action: bool = True
    system_prompt: str = ""


class Archetype(IncentiveSpecModel):
    policy: PolicyBackend
    role: str
    visibility_profile: str
    scalarizer: dict[str, float] = Field(default_factory=dict)
    behavior: dict[str, float] = Field(default_factory=dict)
    memory: MemoryConfig | None = None
    llm: LLMConfig | None = None
    initial_state: str | None = None
    initial_resources: dict[str, float] = Field(default_factory=dict)


class PopulationEntry(IncentiveSpecModel):
    archetype: str
    count: int = Field(ge=1)


class ActorSelector(IncentiveSpecModel):
    population: list[str] | str | None = None
    archetype: list[str] | str | None = None
    role: list[str] | str | None = None
    attributes: dict[str, Scalar] = Field(default_factory=dict)


class Selector(IncentiveSpecModel):
    actor: ActorSelector | None = None
    target: dict[str, Scalar | list[str]] = Field(default_factory=dict)
    task: dict[str, Scalar | list[str]] = Field(default_factory=dict)
    global_state: dict[str, Scalar | list[str]] = Field(default_factory=dict, alias="global")
    relation: dict[str, Scalar | list[str]] = Field(default_factory=dict)


class ConditionalEffect(IncentiveSpecModel):
    priority: int = 0
    operation: EffectOperation = EffectOperation.ADD
    selector: Selector = Field(default_factory=Selector)
    effects: OutcomeVector = Field(default_factory=dict)
    effects_if_detected: OutcomeVector = Field(default_factory=dict)


class Enforcement(IncentiveSpecModel):
    audit_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    detection_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    false_positive_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    false_negative_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    enforcement_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    sanction_if_detected: OutcomeVector = Field(default_factory=dict)
    reward_if_compliant: OutcomeVector = Field(default_factory=dict)
    restorative_action: str | None = None
    appeal_action: str | None = None


class PromptDescription(IncentiveSpecModel):
    public: str | None = None
    actor_view: str | None = None
    auditor_view: str | None = None
    hidden_designer_note: str | None = None


class Transition(IncentiveSpecModel):
    model_config = ConfigDict(
        strict=False,
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    id: str
    from_state: str = Field(alias="from")
    action: str
    to_state: str = Field(alias="to")
    availability: Availability = Availability.HARD_AVAILABLE
    norm_status: NormStatus = NormStatus.UNKNOWN
    tags: list[str] = Field(default_factory=list)
    effects: OutcomeVector = Field(default_factory=dict)
    conditional_effects: list[ConditionalEffect] = Field(default_factory=list)
    enforcement: Enforcement | None = None
    prompt: PromptDescription | None = None
    state_updates: dict[str, Any] = Field(default_factory=dict)


class MetricSpec(IncentiveSpecModel):
    type: MetricType
    channel: str | None = None
    proxy: str | None = None
    target: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    normalization: str | None = None
    where_tags_include: list[str] = Field(default_factory=list)


class IncentiveSpec(IncentiveSpecModel):
    spec: SpecHeader
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    outcome_space: OutcomeSpace
    states: StateSpace
    actions: ActionSpace
    visibility_profiles: dict[str, VisibilityProfile] = Field(default_factory=dict)
    archetypes: dict[str, Archetype] = Field(default_factory=dict)
    population: list[PopulationEntry] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)
    metrics: dict[str, MetricSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> IncentiveSpec:
        channels = set(self.outcome_space.channels)
        states = set(self.states.all)
        actions = set(self.actions.all)
        visibility_profiles = set(self.visibility_profiles)
        archetypes = set(self.archetypes)
        transition_ids = set()

        for name, archetype in self.archetypes.items():
            unknown_channels = set(archetype.scalarizer) - channels
            if unknown_channels:
                raise ValueError(
                    f"archetype {name} references undeclared channels: {sorted(unknown_channels)}"
                )
            if archetype.visibility_profile not in visibility_profiles:
                raise ValueError(
                    f"archetype {name} references unknown visibility profile "
                    f"{archetype.visibility_profile!r}"
                )
            if archetype.initial_state is not None and archetype.initial_state not in states:
                raise ValueError(f"archetype {name} references unknown initial_state")

        for entry in self.population:
            if entry.archetype not in archetypes:
                raise ValueError(f"population references unknown archetype {entry.archetype!r}")

        for transition in self.transitions:
            if transition.id in transition_ids:
                raise ValueError(f"duplicate transition id {transition.id!r}")
            transition_ids.add(transition.id)
            if transition.from_state not in states:
                raise ValueError(f"transition {transition.id} references unknown from state")
            if transition.to_state not in states:
                raise ValueError(f"transition {transition.id} references unknown to state")
            if transition.action not in actions:
                raise ValueError(f"transition {transition.id} references unknown action")
            self._check_channels(
                channels,
                transition.effects,
                f"transition {transition.id}.effects",
            )
            if transition.enforcement is not None:
                self._check_channels(
                    channels,
                    transition.enforcement.sanction_if_detected,
                    f"transition {transition.id}.enforcement.sanction_if_detected",
                )
                self._check_channels(
                    channels,
                    transition.enforcement.reward_if_compliant,
                    f"transition {transition.id}.enforcement.reward_if_compliant",
                )
                for action in (
                    transition.enforcement.restorative_action,
                    transition.enforcement.appeal_action,
                ):
                    if action is not None and action not in actions:
                        raise ValueError(
                            f"transition {transition.id} enforcement references unknown action"
                        )
            for conditional in transition.conditional_effects:
                self._check_channels(
                    channels,
                    conditional.effects,
                    f"transition {transition.id}.conditional_effects.effects",
                )
                self._check_channels(
                    channels,
                    conditional.effects_if_detected,
                    f"transition {transition.id}.conditional_effects.effects_if_detected",
                )

        for name, metric in self.metrics.items():
            for field_name in ("channel", "proxy", "target"):
                channel = getattr(metric, field_name)
                if (
                    channel is not None
                    and not channel.startswith("metric.")
                    and channel not in channels
                ):
                    raise ValueError(f"metric {name} references undeclared channel {channel!r}")
            for field_name in ("numerator", "denominator"):
                reference = getattr(metric, field_name)
                if (
                    reference is not None
                    and reference != "all_actions"
                    and not reference.startswith("metric.")
                    and reference not in channels
                ):
                    raise ValueError(f"metric {name} references undeclared value {reference!r}")
        return self

    @staticmethod
    def _check_channels(channels: set[str], effects: OutcomeVector, location: str) -> None:
        unknown_channels = set(effects) - channels
        if unknown_channels:
            raise ValueError(
                f"{location} references undeclared channels: {sorted(unknown_channels)}"
            )


def load_incentive_spec(path: str | Path) -> IncentiveSpec:
    with Path(path).open("rb") as file:
        payload = tomllib.load(file)
    return IncentiveSpec.model_validate(payload)
