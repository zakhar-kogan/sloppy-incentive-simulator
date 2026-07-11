from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import ICFrameModel, Scalar

OutcomeVector = dict[str, float]


class SpecModel(ICFrameModel):
    """Strict TOML boundary model with conventional enum coercion."""

    model_config = ConfigDict(strict=False, extra="forbid", validate_assignment=True)


class ScheduleMode(StrEnum):
    SEQUENTIAL_FIXED = "sequential_fixed"
    SEQUENTIAL_RANDOM = "sequential_random"
    PARALLEL_SIMULTANEOUS = "parallel_simultaneous"


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


class PolicyKind(StrEnum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC_WEIGHTED = "stochastic_weighted"
    EPSILON_GREEDY = "epsilon_greedy_bandit"
    UCB = "ucb_bandit"
    GAUSSIAN_THOMPSON = "gaussian_thompson_bandit"
    CONTEXTUAL = "contextual_bandit"
    Q_LEARNING = "q_learning_simple"
    LLM = "llm_policy"
    EXTERNAL = "external"


class GraphVisibility(StrEnum):
    FULL = "full_graph"
    LOCAL = "local_graph"
    PROMPT_ONLY = "prompt_only"
    NONE = "none"


class OutcomeVisibility(StrEnum):
    FULL_NUMERIC = "full_numeric"
    OWN_SCALAR = "own_scalar"
    LABEL_ONLY = "label_only"
    HIDDEN = "hidden"


class Operation(StrEnum):
    ADD = "add"
    MULTIPLY = "multiply"
    SET = "set"


class EffectScope(StrEnum):
    ACTOR = "actor"
    TARGET = "target"
    POPULATION = "population"
    ALL_AGENTS = "all_agents"
    GLOBAL = "global"


class MetricType(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    EVENT_COUNT = "event_count"
    EVENT_RATE = "event_rate"
    DIFFERENCE = "difference"
    RATIO = "ratio"
    WEIGHTED_SUM = "weighted_sum"


class MetricScope(StrEnum):
    GLOBAL = "global"
    AGENTS = "agents"
    ALL = "all"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SeedReducer(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    WORST = "worst"
    QUANTILE = "quantile"


class ConstraintOperator(StrEnum):
    LE = "le"
    GE = "ge"


class RetentionProfile(StrEnum):
    AUDIT = "audit"
    EXPERIMENT = "experiment"
    TRAINING = "training"


class PromptCapture(StrEnum):
    NONE = "none"
    HASH = "hash"
    FULL = "full"


class ResponseCapture(StrEnum):
    PARSED = "parsed"
    PARSED_AND_HASH = "parsed_and_hash"
    FULL = "full"


class ParameterType(StrEnum):
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CHOICE = "choice"


class ParameterEntity(StrEnum):
    EXPERIMENT = "experiment"
    ARCHETYPE = "archetype"
    POPULATION = "population"
    TRANSITION = "transition"
    HOOK_CONFIG = "hook_config"


class SpecHeader(SpecModel):
    version: Literal["0.4"]
    name: str = Field(min_length=1)
    domain: str = "generic"


class ExperimentConfig(SpecModel):
    steps: int = Field(default=1, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    schedule: ScheduleMode = ScheduleMode.SEQUENTIAL_FIXED

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("experiment.seeds contains duplicates")
        return value


class StateSpace(SpecModel):
    initial: str
    all: list[str] = Field(min_length=1)
    global_values: dict[str, Scalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def declared_initial_state(self) -> StateSpace:
        if self.initial not in self.all:
            raise ValueError("states.initial must be declared in states.all")
        if len(self.all) != len(set(self.all)):
            raise ValueError("states.all contains duplicates")
        return self


class ActionSpace(SpecModel):
    all: list[str] = Field(min_length=1)

    @field_validator("all")
    @classmethod
    def unique_actions(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("actions.all contains duplicates")
        return value


class OutcomeSpace(SpecModel):
    channels: list[str] = Field(min_length=1)

    @field_validator("channels")
    @classmethod
    def unique_channels(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("outcome_space.channels contains duplicates")
        return value


class VisibilityProfile(SpecModel):
    graph: GraphVisibility = GraphVisibility.PROMPT_ONLY
    outcomes: OutcomeVisibility = OutcomeVisibility.HIDDEN
    sanctions: OutcomeVisibility = OutcomeVisibility.HIDDEN
    prompts: bool = True
    history_events: int = Field(default=0, ge=0, le=1000)


class LLMConfig(SpecModel):
    provider: str = "litellm"
    model: str
    temperature: float = Field(default=0.0, ge=0.0)
    system_prompt: str = ""
    action_field: str = "action"
    target_field: str = "target_id"
    require_json: bool = True


class Archetype(SpecModel):
    policy: PolicyKind
    role: str
    visibility_profile: str
    scalarizer: dict[str, float] = Field(default_factory=dict)
    policy_config: dict[str, Any] = Field(default_factory=dict)
    initial_state: str | None = None
    initial_resources: dict[str, float] = Field(default_factory=dict)
    attributes: dict[str, Scalar] = Field(default_factory=dict)
    llm: LLMConfig | None = None

    @model_validator(mode="after")
    def llm_policy_has_config(self) -> Archetype:
        if self.policy is PolicyKind.LLM and self.llm is None:
            raise ValueError("llm_policy requires archetype.llm")
        if self.policy is not PolicyKind.LLM and self.llm is not None:
            raise ValueError("archetype.llm is only valid for llm_policy")
        return self


class PopulationEntry(SpecModel):
    archetype: str
    count: int = Field(ge=1)


class ScopedEffect(SpecModel):
    scope: EffectScope = EffectScope.ACTOR
    population: str | None = None
    operation: Operation = Operation.ADD
    values: OutcomeVector = Field(default_factory=dict)

    @model_validator(mode="after")
    def population_scope_has_name(self) -> ScopedEffect:
        if (self.scope is EffectScope.POPULATION) != (self.population is not None):
            raise ValueError(
                "population effects require only scope=population and population=<name>"
            )
        return self


class StateUpdate(SpecModel):
    scope: EffectScope
    field: list[str] = Field(min_length=1)
    operation: Operation = Operation.ADD
    value: Scalar
    population: str | None = None

    @model_validator(mode="after")
    def validate_update(self) -> StateUpdate:
        if self.scope is EffectScope.GLOBAL and self.population is not None:
            raise ValueError("global state updates cannot name a population")
        if (self.scope is EffectScope.POPULATION) != (self.population is not None):
            raise ValueError(
                "population updates require only scope=population and population=<name>"
            )
        if self.operation is not Operation.SET and not isinstance(self.value, int | float):
            raise ValueError("add and multiply state updates require numeric values")
        return self


class Enforcement(SpecModel):
    audit_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    detection_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    false_positive_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    false_negative_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    enforcement_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    sanctions: list[ScopedEffect] = Field(default_factory=list)
    compliance_rewards: list[ScopedEffect] = Field(default_factory=list)
    remediation_actions: list[str] = Field(default_factory=list)


class PromptDescription(SpecModel):
    label: str | None = None
    description: str | None = None


class Transition(SpecModel):
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
    requires_target: bool = False
    target_populations: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    effects: list[ScopedEffect] = Field(default_factory=list)
    state_updates: list[StateUpdate] = Field(default_factory=list)
    enforcement: Enforcement | None = None
    prompt: PromptDescription | None = None

    @model_validator(mode="after")
    def target_contract(self) -> Transition:
        if self.target_populations and not self.requires_target:
            raise ValueError("target_populations requires requires_target=true")
        return self


class MetricSpec(SpecModel):
    type: MetricType
    channel: str | None = None
    scope: MetricScope = MetricScope.ALL
    where_tags_include: list[str] = Field(default_factory=list)
    left: str | None = None
    right: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    terms: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def required_fields(self) -> MetricSpec:
        if self.type in {MetricType.SUM, MetricType.MEAN} and self.channel is None:
            raise ValueError(f"{self.type.value} metrics require channel")
        if self.type is MetricType.DIFFERENCE and (self.left is None or self.right is None):
            raise ValueError("difference metrics require left and right")
        if self.type is MetricType.RATIO and (self.numerator is None or self.denominator is None):
            raise ValueError("ratio metrics require numerator and denominator")
        if self.type is MetricType.WEIGHTED_SUM and not self.terms:
            raise ValueError("weighted_sum metrics require terms")
        return self


class ObjectiveSpec(SpecModel):
    metric: str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    seed_reducer: SeedReducer = SeedReducer.MEAN
    quantile: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def quantile_contract(self) -> ObjectiveSpec:
        if self.seed_reducer is SeedReducer.QUANTILE and self.quantile is None:
            raise ValueError("quantile reducers require quantile")
        if self.seed_reducer is not SeedReducer.QUANTILE and self.quantile is not None:
            raise ValueError("quantile is only valid with seed_reducer=quantile")
        return self


class TrustedConstraint(SpecModel):
    metric: str
    operator: ConstraintOperator
    threshold: float
    require_all_seeds: bool = True
    seed_reducer: SeedReducer = SeedReducer.MEAN
    quantile: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def quantile_contract(self) -> TrustedConstraint:
        if self.seed_reducer is SeedReducer.QUANTILE and self.quantile is None:
            raise ValueError("quantile reducers require quantile")
        if self.seed_reducer is not SeedReducer.QUANTILE and self.quantile is not None:
            raise ValueError("quantile is only valid with seed_reducer=quantile")
        return self


class EvaluationConfig(SpecModel):
    objectives: dict[str, ObjectiveSpec] = Field(default_factory=dict)
    constraints: list[TrustedConstraint] = Field(default_factory=list)


class RedactionConfig(SpecModel):
    prompt_capture: PromptCapture = PromptCapture.HASH
    response_capture: ResponseCapture = ResponseCapture.PARSED_AND_HASH


class ObservabilityConfig(SpecModel):
    retention: RetentionProfile = RetentionProfile.EXPERIMENT
    sample_every_steps: int | None = Field(default=None, ge=1)
    max_checkpoints: int = Field(default=200, ge=2, le=10_000)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)


class SymbolicConfig(SpecModel):
    enabled: bool = False
    rules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enabled_rules(self) -> SymbolicConfig:
        if self.rules and not self.enabled:
            raise ValueError("symbolic.rules requires symbolic.enabled=true")
        return self


class IncentiveSpec(SpecModel):
    spec: SpecHeader
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    states: StateSpace
    actions: ActionSpace
    outcome_space: OutcomeSpace
    visibility_profiles: dict[str, VisibilityProfile]
    archetypes: dict[str, Archetype]
    population: list[PopulationEntry] = Field(min_length=1)
    transitions: list[Transition] = Field(min_length=1)
    metrics: dict[str, MetricSpec] = Field(default_factory=dict)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    symbolic: SymbolicConfig = Field(default_factory=SymbolicConfig)
    hook_config: dict[str, Scalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_are_valid(self) -> IncentiveSpec:
        states = set(self.states.all)
        actions = set(self.actions.all)
        channels = set(self.outcome_space.channels)
        populations = set(self.archetypes)
        transition_ids: set[str] = set()

        for name, archetype in self.archetypes.items():
            if archetype.visibility_profile not in self.visibility_profiles:
                raise ValueError(f"archetype {name} references unknown visibility profile")
            if archetype.initial_state is not None and archetype.initial_state not in states:
                raise ValueError(f"archetype {name} references unknown initial_state")
            unknown = set(archetype.scalarizer) - channels
            if unknown:
                raise ValueError(f"archetype {name} references unknown channels: {sorted(unknown)}")

        for entry in self.population:
            if entry.archetype not in self.archetypes:
                raise ValueError(f"population references unknown archetype {entry.archetype!r}")

        for transition in self.transitions:
            if transition.id in transition_ids:
                raise ValueError(f"duplicate transition id {transition.id!r}")
            transition_ids.add(transition.id)
            if transition.from_state not in states or transition.to_state not in states:
                raise ValueError(f"transition {transition.id} references an unknown state")
            if transition.action not in actions:
                raise ValueError(f"transition {transition.id} references unknown action")
            if not set(transition.target_populations).issubset(populations):
                raise ValueError(f"transition {transition.id} references unknown target population")
            enforcement = transition.enforcement
            effects = [
                *transition.effects,
                *(enforcement.sanctions if enforcement else []),
                *(enforcement.compliance_rewards if enforcement else []),
            ]
            for effect in effects:
                unknown = set(effect.values) - channels
                if unknown:
                    raise ValueError(
                        f"transition {transition.id} references unknown channels: {sorted(unknown)}"
                    )
                if effect.population is not None and effect.population not in populations:
                    raise ValueError(f"transition {transition.id} references unknown population")
            for update in transition.state_updates:
                if update.population is not None and update.population not in populations:
                    raise ValueError(f"transition {transition.id} references unknown population")
            if transition.enforcement:
                unknown_actions = set(transition.enforcement.remediation_actions) - actions
                if unknown_actions:
                    raise ValueError(
                        f"transition {transition.id} references unknown remediation actions"
                    )

        for metric_name, metric in self.metrics.items():
            if metric.channel is not None and metric.channel not in channels:
                raise ValueError(f"metric {metric_name} references unknown channel")
        metric_names = set(self.metrics)
        for name, objective in self.evaluation.objectives.items():
            if objective.metric not in metric_names:
                raise ValueError(f"objective {name} references unknown metric")
        for constraint in self.evaluation.constraints:
            if constraint.metric not in metric_names:
                raise ValueError("trusted constraint references unknown metric")
        return self


class ParameterTarget(SpecModel):
    entity: ParameterEntity
    entity_id: str | None = None
    field: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def entity_id_contract(self) -> ParameterTarget:
        needs_id = self.entity in {
            ParameterEntity.ARCHETYPE,
            ParameterEntity.POPULATION,
            ParameterEntity.TRANSITION,
        }
        if needs_id != (self.entity_id is not None):
            raise ValueError(f"{self.entity.value} targets require entity_id={needs_id}")
        return self


class GuidedParameter(SpecModel):
    id: str
    label: str
    description: str = ""
    unit: str | None = None
    type: ParameterType
    default: Scalar
    target: ParameterTarget
    optimizable: bool = False
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    slider: bool = True
    choices: list[Scalar] = Field(default_factory=list)

    @model_validator(mode="after")
    def bounds_match_type(self) -> GuidedParameter:
        if self.type in {ParameterType.FLOAT, ParameterType.INTEGER}:
            if self.minimum is None or self.maximum is None:
                raise ValueError("numeric parameters require minimum and maximum")
            if self.minimum > self.maximum:
                raise ValueError("parameter minimum exceeds maximum")
            if self.step is not None and self.step <= 0:
                raise ValueError("numeric parameter step must be positive")
        elif self.minimum is not None or self.maximum is not None or self.step is not None:
            raise ValueError("only numeric parameters may define bounds or step")
        if self.type is ParameterType.CHOICE and not self.choices:
            raise ValueError("choice parameters require choices")
        if self.type is not ParameterType.CHOICE and self.choices:
            raise ValueError("choices are only valid for choice parameters")
        return self


class DomainPackHeader(SpecModel):
    id: str
    title: str
    description: str = ""
    spec_file: str = "spec.toml"
    hook: str | None = None


class StudyDefaults(SpecModel):
    single_objective: str
    pareto_objectives: list[str] = Field(min_length=2)


class ValidationDefaults(SpecModel):
    golden_seeds: list[int] = Field(min_length=1)
    report_metrics: list[str] = Field(min_length=1)


class DomainPackManifest(SpecModel):
    pack: DomainPackHeader
    study: StudyDefaults
    validation: ValidationDefaults
    parameters: list[GuidedParameter] = Field(default_factory=list)

    @field_validator("parameters")
    @classmethod
    def unique_parameter_ids(cls, value: list[GuidedParameter]) -> list[GuidedParameter]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("domain pack parameter ids must be unique")
        return value


def load_incentive_spec(path: str | Path) -> IncentiveSpec:
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        raise ValueError(
            "legacy JSON scenarios are unsupported; migrate the scenario into an "
            "IncentiveSpec v0.4 domain pack"
        )
    with file_path.open("rb") as file:
        payload = tomllib.load(file)
    version = payload.get("spec", {}).get("version")
    if version != "0.4":
        raise ValueError(
            f"{file_path} uses IncentiveSpec {version!r}; only v0.4 is supported. "
            "Migrate the domain pack instead of relying on legacy runtime behavior."
        )
    return IncentiveSpec.model_validate(payload)


def load_domain_pack_manifest(path: str | Path) -> DomainPackManifest:
    with Path(path).open("rb") as file:
        return DomainPackManifest.model_validate(tomllib.load(file))
