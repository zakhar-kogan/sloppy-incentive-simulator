from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import PolicyKind, RetentionProfile
from icframe.domain.run import ParameterRange, PlannerKind, StudyMode

MAX_SEED_BATCH = 100


class APIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaginationQuery(APIRequest):
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class LLMRequest(APIRequest):
    llm_mode: Literal["none", "live"] = "none"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    llm_system_prompt: str | None = None


class PopulationLLMOverride(APIRequest):
    provider: str = "litellm"
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    system_prompt: str = ""
    action_field: str = "action"
    target_field: str = "target_id"
    require_json: bool = True
    input_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0.0)
    output_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0.0)


class PopulationOverride(APIRequest):
    archetype_id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    count: int = Field(ge=1, le=10_000)
    policy: PolicyKind
    role: str = Field(default="agent", min_length=1)
    visibility_profile: str = Field(min_length=1)
    scalarizer: dict[str, float] = Field(default_factory=dict)
    policy_config: dict[str, Any] = Field(default_factory=dict)
    initial_state: str | None = None
    initial_resources: dict[str, float] = Field(default_factory=dict)
    attributes: dict[str, Scalar] = Field(default_factory=dict)
    llm: PopulationLLMOverride | None = None

    @model_validator(mode="after")
    def policy_contract(self) -> PopulationOverride:
        if self.policy is PolicyKind.EXTERNAL:
            raise ValueError("external policies cannot run in UI-managed experiments")
        if (self.policy is PolicyKind.LLM) != (self.llm is not None):
            raise ValueError("llm configuration is required only for llm_policy")
        return self


class ExperimentRequest(LLMRequest):
    execution_profile: str = Field(default="local", min_length=1)
    llm_profile: str | None = Field(default=None, min_length=1)
    population_overrides: list[PopulationOverride] | None = Field(
        default=None, min_length=1, max_length=100
    )

    @model_validator(mode="after")
    def unique_population_archetypes(self) -> ExperimentRequest:
        if self.population_overrides is not None:
            ids = [item.archetype_id for item in self.population_overrides]
            if len(ids) != len(set(ids)):
                raise ValueError("population override archetype ids must be unique")
        return self


class RunRequest(ExperimentRequest):
    pack: str = Field(min_length=1)
    seed: int | None = None
    seeds: list[int] | None = Field(default=None, min_length=1, max_length=MAX_SEED_BATCH)
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    retention: RetentionProfile = RetentionProfile.EXPERIMENT
    sample_every_steps: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def unique_seeds(self) -> RunRequest:
        if self.seeds is not None and len(self.seeds) != len(set(self.seeds)):
            raise ValueError("run seeds contain duplicates")
        return self


class StudyRequest(ExperimentRequest):
    pack: str = Field(min_length=1)
    mode: StudyMode = StudyMode.SINGLE
    objectives: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    parameter_ranges: dict[str, ParameterRange] = Field(default_factory=dict)
    trials: int = Field(default=20, ge=1, le=10_000)
    seeds: list[int] | None = Field(default=None, min_length=1, max_length=MAX_SEED_BATCH)
    workers: int | None = Field(default=None, ge=1, le=32)
    allow_live_llm: bool = False
    max_llm_calls: int | None = Field(default=None, ge=1)
    max_llm_cost_usd: float | None = Field(default=None, gt=0)
    composition_parameter_ranges: dict[str, ParameterRange] = Field(default_factory=dict)
    planner: PlannerKind = PlannerKind.RANDOM
    planner_seed: int = 0
    parameter_matrix: dict[str, list[Scalar]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_study_request(self) -> StudyRequest:
        if self.seeds is not None and len(self.seeds) != len(set(self.seeds)):
            raise ValueError("study seeds contain duplicates")
        if self.allow_live_llm and self.workers not in {None, 1}:
            raise ValueError("live LLM studies require workers=1")
        if self.planner is PlannerKind.MATRIX and not self.parameter_matrix:
            raise ValueError("matrix studies require parameter_matrix")
        if self.planner is not PlannerKind.MATRIX and self.parameter_matrix:
            raise ValueError("parameter_matrix requires planner=matrix")
        return self


class ModelsRequest(APIRequest):
    base_url: str | None = None
    api_key: str | None = None
    llm_profile: str | None = None


class TrialRerunRequest(APIRequest):
    seeds: list[int] | None = Field(default=None, min_length=1, max_length=MAX_SEED_BATCH)
    retention: RetentionProfile = RetentionProfile.EXPERIMENT


def validated_payload(model: type[APIRequest], payload: dict[str, Any]) -> dict[str, Any]:
    return model.model_validate(payload).model_dump(mode="python", exclude_none=True)
