from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import RetentionProfile
from icframe.domain.run import ParameterRange, StudyMode

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


class RunRequest(LLMRequest):
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


class StudyRequest(LLMRequest):
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

    @model_validator(mode="after")
    def valid_study_request(self) -> StudyRequest:
        if self.seeds is not None and len(self.seeds) != len(set(self.seeds)):
            raise ValueError("study seeds contain duplicates")
        if self.allow_live_llm and self.workers not in {None, 1}:
            raise ValueError("live LLM studies require workers=1")
        return self


class ModelsRequest(APIRequest):
    base_url: str | None = None
    api_key: str | None = None


class TrialRerunRequest(APIRequest):
    seeds: list[int] | None = Field(default=None, min_length=1, max_length=MAX_SEED_BATCH)
    retention: RetentionProfile = RetentionProfile.EXPERIMENT


def validated_payload(model: type[APIRequest], payload: dict[str, Any]) -> dict[str, Any]:
    return model.model_validate(payload).model_dump(mode="python", exclude_none=True)
