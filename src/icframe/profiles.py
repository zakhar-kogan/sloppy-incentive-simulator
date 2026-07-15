from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from icframe.core.packs import LoadedDomainPack
from icframe.domain.base import ICFrameModel
from icframe.domain.incentive_spec import IncentiveSpec
from icframe.llm import LiteLLMClient, LLMClient
from icframe.runtime_settings import RuntimeLLMSettings

TOKEN_FACTORY_BASE_URL = "https://api.tokenfactory.nebius.com/v1"


class ExecutionProfile(ICFrameModel):
    type: Literal["local", "nebius_jobs"] = "local"
    workers: int = Field(default=4, ge=1)
    parent_id: str | None = None
    image: str | None = None
    bucket: str | None = None
    subnet_id: str | None = None
    platform: str = "cpu-d3"
    preset: str = "4vcpu-16gb"
    shard_size: int = Field(default=32, ge=1)
    max_in_flight: int = Field(default=4, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    poll_seconds: float = Field(default=10.0, gt=0.0)
    timeout: str = "1h"
    s3_endpoint_url: str | None = None
    s3_profile: str | None = None

    @model_validator(mode="after")
    def remote_requirements(self) -> ExecutionProfile:
        if self.type == "nebius_jobs":
            missing = [
                name
                for name, value in {
                    "parent_id": self.parent_id,
                    "image": self.image,
                    "bucket": self.bucket,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(f"Nebius execution profile is missing: {', '.join(missing)}")
        return self

    def redacted(self) -> dict[str, object]:
        payload = self.model_dump(mode="json")
        payload["status"] = (
            "ready"
            if self.type == "local"
            else "configured"
            if find_spec("nebius") is not None and find_spec("boto3") is not None
            else "install icframe[nebius]"
        )
        payload["capabilities"] = (
            ["run", "study", "cancel"]
            if self.type == "local"
            else ["run", "sharded-study", "retry", "cancel", "resume"]
        )
        return payload


class LLMProfile(ICFrameModel):
    type: Literal["openai_compatible"] = "openai_compatible"
    provider: str = "openai-compatible"
    base_url: str
    api_key_env: str
    remote_secret: str | None = None
    model: str | None = None
    input_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0.0)
    output_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0.0)

    def redacted(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        values = env if env is not None else os.environ
        payload = self.model_dump(mode="json")
        payload["has_api_key"] = bool(values.get(self.api_key_env))
        payload["has_remote_secret"] = bool(payload.pop("remote_secret", None))
        payload["status"] = "ready" if payload["has_api_key"] else "missing local key"
        payload["model_discovery"] = True
        return payload


class ProfileRegistry(ICFrameModel):
    execution: dict[str, ExecutionProfile] = Field(default_factory=dict)
    llm: dict[str, LLMProfile] = Field(default_factory=dict)

    def execution_profile(self, name: str) -> ExecutionProfile:
        try:
            return self.execution[name]
        except KeyError as exc:
            raise ValueError(f"unknown execution profile: {name}") from exc

    def llm_profile(self, name: str) -> LLMProfile:
        try:
            return self.llm[name]
        except KeyError as exc:
            raise ValueError(f"unknown LLM profile: {name}") from exc

    def public_payload(self, env: Mapping[str, str] | None = None) -> dict[str, object]:
        return {
            "execution": {
                name: profile.redacted() for name, profile in sorted(self.execution.items())
            },
            "llm": {
                name: profile.redacted(env) for name, profile in sorted(self.llm.items())
            },
        }


def load_profiles(
    path: str | Path = "icframe.toml",
    *,
    env: Mapping[str, str] | None = None,
) -> ProfileRegistry:
    values = env if env is not None else os.environ
    config_path = Path(values.get("ICFRAME_CONFIG", str(path)))
    payload: dict[str, object] = {}
    if config_path.exists():
        with config_path.open("rb") as file:
            payload = tomllib.load(file)
    execution = dict(payload.get("execution", {})) if isinstance(payload, dict) else {}
    llm = dict(payload.get("llm", {})) if isinstance(payload, dict) else {}
    execution.setdefault("local", {"type": "local", "workers": 4})
    llm.setdefault(
        "nebius-token-factory",
        {
            "type": "openai_compatible",
            "provider": "nebius-token-factory",
            "base_url": TOKEN_FACTORY_BASE_URL,
            "api_key_env": "NEBIUS_API_KEY",
        },
    )
    return ProfileRegistry.model_validate({"execution": execution, "llm": llm})


def llm_client_for_profile(
    profile: LLMProfile,
    *,
    env: Mapping[str, str] | None = None,
) -> LLMClient:
    values = env if env is not None else os.environ
    api_key = values.get(profile.api_key_env)
    if not api_key:
        raise ValueError(f"LLM API key environment variable {profile.api_key_env} is not set")
    return LiteLLMClient(
        RuntimeLLMSettings(
            base_url=profile.base_url,
            api_key=api_key,
            model=profile.model,
            api_key_source=f"environment:{profile.api_key_env}",
        )
    )


def apply_llm_profile(pack: LoadedDomainPack, profile: LLMProfile) -> LoadedDomainPack:
    """Apply provider identity, selected model, and optional prices to LLM policies."""
    payload = pack.spec.model_dump(mode="python", by_alias=True)
    changed = False
    for archetype in payload["archetypes"].values():
        llm = archetype.get("llm")
        if llm is None:
            continue
        llm["provider"] = profile.provider
        if profile.model:
            llm["model"] = profile.model
        if (
            profile.input_cost_per_million_tokens_usd is not None
            and profile.output_cost_per_million_tokens_usd is not None
        ):
            llm["input_cost_per_million_tokens_usd"] = (
                profile.input_cost_per_million_tokens_usd
            )
            llm["output_cost_per_million_tokens_usd"] = (
                profile.output_cost_per_million_tokens_usd
            )
        changed = True
    return replace(pack, spec=IncentiveSpec.model_validate(payload)) if changed else pack
