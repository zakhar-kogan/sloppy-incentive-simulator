from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from icframe.domain.base import ICFrameModel, Scalar
from icframe.domain.incentive_spec import RetentionProfile
from icframe.domain.run import LiveLLMBudget, RunStatus, RunSummary, TrialRecord
from icframe.planning import TrialSpec


class BackendJobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StudyShardRequest(ICFrameModel):
    schema_version: str = "1"
    kind: Literal["study"] = "study"
    logical_job_id: str
    shard_id: str
    plan_hash: str
    pack_id: str
    pack_hash: str
    runtime_hash: str
    effective_manifest: dict[str, object] | None = None
    effective_spec: dict[str, object] | None = None
    trials: list[TrialSpec] = Field(min_length=1)
    llm_profile: str | None = None
    llm_config: dict[str, object] | None = None
    live_llm: LiveLLMBudget = Field(default_factory=LiveLLMBudget)


class RunShardRequest(ICFrameModel):
    schema_version: str = "1"
    kind: Literal["run"] = "run"
    logical_job_id: str
    shard_id: str
    pack_id: str
    pack_hash: str
    runtime_hash: str
    effective_manifest: dict[str, object] | None = None
    effective_spec: dict[str, object] | None = None
    seed: int | None = None
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    retention: RetentionProfile = RetentionProfile.EXPERIMENT
    sample_every_steps: int | None = Field(default=None, ge=1)
    llm_profile: str | None = None
    llm_config: dict[str, object] | None = None
    live_llm: LiveLLMBudget = Field(default_factory=LiveLLMBudget)


class StudyShardResult(ICFrameModel):
    schema_version: str = "1"
    logical_job_id: str
    shard_id: str
    plan_hash: str
    status: RunStatus
    trials: list[TrialRecord] = Field(default_factory=list)
    error: str | None = None


class RunShardResult(ICFrameModel):
    schema_version: str = "1"
    logical_job_id: str
    shard_id: str
    status: RunStatus
    summary: RunSummary


class BackendJobRef(ICFrameModel):
    id: str
    backend: str
    shard_id: str
    attempt: int = Field(default=1, ge=1)
    state: BackendJobState = BackendJobState.QUEUED
    artifact_uri: str | None = None
    error: str | None = None
    provider_data: dict[str, object] = Field(default_factory=dict)


class JobHandle(ICFrameModel):
    id: str
    kind: Literal["run", "study"]
    backend_profile: str
    status: RunStatus = RunStatus.QUEUED
    artifact_root: str
    planned_trials: int = 0
    completed_trials: int = 0
    shard_count: int = 0
    remote_job_ids: list[str] = Field(default_factory=list)
    retry_count: int = 0
    artifact_import_state: str = "pending"
    cancel_requested: bool = False
    error: str | None = None


class BundleFile(ICFrameModel):
    path: str
    size: int = Field(ge=0)
    sha256: str


class ArtifactBundleManifest(ICFrameModel):
    schema_version: str = "1"
    logical_id: str
    files: list[BundleFile] = Field(default_factory=list)


WorkerRequest = RunShardRequest | StudyShardRequest
