from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from icframe.orchestration.bundles import bundle_sha256
from icframe.orchestration.models import BackendJobRef, BackendJobState, WorkerRequest
from icframe.orchestration.worker import execute_worker_request
from icframe.profiles import ExecutionProfile, LLMProfile


class ExecutionBackend(Protocol):
    name: str

    async def submit(self, request: WorkerRequest, *, attempt: int = 1) -> BackendJobRef:
        ...

    async def inspect(self, job: BackendJobRef) -> BackendJobRef:
        ...

    async def cancel(self, job: BackendJobRef) -> BackendJobRef:
        ...

    async def collect(self, job: BackendJobRef, destination: Path) -> Path:
        ...


class LocalExecutionBackend:
    name = "local"

    def __init__(self, exchange_root: str | Path) -> None:
        self.exchange_root = Path(exchange_root)
        self.exchange_root.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, asyncio.Task[Path]] = {}

    async def submit(self, request: WorkerRequest, *, attempt: int = 1) -> BackendJobRef:
        job_id = f"local_{uuid.uuid4().hex[:12]}"
        output = self.exchange_root / job_id
        self._tasks[job_id] = asyncio.create_task(
            asyncio.to_thread(execute_worker_request, request, output)
        )
        return BackendJobRef(
            id=job_id,
            backend=self.name,
            shard_id=request.shard_id,
            attempt=attempt,
            artifact_uri=str(output / "artifact.tar.gz"),
        )

    async def inspect(self, job: BackendJobRef) -> BackendJobRef:
        task = self._tasks.get(job.id)
        if task is None:
            return job.model_copy(update={"state": BackendJobState.FAILED, "error": "job lost"})
        if task.cancelled():
            return job.model_copy(update={"state": BackendJobState.CANCELLED})
        if not task.done():
            return job.model_copy(update={"state": BackendJobState.RUNNING})
        error = task.exception()
        if error is not None:
            return job.model_copy(
                update={
                    "state": BackendJobState.FAILED,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        return job.model_copy(update={"state": BackendJobState.COMPLETED})

    async def cancel(self, job: BackendJobRef) -> BackendJobRef:
        task = self._tasks.get(job.id)
        if task is not None:
            task.cancel()
        return job.model_copy(update={"state": BackendJobState.CANCELLED})

    async def collect(self, job: BackendJobRef, destination: Path) -> Path:
        task = self._tasks.get(job.id)
        if task is None:
            raise RuntimeError("local execution job is unavailable")
        source = await task
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination


class NebiusJobsBackend:
    name = "nebius_jobs"

    def __init__(
        self,
        profile: ExecutionProfile,
        *,
        llm_profiles: Mapping[str, LLMProfile] | None = None,
    ) -> None:
        if profile.type != "nebius_jobs":
            raise ValueError("NebiusJobsBackend requires a nebius_jobs execution profile")
        self.profile = profile
        self.llm_profiles = dict(llm_profiles or {})
        self._sdk = None
        self._service = None
        self._s3 = None

    async def submit(self, request: WorkerRequest, *, attempt: int = 1) -> BackendJobRef:
        service = self._job_service()
        s3 = self._s3_client()
        prefix = f"icframe/jobs/{request.logical_job_id}/{request.shard_id}/attempt-{attempt}"
        request_key = f"{prefix}/request.json"
        bundle_key = f"{prefix}/output/artifact.tar.gz"
        marker_key = f"{prefix}/output/complete.json"
        await asyncio.to_thread(
            s3.put_object,
            Bucket=self.profile.bucket,
            Key=request_key,
            Body=request.model_dump_json().encode(),
            ContentType="application/json",
        )

        from nebius.api.nebius.ai.v1 import CreateJobRequest, JobSpec
        from nebius.api.nebius.common.v1 import ResourceMetadata

        environment = []
        if request.llm_profile:
            profile = self.llm_profiles[request.llm_profile]
            if not profile.remote_secret:
                raise ValueError("remote LLM profile requires a MysteryBox secret")
            environment.append(
                JobSpec.EnvironmentVariable(
                    name=profile.api_key_env,
                    mysterybox_secret=JobSpec.MysteryBoxSecretRef(
                        secret_id=profile.remote_secret
                    ),
                )
            )
        output_path = f"/exchange/{prefix}/output"
        spec_values = dict(
            image=self.profile.image,
            container_command="icframe",
            args=(
                "worker execute-shard "
                f"--request /exchange/{request_key} --output {output_path}"
            ),
            environment_variables=environment,
            volumes=[
                JobSpec.VolumeMount(
                    source=self.profile.bucket,
                    container_path="/exchange",
                    mode=JobSpec.VolumeMount.Mode.READ_WRITE,
                )
            ],
            platform=self.profile.platform,
            preset=self.profile.preset,
            timeout=_duration(self.profile.timeout),
            restart_attempts=0,
        )
        if self.profile.subnet_id:
            spec_values["subnet_id"] = self.profile.subnet_id
        if self.profile.public_ip:
            spec_values["public_ip"] = True
        spec = JobSpec(**spec_values)
        operation = await service.create(
            CreateJobRequest(
                metadata=ResourceMetadata(
                    parent_id=self.profile.parent_id,
                    name=_job_name(request, attempt),
                ),
                spec=spec,
            )
        )
        await operation.wait()
        return BackendJobRef(
            id=operation.resource_id,
            backend=self.name,
            shard_id=request.shard_id,
            attempt=attempt,
            artifact_uri=f"s3://{self.profile.bucket}/{bundle_key}",
            provider_data={
                "request_key": request_key,
                "bundle_key": bundle_key,
                "marker_key": marker_key,
            },
        )

    async def inspect(self, job: BackendJobRef) -> BackendJobRef:
        from nebius.api.nebius.ai.v1 import GetJobRequest, JobStatus

        resource = await self._job_service().get(GetJobRequest(id=job.id))
        state = resource.status.state
        mapped = {
            JobStatus.State.PROVISIONING: BackendJobState.QUEUED,
            JobStatus.State.STARTING: BackendJobState.QUEUED,
            JobStatus.State.RUNNING: BackendJobState.RUNNING,
            JobStatus.State.COMPLETED: BackendJobState.COMPLETED,
            JobStatus.State.CANCELLING: BackendJobState.RUNNING,
            JobStatus.State.CANCELLED: BackendJobState.CANCELLED,
            JobStatus.State.FAILED: BackendJobState.FAILED,
            JobStatus.State.ERROR: BackendJobState.FAILED,
        }.get(state, BackendJobState.QUEUED)
        details = resource.status.state_details
        error = details.message if mapped is BackendJobState.FAILED else None
        return job.model_copy(update={"state": mapped, "error": error})

    async def cancel(self, job: BackendJobRef) -> BackendJobRef:
        from nebius.api.nebius.ai.v1 import CancelJobRequest

        operation = await self._job_service().cancel(CancelJobRequest(id=job.id))
        await operation.wait()
        return job.model_copy(update={"state": BackendJobState.CANCELLED})

    async def collect(self, job: BackendJobRef, destination: Path) -> Path:
        marker_key = str(job.provider_data["marker_key"])
        bundle_key = str(job.provider_data["bundle_key"])
        s3 = self._s3_client()
        marker = await asyncio.to_thread(
            s3.get_object,
            Bucket=self.profile.bucket,
            Key=marker_key,
        )
        payload = json.loads(marker["Body"].read())
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            s3.download_file,
            self.profile.bucket,
            bundle_key,
            str(destination),
        )
        if destination.stat().st_size != int(payload["size"]):
            raise ValueError("remote artifact bundle size does not match completion marker")
        if bundle_sha256(destination) != payload["sha256"]:
            raise ValueError("remote artifact bundle checksum does not match completion marker")
        return destination

    async def close(self) -> None:
        if self._sdk is not None:
            await self._sdk.close()

    def _job_service(self):
        if self._service is None:
            try:
                from nebius.aio.cli_config import Config
                from nebius.api.nebius.ai.v1 import JobServiceClient
                from nebius.sdk import SDK
            except ImportError as exc:  # pragma: no cover - optional installation
                raise RuntimeError("install icframe[nebius] to use NebiusJobsBackend") from exc
            self._sdk = SDK(config_reader=Config())
            self._service = JobServiceClient(self._sdk)
        return self._service

    def _s3_client(self):
        if self._s3 is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - optional installation
                raise RuntimeError("install icframe[nebius] to use Object Storage") from exc
            session = boto3.Session(profile_name=self.profile.s3_profile)
            self._s3 = session.client("s3", endpoint_url=self.profile.s3_endpoint_url)
        return self._s3


def _duration(value: str) -> timedelta:
    if value.endswith("h"):
        return timedelta(hours=float(value[:-1]))
    if value.endswith("m"):
        return timedelta(minutes=float(value[:-1]))
    if value.endswith("s"):
        return timedelta(seconds=float(value[:-1]))
    raise ValueError("job timeout must end in h, m, or s")


def _job_name(request: WorkerRequest, attempt: int) -> str:
    base = f"icframe-{request.logical_job_id}-{request.shard_id}-a{attempt}"
    return base.replace("_", "-")[:63]
