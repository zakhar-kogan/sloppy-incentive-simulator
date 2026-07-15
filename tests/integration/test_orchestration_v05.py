from __future__ import annotations

import time
from pathlib import Path

from icframe.catalog import Catalog
from icframe.domain.run import PlannerKind, RunConfig, RunStatus, StudyConfig, StudyMode
from icframe.orchestration.models import BackendJobRef, BackendJobState
from icframe.orchestration.service import JobCoordinator
from icframe.orchestration.worker import execute_worker_request
from icframe.profiles import ExecutionProfile, ProfileRegistry, load_profiles


def _wait(coordinator: JobCoordinator, job_id: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle = coordinator.get_job(job_id)
        assert handle is not None
        if handle.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
            return handle
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_local_async_run_and_matrix_study_import_normal_artifacts(tmp_path) -> None:
    coordinator = JobCoordinator(tmp_path)
    try:
        run = coordinator.submit_run(
            "software_organization",
            RunConfig(run_id="async-run", seed=0, artifact_root=tmp_path),
        )
        run_handle = _wait(coordinator, run.id)
        assert run_handle.status is RunStatus.COMPLETED
        assert (tmp_path / "runs" / run.id / "summary.json").exists()

        study = coordinator.submit_study(
            "software_organization",
            StudyConfig(
                study_id="async-study",
                mode=StudyMode.SINGLE,
                objectives=["trusted_score"],
                parameters=["proxy_agents", "audit_probability"],
                seeds=[0],
                artifact_root=tmp_path,
                planner=PlannerKind.MATRIX,
                parameter_matrix={
                    "proxy_agents": [1, 4],
                    "audit_probability": [0.0, 0.6],
                },
            ),
        )
        study_handle = _wait(coordinator, study.id)
        assert study_handle.status is RunStatus.COMPLETED
        assert study_handle.completed_trials == 4
        assert (tmp_path / "studies" / study.id / "plan.json").exists()
        assert (tmp_path / "studies" / study.id / "trials.jsonl").read_text().count("\n") == 4
        assert (tmp_path / "runs" / f"{study.id}-baseline-000" / "summary.json").exists()
    finally:
        coordinator.close()


class _OutOfOrderBackend:
    name = "fake-nebius"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests = {}
        self.polls = {}
        self.cancelled = []

    async def submit(self, request, *, attempt: int = 1):
        job_id = f"{request.shard_id}-attempt-{attempt}"
        self.requests[job_id] = request
        return BackendJobRef(
            id=job_id,
            backend=self.name,
            shard_id=request.shard_id,
            attempt=attempt,
        )

    async def inspect(self, job):
        self.polls[job.id] = self.polls.get(job.id, 0) + 1
        if job.shard_id.endswith("0001") and job.attempt == 1:
            return job.model_copy(
                update={
                    "state": BackendJobState.FAILED,
                    "error": "transient transport failure",
                }
            )
        shard_number = int(job.shard_id.rsplit("-", 1)[-1]) if "shard" in job.shard_id else 0
        threshold = max(1, 4 - shard_number)
        state = (
            BackendJobState.COMPLETED
            if self.polls[job.id] >= threshold
            else BackendJobState.RUNNING
        )
        return job.model_copy(update={"state": state})

    async def cancel(self, job):
        self.cancelled.append(job.id)
        return job.model_copy(update={"state": BackendJobState.CANCELLED})

    async def collect(self, job, destination):
        bundle = execute_worker_request(self.requests[job.id], self.root / job.id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bundle.read_bytes())
        return destination


def test_fake_remote_matches_local_despite_retry_and_out_of_order_shards(tmp_path) -> None:
    local_root = tmp_path / "local"
    remote_root = tmp_path / "remote"
    config = StudyConfig(
        study_id="equivalent-local",
        mode=StudyMode.SINGLE,
        objectives=["trusted_score"],
        parameters=["proxy_agents", "audit_probability"],
        seeds=[0],
        artifact_root=local_root,
        planner=PlannerKind.MATRIX,
        parameter_matrix={
            "proxy_agents": [1, 4],
            "audit_probability": [0.0, 0.6],
        },
    )
    from icframe import run_study

    local = run_study("software_organization", config)
    profiles = load_profiles()
    registry = ProfileRegistry(
        execution={
            **profiles.execution,
            "fake": ExecutionProfile(
                type="local",
                shard_size=1,
                max_in_flight=4,
                max_attempts=3,
                poll_seconds=0.01,
            ),
        },
        llm=profiles.llm,
    )
    coordinator = JobCoordinator(remote_root, profiles=registry)
    fake = _OutOfOrderBackend(tmp_path / "exchange")
    coordinator._backends["fake"] = fake
    try:
        handle = coordinator.submit_study(
            "software_organization",
            config.model_copy(
                update={"study_id": "equivalent-remote", "artifact_root": remote_root}
            ),
            backend_profile="fake",
        )
        completed = _wait(coordinator, handle.id)
        assert completed.status is RunStatus.COMPLETED
        assert completed.retry_count == 1
        remote = Catalog(remote_root).get_study(handle.id)
        assert remote is not None
        assert remote.best_trial == local.best_trial
        assert remote.pareto_trials == local.pareto_trials
        assert remote.trials == local.trials
    finally:
        coordinator.close()
