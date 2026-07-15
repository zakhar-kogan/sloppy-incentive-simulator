from __future__ import annotations

import os
import subprocess
import time

import pytest

from icframe import RunConfig, RunStatus, cancel_job, get_job, load_profiles, submit_run

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("NEBIUS_LIVE_TEST") != "1",
        reason="set NEBIUS_LIVE_TEST=1 with configured Nebius profiles to incur cloud work",
    ),
]


def _wait(job_id: str, artifact_root, timeout: float = 3600.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle = get_job(job_id, artifact_root=artifact_root)
        assert handle is not None
        if handle.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
            return handle
        time.sleep(10)
    raise AssertionError(f"live Nebius job {job_id} timed out")


def test_live_nebius_worker_token_factory_import_and_cancel(tmp_path) -> None:
    subprocess.run(
        ["docker", "build", "-t", "icframe-worker-live:0.5.0", "."],
        check=True,
    )
    profiles = load_profiles()
    profiles.execution_profile("nebius")
    profiles.llm_profile("nebius-token-factory")

    completed = submit_run(
        "software_organization",
        RunConfig(
            run_id="live-nebius-token-factory",
            seed=19,
            parameters={"steps": 20},
            artifact_root=tmp_path,
        ),
        backend_profile="nebius",
        llm_profile="nebius-token-factory",
    )
    completed = _wait(completed.id, tmp_path)
    assert completed.status is RunStatus.COMPLETED
    assert completed.remote_job_ids
    assert (tmp_path / "runs" / completed.id / "summary.json").exists()

    cancelled = submit_run(
        "software_organization",
        RunConfig(run_id="live-nebius-cancel", seed=23, artifact_root=tmp_path),
        backend_profile="nebius",
        llm_profile="nebius-token-factory",
    )
    cancel_job(cancelled.id, artifact_root=tmp_path)
    cancelled = _wait(cancelled.id, tmp_path)
    assert cancelled.status is RunStatus.CANCELLED
