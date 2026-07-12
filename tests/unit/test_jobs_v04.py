from __future__ import annotations

import json

from icframe.domain.run import RunStatus
from icframe.ui.server import Job, JobManager


def test_restart_marks_running_manifests_interrupted(tmp_path) -> None:
    manifest = tmp_path / "runs" / "abandoned" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"run_id": "abandoned", "status": "running"}))
    manager = JobManager(tmp_path, workers=1)
    try:
        assert json.loads(manifest.read_text())["status"] == "interrupted"
    finally:
        manager.close()


def test_completed_job_state_is_bounded(tmp_path) -> None:
    manager = JobManager(tmp_path, workers=1, max_completed_jobs=2)
    try:
        active = Job("active", "run", RunStatus.RUNNING, {"pack": "public_goods"})
        completed = [
            Job(f"done-{index}", "run", RunStatus.RUNNING, {"pack": "public_goods"})
            for index in range(3)
        ]
        manager.jobs = {item.id: item for item in [active, *completed]}
        for item in completed:
            manager._set(item, RunStatus.COMPLETED)

        assert manager.get("active") is active
        assert len(manager.jobs) == 3
        assert manager.get("done-0") is None
    finally:
        manager.close()
