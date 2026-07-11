from __future__ import annotations

import json

from icframe.ui.server import JobManager


def test_restart_marks_running_manifests_interrupted(tmp_path) -> None:
    manifest = tmp_path / "runs" / "abandoned" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"run_id": "abandoned", "status": "running"}))
    manager = JobManager(tmp_path, workers=1)
    try:
        assert json.loads(manifest.read_text())["status"] == "interrupted"
    finally:
        manager.close()
