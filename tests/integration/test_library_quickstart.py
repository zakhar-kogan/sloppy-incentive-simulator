from __future__ import annotations

import runpy
from pathlib import Path


def test_notebook_quickstart_uses_public_api(tmp_path) -> None:
    notebook = Path(__file__).parents[2] / "notebooks" / "library_quickstart.py"
    namespace = runpy.run_path(str(notebook))
    summary = namespace["run_demo"](tmp_path)
    assert summary.pack_id == "software_organization"
    assert summary.steps_completed == 12
