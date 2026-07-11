from __future__ import annotations

import json

from icframe import RunConfig, run_experiment
from icframe.reports import render_html_report, write_html_report
from icframe.reports.view_models import run_view_model


def test_ui_and_static_report_use_the_same_projection(tmp_path) -> None:
    summary = run_experiment(
        "public_goods",
        RunConfig(seed=7, parameters={"steps": 2}, artifact_root=tmp_path),
    )
    view = run_view_model(summary)
    html = render_html_report(summary)
    payload = html.split('<script id="view-model" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]
    assert json.loads(payload) == view.model_dump(mode="json")
    assert "Metrics over time" in html
    assert "Projection" not in html
    path = write_html_report(tmp_path / "runs" / summary.run_id)
    assert path.exists()
    assert "ICFRAME report" in path.read_text()
