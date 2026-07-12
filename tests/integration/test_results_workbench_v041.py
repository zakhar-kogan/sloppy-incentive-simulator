from __future__ import annotations

import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from pydantic import ValidationError

from icframe import RunConfig, load_domain_pack, run_experiment
from icframe.domain.incentive_spec import DomainPackManifest, RetentionProfile
from icframe.domain.run import LiveLLMBudget, StudyConfig, StudyMode
from icframe.llm import LLMResponse
from icframe.reports.view_models import run_view_model
from icframe.study import run_study
from icframe.ui.server import create_server


def _request(url: str):
    with urlopen(Request(url), timeout=10) as response:
        return response.status, json.loads(response.read())


def test_report_contracts_aggregates_and_mechanics_are_reproducible(tmp_path) -> None:
    pack = load_domain_pack("public_goods")
    summary = run_experiment(
        pack,
        RunConfig(
            seed=7,
            parameters={"steps": 4},
            retention=RetentionProfile.AUDIT,
            artifact_root=tmp_path,
        ),
    )
    run_dir = tmp_path / "runs" / summary.run_id
    persisted_manifest = DomainPackManifest.model_validate_json(
        (run_dir / "domain_pack_manifest.json").read_text()
    )
    persisted_spec = pack.spec.model_validate_json((run_dir / "spec.json").read_text())
    view = run_view_model(summary, persisted_manifest, persisted_spec)

    assert sum(summary.transition_counts.values()) == summary.event_count
    assert sum(sum(agent.statistics.action_counts.values()) for agent in summary.agents) == (
        summary.event_count
    )
    assert view.metrics[0].label == "Trusted score"
    assert view.metrics[0].formula
    assert any(item.kind == "behavior" for item in view.findings)
    assert {item.id for item in view.mechanics.transitions} == {
        item.id for item in pack.spec.transitions
    }
    assert sum(item.frequency for item in view.mechanics.transitions) == summary.event_count
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    for agent in summary.agents:
        actor_events = [item for item in events if item["actor_id"] == agent.id]
        assert sum(agent.statistics.action_counts.values()) == len(actor_events)
        assert agent.statistics.enforcement == sum(item["enforced"] for item in actor_events)
        assert agent.statistics.violations == sum(len(item["violations"]) for item in actor_events)


def test_llm_summary_matches_redacted_calls_and_unknown_cost(
    tmp_path, deterministic_llm_client
) -> None:
    summary = run_experiment(
        "software_organization",
        RunConfig(seed=19, parameters={"steps": 3}, artifact_root=tmp_path),
        llm_client=deterministic_llm_client("refactor_core", estimated_cost=None),
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "runs" / summary.run_id / "llm_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert summary.llm_usage.attempted == len(records)
    assert summary.llm_usage.total_tokens == sum(item["total_tokens"] for item in records)
    assert summary.llm_usage.estimated_cost_usd is None
    assert len(summary.llm_usage.latency_buckets) == 8
    assert all({"step", "agent_id", "status", "latency_ms"} <= set(item) for item in records)


def test_llm_calls_endpoint_supports_completed_runs_and_caps_pages(
    tmp_path, deterministic_llm_client
) -> None:
    summary = run_experiment(
        "software_organization",
        RunConfig(seed=19, parameters={"steps": 3}, artifact_root=tmp_path),
        llm_client=deterministic_llm_client("refactor_core"),
    )
    server = create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        _, first = _request(f"{base}/api/runs/{summary.run_id}/llm-calls?limit=1")
        assert first["total"] == summary.llm_usage.attempted
        assert len(first["calls"]) == 1
        with pytest.raises(HTTPError) as error:
            _request(f"{base}/api/runs/{summary.run_id}/llm-calls?limit=101")
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_job_polling_exposes_bounded_live_progress(tmp_path) -> None:
    server = create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(
            f"{base}/api/runs",
            data=json.dumps(
                {"pack": "public_goods", "seeds": [7], "parameters": {"steps": 50}}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=10) as response:
            job_id = json.loads(response.read())["jobs"][0]["id"]
        job = None
        for _ in range(200):
            _, payload = _request(f"{base}/api/jobs/{job_id}")
            job = payload["job"]
            if job["status"] == "completed":
                break
            time.sleep(0.01)
        assert job is not None
        assert job["progress"]["steps_planned"] == 50
        assert job["progress"]["steps_completed"] == 50
        assert len(job["progress"]["metrics"]) < 100
    finally:
        server.shutdown()
        server.server_close()


def test_report_metadata_rejects_unknown_chart_metrics() -> None:
    payload = load_domain_pack("public_goods").manifest.model_dump(mode="python")
    payload["report"]["chart_groups"][0]["metrics"].append("not_a_metric")
    with pytest.raises(ValidationError, match="without metadata"):
        DomainPackManifest.model_validate(payload)


def test_cost_bounded_study_rejects_unknown_pricing(tmp_path) -> None:
    class UnknownPriceClient:
        def complete(self, request):
            return LLMResponse(
                content='{"action":"refactor_core"}',
                parsed={"action": "refactor_core"},
                provider=request.provider,
                model=request.model,
            )

    with pytest.raises(RuntimeError, match="cost is unavailable"):
        run_study(
            "software_organization",
            StudyConfig(
                mode=StudyMode.SINGLE,
                objectives=["trusted_score"],
                parameters=["audit_probability"],
                trials=1,
                seeds=[19],
                workers=1,
                artifact_root=tmp_path,
                live_llm=LiveLLMBudget(
                    enabled=True,
                    max_calls=10,
                    max_cost_usd=1.0,
                ),
            ),
            llm_client=UnknownPriceClient(),
        )
