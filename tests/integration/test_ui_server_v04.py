from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import icframe.ui.server as ui_server
from icframe.domain.run import RunStatus, StudyMode, StudySummary, TrialRecord
from icframe.ui.server import create_server


def _request(url: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return response.status, response.headers.get_content_type(), response.read()


def _error_request(url: str, payload=None):
    try:
        _request(url, payload)
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())
    raise AssertionError("request unexpectedly succeeded")


def test_catalog_backed_ui_run_flow(tmp_path) -> None:
    server = create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, content_type, body = _request(base + "/")
        assert (status, content_type) == (200, "text/html")
        assert b"Domain pack" in body
        assert b"Fake action" not in body
        packs = json.loads(_request(base + "/api/packs")[2])["packs"]
        runtime = json.loads(_request(base + "/api/runtime")[2])["runtime"]
        assert runtime["ui_api_version"] == "1"
        assert {"quick_values", "population_overrides", "runtime_handshake"} <= set(
            runtime["capabilities"]
        )
        assert {pack["id"] for pack in packs} == {
            "public_goods",
            "software_organization",
            "delayed_reward_learning",
        }
        llm_pack = next(pack for pack in packs if pack["id"] == "software_organization")
        assert llm_pack["llm"]["enabled"] is True
        assert "valid_actions" in llm_pack["llm"]["prompt_preview"]
        assert {item["policy"] for item in llm_pack["policy_templates"]} >= {
            "epsilon_greedy_bandit",
            "q_learning_simple",
            "llm_policy",
        }
        assert llm_pack["composition"]["population"]
        assert all(pack["parameters"][0]["quick_values"] for pack in packs)
        assert all(pack["population_templates"] for pack in packs)
        assert all(pack["mechanics_flow"]["nodes"] for pack in packs)
        assert all(
            template["scalarizer"]
            for pack in packs
            for template in pack["population_templates"]
        )
        response = json.loads(
            _request(
                base + "/api/runs",
                {
                    "pack": "public_goods",
                    "seeds": [7],
                    "parameters": {"steps": 2},
                    "retention": "experiment",
                },
            )[2]
        )
        job_id = response["jobs"][0]["id"]
        pending = json.loads(_request(base + "/api/runs")[2])
        assert any(row["id"] == job_id for row in pending["runs"])
        for _ in range(100):
            job = json.loads(_request(base + f"/api/jobs/{job_id}")[2])["job"]
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert job["status"] == "completed", job
        result = json.loads(_request(base + f"/api/runs/{job_id}")[2])
        assert result["view"]["kind"] == "run"
        assert result["summary"]["steps_completed"] == 2
        history = json.loads(_request(base + "/api/runs")[2])
        stored = next(row for row in history["runs"] if row["id"] == job_id)
        assert "checkpoints" not in stored
        assert _request(base + f"/api/runs/{job_id}/report")[1] == "text/html"
        calls = json.loads(_request(base + f"/api/jobs/{job_id}/llm-calls?limit=10")[2])
        assert calls == {"calls": [], "total": 0, "limit": 10, "offset": 0}
    finally:
        server.shutdown()
        server.server_close()


def test_static_assets_are_snapshotted_at_server_start(tmp_path, monkeypatch) -> None:
    server = create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)
    expected = server.RequestHandlerClass.static_assets["app.js"][0]

    def fail_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"static source was read after startup: {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = _request(
            f"http://127.0.0.1:{server.server_port}/static/app.js"
        )[2]
        assert body == expected
    finally:
        server.shutdown()
        server.server_close()


def test_invalid_pack_prevents_server_binding(tmp_path, monkeypatch) -> None:
    def fail_pack(_pack_id):
        raise ValueError("pack schema is incompatible")

    monkeypatch.setattr(ui_server, "load_domain_pack", fail_pack)
    with pytest.raises(ValueError, match="pack schema is incompatible"):
        create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)


def test_client_has_blocking_runtime_mismatch_recovery() -> None:
    static_root = Path(ui_server.__file__).parent / "static"
    app = (static_root / "app.js").read_text()
    page = (static_root / "index.html").read_text()
    assert 'api("/api/runtime")' in app
    assert "showStartupError" in app
    assert "runtime.ui_api_version" in app
    assert 'id="startupError"' in page
    assert "uv run icframe ui" in page


def test_client_persists_llm_defaults_and_uses_population_templates() -> None:
    app = (Path(ui_server.__file__).parent / "static" / "app.js").read_text()
    assert 'localStorage.setItem(LLM_DEFAULTS_KEY' in app
    assert 'sessionStorage.setItem(LLM_API_KEY' in app
    assert "currentLLMDefaults()" in app
    assert "applyLLMDefaultsToGroups" in app
    assert "state.selectedPack.population_templates" in app
    assert "addPopulationFromTemplate" in app
    assert "renderPopulationWarnings" in app


def test_run_accepts_validated_population_composition(tmp_path) -> None:
    server = create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        pack = next(
            item
            for item in json.loads(_request(base + "/api/packs")[2])["packs"]
            if item["id"] == "public_goods"
        )
        composition = []
        counts = {item["archetype"]: item["count"] for item in pack["composition"]["population"]}
        for archetype_id, archetype in pack["composition"]["archetypes"].items():
            composition.append(
                {"archetype_id": archetype_id, "count": counts[archetype_id], **archetype}
            )
        composition[0]["count"] += 1
        response = json.loads(
            _request(
                base + "/api/runs",
                {
                    "pack": "public_goods",
                    "seeds": [7],
                    "parameters": {"steps": 2},
                    "population_overrides": composition,
                },
            )[2]
        )
        job_id = response["jobs"][0]["id"]
        for _ in range(100):
            job = json.loads(_request(base + f"/api/jobs/{job_id}")[2])["job"]
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)
        assert job["status"] == "completed", job
        result = json.loads(_request(base + f"/api/runs/{job_id}")[2])
        assert len(result["view"]["agents"]) == sum(item["count"] for item in composition)
        assert result["view"]["composition"][0]["count"] == composition[0]["count"]
    finally:
        server.shutdown()
        server.server_close()


def test_ui_rejects_invalid_pagination_and_large_seed_batches(tmp_path) -> None:
    server = create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body = _error_request(base + "/api/runs?limit=-1")
        assert status == 400
        assert "limit" in body["error"]

        status, body = _error_request(
            base + "/api/runs", {"pack": "public_goods", "seeds": list(range(101))}
        )
        assert status == 400
        assert "seeds" in body["error"]
        assert server.jobs.list() == []
    finally:
        server.shutdown()
        server.server_close()


def test_model_provider_failures_return_bad_gateway(tmp_path, monkeypatch) -> None:
    def fail_models(base_url, api_key):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(ui_server, "fetch_openai_compatible_models", fail_models)
    server = create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, body = _error_request(
            base + "/api/models",
            {"base_url": "https://models.example/v1", "api_key": "secret"},
        )
        assert status == 502
        assert body == {"error": "provider unavailable"}
    finally:
        server.shutdown()
        server.server_close()


def test_study_trials_endpoint_is_paginated_and_ui_loads_every_page(tmp_path) -> None:
    trials = [
        TrialRecord(
            number=number,
            parameters={"rate": number},
            seeds=[],
            objective_values={"score": float(number)},
            feasible=True,
        )
        for number in range(205)
    ]
    summary = StudySummary(
        study_id="large-study",
        pack_id="public_goods",
        mode=StudyMode.SINGLE,
        status=RunStatus.COMPLETED,
        objectives=["score"],
        parameters=["rate"],
        seeds=[7],
        trial_count=len(trials),
        trials=trials[:200],
    )
    server = create_server(host="127.0.0.1", port=0, artifact_root=tmp_path)
    server.jobs.catalog.upsert_study(summary)
    server.jobs.catalog.replace_trials(summary.study_id, trials)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        payload = json.loads(
            _request(base + "/api/studies/large-study/trials?limit=100&offset=200")[2]
        )
        assert payload["total"] == 205
        assert len(payload["trials"]) == 5
        assert payload["trials"][0]["number"] == 200
    finally:
        server.shutdown()
        server.server_close()

    app = (ui_server.Path(ui_server.__file__).parent / "static" / "app.js").read_text()
    assert "loadAllStudyTrials" in app
    assert "Complete trial set" in app
    assert "/trials?limit=${pageSize}&offset=${trials.length}" in app
    assert "Seeds must be comma-separated integers" in app
