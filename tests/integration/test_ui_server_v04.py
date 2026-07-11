from __future__ import annotations

import json
import threading
import time
from urllib.request import Request, urlopen

from icframe.ui.server import create_server


def _request(url: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return response.status, response.headers.get_content_type(), response.read()


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
        assert {pack["id"] for pack in packs} == {
            "public_goods",
            "software_organization",
            "delayed_reward_learning",
        }
        llm_pack = next(pack for pack in packs if pack["id"] == "software_organization")
        assert llm_pack["llm"]["enabled"] is True
        assert "valid_actions" in llm_pack["llm"]["prompt_preview"]
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
    finally:
        server.shutdown()
        server.server_close()
