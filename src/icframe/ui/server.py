from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from icframe.artifacts import ArtifactObserver, update_manifest
from icframe.catalog import Catalog
from icframe.core import list_domain_packs, load_domain_pack, run_experiment
from icframe.domain.incentive_spec import IncentiveSpec, PolicyKind, RetentionProfile
from icframe.domain.run import LiveLLMBudget, RunConfig, RunStatus, StudyConfig, StudyMode
from icframe.llm import LiteLLMClient, LLMClient
from icframe.reports import render_html_report
from icframe.reports.view_models import run_view_model, study_view_model
from icframe.runtime_settings import (
    fetch_openai_compatible_models,
    load_runtime_llm_settings,
)
from icframe.study import run_study
from icframe.ui.request_models import (
    ModelsRequest,
    PaginationQuery,
    RunRequest,
    StudyRequest,
    TrialRerunRequest,
    validated_payload,
)


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    status: RunStatus
    request: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "request": self.request,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    def catalog_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "pack_id": self.request.get("pack", ""),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "job": True,
        }
        if self.kind == "run":
            payload.update(
                {
                    "seed": self.request.get("seed"),
                    "retention": self.request.get("retention", "experiment"),
                }
            )
        else:
            payload.update(
                {
                    "mode": self.request.get("mode", "single"),
                    "trial_count": 0,
                    "requested_trials": self.request.get("trials", 20),
                }
            )
        return payload


class CancellableArtifactObserver(ArtifactObserver):
    def __init__(self, *args: Any, cancel_event: threading.Event, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cancel_event = cancel_event

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


class JobManager:
    """Keep active jobs and a bounded recent-job window in memory."""

    def __init__(
        self, artifact_root: Path, workers: int = 4, max_completed_jobs: int = 200
    ) -> None:
        self.artifact_root = artifact_root
        self.catalog = Catalog(artifact_root)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="icframe-ui")
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.max_completed_jobs = max(0, max_completed_jobs)
        self._recover_interrupted_manifests()

    def submit_run(self, payload: dict[str, Any]) -> Job:
        payload = validated_payload(RunRequest, payload)
        job = Job(
            id=f"run_{uuid.uuid4().hex[:12]}",
            kind="run",
            status=RunStatus.QUEUED,
            request=_public_request(payload),
        )
        with self.lock:
            self.jobs[job.id] = job
        self.executor.submit(self._run, job, payload)
        return job

    def submit_study(self, payload: dict[str, Any]) -> Job:
        payload = validated_payload(StudyRequest, payload)
        job = Job(
            id=f"study_{uuid.uuid4().hex[:12]}",
            kind="study",
            status=RunStatus.QUEUED,
            request=_public_request(payload),
        )
        with self.lock:
            self.jobs[job.id] = job
        self.executor.submit(self._study, job, payload)
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [item.payload() for item in jobs]

    def catalog_rows(self, kind: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self.lock:
            persisted_ids = {str(row["id"]) for row in rows}
            jobs = sorted(
                (job for job in self.jobs.values() if job.kind == kind),
                key=lambda item: item.created_at,
                reverse=True,
            )
            transient = [
                job.catalog_payload()
                for job in jobs
                if job.status in {RunStatus.QUEUED, RunStatus.RUNNING}
                or (
                    job.status in {RunStatus.FAILED, RunStatus.CANCELLED}
                    and job.id not in persisted_ids
                )
            ]
            transient_ids = {str(row["id"]) for row in transient}
            return transient + [row for row in rows if str(row["id"]) not in transient_ids]

    def active_count(self, kind: str) -> int:
        with self.lock:
            return sum(
                job.kind == kind and job.status in {RunStatus.QUEUED, RunStatus.RUNNING}
                for job in self.jobs.values()
            )

    def cancel(self, job_id: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
                job.cancel_event.set()
                if job.status is RunStatus.QUEUED:
                    job.status = RunStatus.CANCELLED
                job.updated_at = time.time()
                self._prune_completed_jobs()
            return job

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: Job, payload: dict[str, Any]) -> None:
        if job.cancel_event.is_set():
            return
        self._set(job, RunStatus.RUNNING)
        try:
            pack = _configure_llm_pack(load_domain_pack(_pack_id(payload)), payload)
            retention = RetentionProfile(payload.get("retention", "experiment"))
            observer = CancellableArtifactObserver(
                self.artifact_root,
                job.id,
                retention,
                cancel_event=job.cancel_event,
            )
            run_experiment(
                pack,
                RunConfig(
                    run_id=job.id,
                    seed=_optional_int(payload.get("seed")),
                    parameters=dict(payload.get("parameters") or {}),
                    retention=retention,
                    sample_every_steps=_optional_int(payload.get("sample_every_steps")),
                    artifact_root=self.artifact_root,
                ),
                llm_client=_llm_client(payload),
                observer=observer,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._mark_manifest(job, RunStatus.FAILED, error)
            self._set(job, RunStatus.FAILED, error)
            return
        self._set(
            job,
            RunStatus.CANCELLED if job.cancel_event.is_set() else RunStatus.COMPLETED,
        )

    def _study(self, job: Job, payload: dict[str, Any]) -> None:
        if job.cancel_event.is_set():
            return
        self._set(job, RunStatus.RUNNING)
        try:
            pack = _configure_llm_pack(load_domain_pack(_pack_id(payload)), payload)
            mode = StudyMode(payload.get("mode", "single"))
            objectives = list(payload.get("objectives") or [])
            if not objectives:
                objectives = (
                    [pack.manifest.study.single_objective]
                    if mode is StudyMode.SINGLE
                    else list(pack.manifest.study.pareto_objectives)
                )
            allow_live = bool(payload.get("allow_live_llm", False))
            config = StudyConfig(
                study_id=job.id,
                mode=mode,
                objectives=objectives,
                parameters=list(
                    payload.get("parameters")
                    or [item.id for item in pack.manifest.parameters if item.optimizable]
                ),
                parameter_ranges=dict(payload.get("parameter_ranges") or {}),
                trials=int(payload.get("trials", 20)),
                seeds=[int(seed) for seed in payload.get("seeds") or pack.spec.experiment.seeds],
                workers=int(
                    payload.get(
                        "workers",
                        1 if allow_live else min(4, os.cpu_count() or 1),
                    )
                ),
                artifact_root=self.artifact_root,
                live_llm=LiveLLMBudget(
                    enabled=allow_live,
                    max_calls=_optional_int(payload.get("max_llm_calls")),
                    max_cost_usd=_optional_float(payload.get("max_llm_cost_usd")),
                ),
            )
            run_study(
                pack,
                config,
                llm_client=_llm_client(payload),
                cancel_check=job.cancel_event.is_set,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._mark_manifest(job, RunStatus.FAILED, error)
            self._set(job, RunStatus.FAILED, error)
            return
        self._set(
            job,
            RunStatus.CANCELLED if job.cancel_event.is_set() else RunStatus.COMPLETED,
        )

    def _set(self, job: Job, status: RunStatus, error: str | None = None) -> None:
        with self.lock:
            job.status = status
            job.error = error
            job.updated_at = time.time()
            if status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                self._prune_completed_jobs()

    def _prune_completed_jobs(self) -> None:
        completed = sorted(
            (
                item
                for item in self.jobs.values()
                if item.status not in {RunStatus.QUEUED, RunStatus.RUNNING}
            ),
            key=lambda item: item.updated_at,
            reverse=True,
        )
        for item in completed[self.max_completed_jobs :]:
            self.jobs.pop(item.id, None)

    def _mark_manifest(self, job: Job, status: RunStatus, error: str) -> None:
        directory = "runs" if job.kind == "run" else "studies"
        path = self.artifact_root / directory / job.id / "manifest.json"
        update_manifest(path, status.value, error=error)

    def _recover_interrupted_manifests(self) -> None:
        for kind in ("runs", "studies"):
            for path in (self.artifact_root / kind).glob("*/manifest.json"):
                try:
                    payload = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("status") != "running":
                    continue
                update_manifest(path, RunStatus.INTERRUPTED.value)


def serve_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    artifact_root: str | Path = ".artifacts/icframe",
) -> None:
    server = create_server(host=host, port=port, artifact_root=artifact_root)
    print(f"ICFRAME simulator UI running at http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down ICFRAME simulator UI")
    finally:
        server.server_close()


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    artifact_root: str | Path = ".artifacts/icframe",
) -> ThreadingHTTPServer:
    manager = JobManager(Path(artifact_root).resolve())

    class Handler(ICFrameUIHandler):
        jobs = manager

    server = ManagedUIHTTPServer((host, port), Handler)
    server.jobs = manager
    return server


class ManagedUIHTTPServer(ThreadingHTTPServer):
    jobs: JobManager

    def server_close(self) -> None:
        jobs = getattr(self, "jobs", None)
        if jobs is not None:
            jobs.close()
        super().server_close()


class ICFrameUIHandler(BaseHTTPRequestHandler):
    jobs: JobManager
    server_version = "ICFrameUI/0.4"

    def do_GET(self) -> None:
        try:
            self._do_get()
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, _validation_message(exc))

    def _do_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/":
            self._send_static("index.html")
        elif path.startswith("/static/"):
            self._send_static(path.removeprefix("/static/"))
        elif path == "/api/packs":
            self._send_json(
                {"packs": [_pack_payload(item.pack.id) for item in list_domain_packs()]}
            )
        elif path == "/api/jobs":
            self._send_json({"jobs": self.jobs.list()})
        elif path == "/api/runs":
            pagination = _pagination(query)
            rows = self.jobs.catalog.list_runs(pagination.limit, pagination.offset)
            self._send_json(
                {
                    "runs": self.jobs.catalog_rows("run", rows),
                    "total": self.jobs.catalog.count_runs() + self.jobs.active_count("run"),
                }
            )
        elif path == "/api/studies":
            pagination = _pagination(query)
            rows = self.jobs.catalog.list_studies(pagination.limit, pagination.offset)
            self._send_json(
                {
                    "studies": self.jobs.catalog_rows("study", rows),
                    "total": self.jobs.catalog.count_studies() + self.jobs.active_count("study"),
                }
            )
        elif path == "/api/settings":
            self._send_json({"settings": load_runtime_llm_settings().redacted()})
        elif path.startswith("/api/jobs/"):
            self._job_get(path)
        elif path.startswith("/api/runs/"):
            self._run_get(path)
        elif path.startswith("/api/studies/"):
            self._study_get(path)
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/runs":
                payload = validated_payload(RunRequest, payload)
                requested_seeds = payload.get("seeds") or [payload.get("seed")]
                jobs = []
                for seed in requested_seeds:
                    request = dict(payload)
                    request.pop("seeds", None)
                    request["seed"] = seed
                    jobs.append(self.jobs.submit_run(request))
                self._send_json(
                    {"jobs": [job.payload() for job in jobs]},
                    HTTPStatus.ACCEPTED,
                )
            elif path == "/api/studies":
                payload = validated_payload(StudyRequest, payload)
                job = self.jobs.submit_study(payload)
                self._send_json({"job": job.payload()}, HTTPStatus.ACCEPTED)
            elif path == "/api/catalog/rebuild":
                self._send_json(self.jobs.catalog.rebuild())
            elif path == "/api/models":
                payload = validated_payload(ModelsRequest, payload)
                settings = load_runtime_llm_settings(
                    base_url=payload.get("base_url"),
                    api_key=payload.get("api_key"),
                )
                if not settings.api_key or not settings.base_url:
                    raise ValueError("base URL and API key are required")
                self._send_json(
                    {"models": fetch_openai_compatible_models(settings.base_url, settings.api_key)}
                )
            elif path.startswith("/api/studies/") and path.endswith("/rerun"):
                payload = validated_payload(TrialRerunRequest, payload)
                self._rerun_trial(path, payload)
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
                job = self.jobs.cancel(job_id)
                if job is None:
                    self._send_error(HTTPStatus.NOT_FOUND, "job not found")
                else:
                    self._send_json({"job": job.payload()})
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, _validation_message(exc))
        except RuntimeError as exc:
            self._send_error(HTTPStatus.BAD_GATEWAY, str(exc))

    def _job_get(self, path: str) -> None:
        job = self.jobs.get(path.removeprefix("/api/jobs/").strip("/"))
        if job is None:
            self._send_error(HTTPStatus.NOT_FOUND, "job not found")
        else:
            self._send_json({"job": job.payload()})

    def _run_get(self, path: str) -> None:
        suffix = path.removeprefix("/api/runs/").strip("/")
        report = suffix.endswith("/report")
        run_id = suffix.removesuffix("/report").strip("/")
        summary = self.jobs.catalog.get_run(run_id)
        if summary is None:
            self._send_error(HTTPStatus.NOT_FOUND, "run not found")
        elif report:
            self._send_html(render_html_report(summary))
        else:
            self._send_json(
                {
                    "summary": summary.model_dump(mode="json"),
                    "view": run_view_model(summary).model_dump(mode="json"),
                }
            )

    def _study_get(self, path: str) -> None:
        suffix = path.removeprefix("/api/studies/").strip("/")
        if suffix.endswith("/trials"):
            study_id = suffix.removesuffix("/trials").strip("/")
            if self.jobs.catalog.get_study(study_id) is None:
                self._send_error(HTTPStatus.NOT_FOUND, "study not found")
                return
            query = parse_qs(urlparse(self.path).query)
            pagination = _pagination(query, default_limit=200)
            trials = self.jobs.catalog.list_trials(
                study_id, pagination.limit, pagination.offset
            )
            self._send_json(
                {
                    "trials": [trial.model_dump(mode="json") for trial in trials],
                    "total": self.jobs.catalog.count_trials(study_id),
                    "limit": pagination.limit,
                    "offset": pagination.offset,
                }
            )
            return
        report = suffix.endswith("/report")
        study_id = suffix.removesuffix("/report").strip("/")
        summary = self.jobs.catalog.get_study(study_id)
        if summary is None:
            self._send_error(HTTPStatus.NOT_FOUND, "study not found")
        elif report:
            self._send_html(render_html_report(summary))
        else:
            self._send_json(
                {
                    "summary": summary.model_dump(mode="json"),
                    "view": study_view_model(summary).model_dump(mode="json"),
                }
            )

    def _rerun_trial(self, path: str, payload: dict[str, Any]) -> None:
        suffix = path.removeprefix("/api/studies/").removesuffix("/rerun").strip("/")
        study_id, marker, number_value = suffix.partition("/trials/")
        if not marker:
            raise ValueError("trial rerun path is invalid")
        study = self.jobs.catalog.get_study(study_id)
        trial = self.jobs.catalog.get_trial(study_id, int(number_value))
        if study is None or trial is None:
            self._send_error(HTTPStatus.NOT_FOUND, "study trial not found")
            return
        seeds = payload.get("seeds") or study.seeds
        jobs = []
        for seed in seeds:
            jobs.append(
                self.jobs.submit_run(
                    {
                        "pack": study.pack_id,
                        "seed": int(seed),
                        "parameters": trial.parameters,
                        "retention": payload.get("retention", "experiment"),
                    }
                )
            )
        self._send_json(
            {"jobs": [job.payload() for job in jobs]},
            HTTPStatus.ACCEPTED,
        )

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request body is too large")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, value: str) -> None:
        body = value.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, name: str) -> None:
        static_root = Path(__file__).parent / "static"
        path = (static_root / name).resolve()
        if static_root.resolve() not in path.parents or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "asset not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _pack_payload(pack_id: str) -> dict[str, Any]:
    pack = load_domain_pack(pack_id)
    return {
        "id": pack.id,
        "title": pack.manifest.pack.title,
        "description": pack.manifest.pack.description,
        "steps": pack.spec.experiment.steps,
        "seeds": pack.spec.experiment.seeds,
        "schedule": pack.spec.experiment.schedule.value,
        "parameters": [item.model_dump(mode="json") for item in pack.manifest.parameters],
        "objectives": {
            name: value.model_dump(mode="json")
            for name, value in pack.spec.evaluation.objectives.items()
        },
        "study": pack.manifest.study.model_dump(mode="json"),
        "llm": _llm_pack_payload(pack),
    }


def _pack_id(payload: dict[str, Any]) -> str:
    value = str(payload.get("pack", "")).strip()
    if not value:
        raise ValueError("pack is required")
    if value.endswith(".json"):
        raise ValueError("legacy JSON scenarios are unsupported; select a v0.4 domain pack")
    return value


def _llm_client(payload: dict[str, Any]) -> LLMClient | None:
    mode = payload.get("llm_mode", "none")
    if mode == "live":
        settings = load_runtime_llm_settings(
            base_url=payload.get("llm_base_url"),
            api_key=payload.get("llm_api_key"),
            model=payload.get("llm_model"),
        )
        if not settings.api_key and not _is_local_endpoint(settings.base_url):
            raise ValueError("a live LLM API key is required for remote endpoints")
        return LiteLLMClient(settings)
    if mode != "none":
        raise ValueError(f"unsupported LLM mode: {mode}")
    return None


def _configure_llm_pack(pack, payload: dict[str, Any]):
    if payload.get("llm_mode", "none") != "live":
        return pack
    raw = pack.spec.model_dump(mode="python", by_alias=True)
    changed = False
    for archetype in raw["archetypes"].values():
        if PolicyKind(archetype["policy"]) is not PolicyKind.LLM:
            continue
        changed = True
        llm = archetype["llm"]
        if str(payload.get("llm_model", "")).strip():
            llm["model"] = str(payload["llm_model"]).strip()
        if "llm_system_prompt" in payload:
            llm["system_prompt"] = str(payload["llm_system_prompt"])
        if payload.get("llm_temperature") not in {None, ""}:
            llm["temperature"] = float(payload["llm_temperature"])
    if not changed:
        raise ValueError(f"domain pack {pack.id} has no LLM policy")
    return replace(pack, spec=IncentiveSpec.model_validate(raw))


def _llm_pack_payload(pack) -> dict[str, Any]:
    for archetype_id, archetype in pack.spec.archetypes.items():
        if archetype.policy is not PolicyKind.LLM or archetype.llm is None:
            continue
        actions = [
            {
                "action": transition.action,
                "label": transition.prompt.label if transition.prompt else None,
                "description": transition.prompt.description if transition.prompt else None,
            }
            for transition in pack.spec.transitions
        ]
        preview = {
            "agent_id": f"{archetype_id}_000",
            "step": 1,
            "state": archetype.initial_state or pack.spec.states.initial,
            "resources": archetype.initial_resources,
            "visible_history": [],
            "valid_actions": actions,
        }
        return {
            "enabled": True,
            "model": archetype.llm.model,
            "temperature": archetype.llm.temperature,
            "system_prompt": archetype.llm.system_prompt,
            "prompt_preview": json.dumps(preview, indent=2),
        }
    return {"enabled": False}


def _is_local_endpoint(base_url: str | None) -> bool:
    return bool(
        base_url
        and (
            base_url.startswith("http://127.0.0.1")
            or base_url.startswith("http://localhost")
            or base_url.startswith("http://[::1]")
        )
    )


def _public_request(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in {"llm_api_key"}}


def _optional_int(value: Any) -> int | None:
    return None if value in {None, ""} else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value in {None, ""} else float(value)


def _pagination(
    query: dict[str, list[str]], *, default_limit: int = 100
) -> PaginationQuery:
    return PaginationQuery.model_validate(
        {
            "limit": query.get("limit", [str(default_limit)])[0],
            "offset": query.get("offset", ["0"])[0],
        }
    )


def _validation_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        error = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in error["loc"])
        return f"{location}: {error['msg']}" if location else str(error["msg"])
    return str(exc)
