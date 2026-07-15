from __future__ import annotations

import hashlib
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
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from icframe.artifacts import ArtifactObserver, update_manifest
from icframe.catalog import Catalog
from icframe.core import (
    LoadedDomainPack,
    list_domain_packs,
    load_domain_pack,
    run_experiment,
)
from icframe.domain.incentive_spec import (
    DomainPackManifest,
    GuidedParameter,
    IncentiveSpec,
    ParameterEntity,
    ParameterTarget,
    ParameterType,
    PolicyKind,
    RetentionProfile,
)
from icframe.domain.run import (
    Checkpoint,
    LiveLLMBudget,
    PlannerKind,
    RunConfig,
    RunStatus,
    RunSummary,
    StudyConfig,
    StudyMode,
)
from icframe.llm import LiteLLMClient, LLMClient
from icframe.orchestration.service import JobCoordinator
from icframe.profiles import (
    ProfileRegistry,
    apply_llm_profile,
    llm_client_for_profile,
    load_profiles,
)
from icframe.reports import render_html_report
from icframe.reports.view_models import run_view_model, study_view_model
from icframe.runtime_settings import (
    RuntimeLLMSettings,
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
from icframe.version import __version__

UI_API_VERSION = "1"
UI_CAPABILITIES = frozenset(
    {
        "live_jobs",
        "causal_mechanics",
        "population_templates",
        "population_overrides",
        "policy_templates",
        "quick_values",
        "runtime_handshake",
        "execution_profiles",
        "planned_studies",
    }
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
    progress: dict[str, Any] = field(default_factory=dict)
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
            "progress": dict(self.progress),
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
    def __init__(
        self,
        *args: Any,
        cancel_event: threading.Event,
        progress_callback,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cancel_event = cancel_event
        self.progress_callback = progress_callback
        self.llm_progress: dict[str, Any] = {
            "attempted": 0,
            "failed": 0,
            "total_tokens": 0,
            "cost": 0.0,
        }
        self.recent_errors: list[str] = []
        self.recent_latencies: list[float] = []

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def start(self, context: dict[str, Any]) -> None:
        super().start(context)
        self.progress_callback(
            steps_completed=0,
            steps_planned=int(context["spec"]["experiment"]["steps"]),
            metrics={},
            llm={"attempted": 0, "failed": 0, "total_tokens": 0, "cost": 0.0},
            recent_errors=[],
        )

    def checkpoint(self, value: Checkpoint) -> None:
        super().checkpoint(value)
        self.progress_callback(steps_completed=value.step, metrics=dict(value.metrics))

    def decision(self, value) -> None:
        super().decision(value)
        if value.llm_call is None:
            return
        current = dict(self.llm_progress)
        current["attempted"] = int(current.get("attempted", 0)) + 1
        current["total_tokens"] = int(current.get("total_tokens", 0)) + int(
            value.llm_call.get("total_tokens", 0) or 0
        )
        if value.llm_call.get("status") == "failed":
            current["failed"] = int(current.get("failed", 0)) + 1
        failure = value.llm_call.get("failure_classification") or value.llm_call.get("error")
        if failure:
            self.recent_errors = [*self.recent_errors, str(failure)][-5:]
        latency = value.llm_call.get("latency_ms")
        if isinstance(latency, int | float):
            self.recent_latencies = [*self.recent_latencies, float(latency)][-200:]
            ordered = sorted(self.recent_latencies)
            current["latency_p50_ms"] = ordered[(len(ordered) - 1) // 2]
            current["latency_p95_ms"] = ordered[round((len(ordered) - 1) * 0.95)]
        cost = value.llm_call.get("estimated_cost")
        current["cost"] = (
            None
            if cost is None or current.get("cost") is None
            else float(current.get("cost", 0.0)) + float(cost)
        )
        self.llm_progress = current
        self.progress_callback(llm=current, recent_errors=list(self.recent_errors))

    def finish(self, value: RunSummary) -> None:
        super().finish(value)
        self.progress_callback(
            steps_completed=value.steps_completed,
            metrics=dict(value.metrics),
            llm={
                "attempted": value.llm_usage.attempted,
                "failed": value.llm_usage.failed,
                "malformed": value.llm_usage.malformed,
                "invalid": value.llm_usage.invalid,
                "total_tokens": value.llm_usage.total_tokens,
                "cost": value.llm_usage.estimated_cost_usd,
            },
        )


class JobManager:
    """Keep active jobs and a bounded recent-job window in memory."""

    def __init__(
        self,
        artifact_root: Path,
        workers: int = 4,
        max_completed_jobs: int = 200,
        packs: dict[str, LoadedDomainPack] | None = None,
        profiles: ProfileRegistry | None = None,
    ) -> None:
        self.artifact_root = artifact_root
        self.catalog = Catalog(artifact_root)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="icframe-ui")
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.max_completed_jobs = max(0, max_completed_jobs)
        self.packs = packs
        self.profiles = profiles or load_profiles()
        self.coordinator = JobCoordinator(artifact_root, profiles=self.profiles)
        self._recover_interrupted_manifests()
        self._recover_orchestrated_jobs()

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
        if payload.get("execution_profile", "local") == "local":
            self.executor.submit(self._run, job, payload)
        else:
            try:
                self._submit_orchestrated_run(job, payload)
            except Exception:
                with self.lock:
                    self.jobs.pop(job.id, None)
                raise
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
        if payload.get("execution_profile", "local") == "local":
            self.executor.submit(self._study, job, payload)
        else:
            try:
                self._submit_orchestrated_study(job, payload)
            except Exception:
                with self.lock:
                    self.jobs.pop(job.id, None)
                raise
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None:
                self._refresh_study_progress(job)
            return job

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
            for job in jobs:
                self._refresh_study_progress(job)
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
                remote = job.request.get("execution_profile", "local") != "local"
                if remote:
                    self.coordinator.cancel_job(job.id)
                if job.status is RunStatus.QUEUED and not remote:
                    job.status = RunStatus.CANCELLED
                job.updated_at = time.time()
                self._prune_completed_jobs()
            return job

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.coordinator.close()

    def _run(self, job: Job, payload: dict[str, Any]) -> None:
        if job.cancel_event.is_set():
            return
        self._set(job, RunStatus.RUNNING)
        try:
            pack = _configure_population_pack(self._load_pack(_pack_id(payload)), payload)
            pack = _configure_llm_pack(pack, payload)
            pack = self._profile_pack(pack, payload)
            retention = RetentionProfile(payload.get("retention", "experiment"))
            observer = CancellableArtifactObserver(
                self.artifact_root,
                job.id,
                retention,
                cancel_event=job.cancel_event,
                progress_callback=lambda **values: self._progress(job, **values),
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
                llm_client=self._llm_client(payload),
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
            pack = _configure_population_pack(self._load_pack(_pack_id(payload)), payload)
            pack = _configure_llm_pack(pack, payload)
            pack = self._profile_pack(pack, payload)
            mode = StudyMode(payload.get("mode", "single"))
            objectives = list(payload.get("objectives") or [])
            if not objectives:
                objectives = (
                    [pack.manifest.study.single_objective]
                    if mode is StudyMode.SINGLE
                    else list(pack.manifest.study.pareto_objectives)
                )
            allow_live = bool(payload.get("allow_live_llm", False))
            config = self._study_config(job, pack, payload, mode, objectives, allow_live)
            run_study(
                pack,
                config,
                llm_client=self._llm_client(payload),
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

    def _progress(self, job: Job, **values: Any) -> None:
        with self.lock:
            job.progress.update(values)
            job.updated_at = time.time()

    def _refresh_study_progress(self, job: Job) -> None:
        if job.kind != "study" or job.status is not RunStatus.RUNNING:
            return
        path = self.artifact_root / "studies" / job.id / "trials.jsonl"
        if not path.exists():
            return
        try:
            completed = sum(1 for line in path.open() if line.strip())
        except OSError:
            return
        job.progress.update(
            trials_completed=completed,
            trials_planned=int(job.request.get("trials", 0) or 0),
        )

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
        resumable_remote = {
            handle.id
            for handle in self.coordinator.sync_jobs()
            if handle.backend_profile != "local"
            and handle.status in {RunStatus.QUEUED, RunStatus.RUNNING}
        }
        for kind in ("runs", "studies"):
            for path in (self.artifact_root / kind).glob("*/manifest.json"):
                if path.parent.name in resumable_remote:
                    continue
                try:
                    payload = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("status") != "running":
                    continue
                update_manifest(path, RunStatus.INTERRUPTED.value)

    def _submit_orchestrated_run(self, job: Job, payload: dict[str, Any]) -> None:
        pack = _configure_population_pack(self._load_pack(_pack_id(payload)), payload)
        pack = _configure_llm_pack(pack, payload)
        pack = self._profile_pack(pack, payload)
        self.coordinator.submit_run(
            pack,
            RunConfig(
                run_id=job.id,
                seed=_optional_int(payload.get("seed")),
                parameters=dict(payload.get("parameters") or {}),
                retention=RetentionProfile(payload.get("retention", "experiment")),
                sample_every_steps=_optional_int(payload.get("sample_every_steps")),
                artifact_root=self.artifact_root,
            ),
            backend_profile=str(payload["execution_profile"]),
            llm_profile=payload.get("llm_profile"),
        )
        self.executor.submit(self._watch_orchestrated, job)

    def _submit_orchestrated_study(self, job: Job, payload: dict[str, Any]) -> None:
        pack = _configure_population_pack(self._load_pack(_pack_id(payload)), payload)
        pack = _configure_llm_pack(pack, payload)
        pack = self._profile_pack(pack, payload)
        mode = StudyMode(payload.get("mode", "single"))
        objectives = list(payload.get("objectives") or [])
        if not objectives:
            objectives = (
                [pack.manifest.study.single_objective]
                if mode is StudyMode.SINGLE
                else list(pack.manifest.study.pareto_objectives)
            )
        allow_live = bool(payload.get("allow_live_llm", False))
        config = self._study_config(job, pack, payload, mode, objectives, allow_live)
        self.coordinator.submit_study(
            pack,
            config,
            backend_profile=str(payload["execution_profile"]),
            llm_profile=payload.get("llm_profile"),
        )
        self.executor.submit(self._watch_orchestrated, job)

    def _study_config(
        self,
        job: Job,
        pack: LoadedDomainPack,
        payload: dict[str, Any],
        mode: StudyMode,
        objectives: list[str],
        allow_live: bool,
    ) -> StudyConfig:
        return StudyConfig(
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
            planner=PlannerKind(payload.get("planner", "random")),
            planner_seed=int(payload.get("planner_seed", 0)),
            parameter_matrix=dict(payload.get("parameter_matrix") or {}),
        )

    def _watch_orchestrated(self, job: Job) -> None:
        while True:
            handle = self.coordinator.get_job(job.id)
            if handle is None:
                self._set(job, RunStatus.FAILED, "orchestration manifest disappeared")
                return
            with self.lock:
                job.status = handle.status
                job.error = handle.error
                job.progress.update(
                    trials_completed=handle.completed_trials,
                    trials_planned=handle.planned_trials,
                    shard_count=handle.shard_count,
                    remote_job_ids=handle.remote_job_ids,
                    backend_profile=handle.backend_profile,
                    retry_count=handle.retry_count,
                    artifact_import_state=handle.artifact_import_state,
                    cancel_requested=handle.cancel_requested,
                )
                job.updated_at = time.time()
            if handle.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                self._prune_completed_jobs()
                return
            time.sleep(0.2)

    def _recover_orchestrated_jobs(self) -> None:
        for handle in self.coordinator.sync_jobs():
            if handle.backend_profile == "local":
                continue
            request_path = self.artifact_root / "jobs" / handle.id / "request.json"
            try:
                request = json.loads(request_path.read_text())
            except (OSError, json.JSONDecodeError):
                request = {}
            if handle.kind == "study" and "config" in request:
                request = {"pack": request.get("pack"), **request.get("config", {})}
            request["execution_profile"] = handle.backend_profile
            job = Job(
                id=handle.id,
                kind=handle.kind,
                status=handle.status,
                request=_public_request(request),
                error=handle.error,
                progress={
                    "trials_completed": handle.completed_trials,
                    "trials_planned": handle.planned_trials,
                    "shard_count": handle.shard_count,
                    "remote_job_ids": handle.remote_job_ids,
                    "retry_count": handle.retry_count,
                    "artifact_import_state": handle.artifact_import_state,
                    "cancel_requested": handle.cancel_requested,
                },
            )
            self.jobs[job.id] = job
            if handle.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
                self.executor.submit(self._watch_orchestrated, job)

    def _llm_client(self, payload: dict[str, Any]) -> LLMClient | None:
        profile_name = payload.get("llm_profile")
        if profile_name:
            return llm_client_for_profile(self.profiles.llm_profile(str(profile_name)))
        return _llm_client(payload)

    def _profile_pack(
        self,
        pack: LoadedDomainPack,
        payload: dict[str, Any],
    ) -> LoadedDomainPack:
        profile_name = payload.get("llm_profile")
        return (
            apply_llm_profile(pack, self.profiles.llm_profile(str(profile_name)))
            if profile_name
            else pack
        )

    def _load_pack(self, pack_id: str) -> LoadedDomainPack:
        if self.packs is None:
            return load_domain_pack(pack_id)
        try:
            return self.packs[pack_id]
        except KeyError as exc:
            raise ValueError(f"unknown domain pack: {pack_id}") from exc


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
    profiles = load_profiles()
    packs = {
        manifest.pack.id: load_domain_pack(manifest.pack.id)
        for manifest in list_domain_packs()
    }
    pack_payload_snapshot = [_pack_payload_from_pack(pack) for pack in packs.values()]
    asset_snapshot = _load_static_assets()
    asset_revision = hashlib.sha256(
        b"".join(body for body, _ in asset_snapshot.values())
    ).hexdigest()[:12]
    manager = JobManager(Path(artifact_root).resolve(), packs=packs, profiles=profiles)

    class Handler(ICFrameUIHandler):
        jobs = manager
        packs = pack_payload_snapshot
        profile_payload: ClassVar[dict[str, object]] = profiles.public_payload()
        static_assets = asset_snapshot
        runtime: ClassVar[dict[str, Any]] = {
            "version": __version__,
            "ui_api_version": UI_API_VERSION,
            "capabilities": sorted(UI_CAPABILITIES),
            "asset_revision": asset_revision,
        }

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
    packs: ClassVar[list[dict[str, Any]]]
    static_assets: ClassVar[dict[str, tuple[bytes, str]]]
    runtime: ClassVar[dict[str, Any]]
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
            self._send_json({"packs": self.packs})
        elif path == "/api/runtime":
            self._send_json({"runtime": self.runtime})
        elif path == "/api/profiles":
            self._send_json({"profiles": self.profile_payload})
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
                if payload.get("llm_profile"):
                    profile = self.jobs.profiles.llm_profile(str(payload["llm_profile"]))
                    settings = RuntimeLLMSettings(
                        base_url=profile.base_url,
                        api_key=os.environ.get(profile.api_key_env),
                        model=profile.model,
                        api_key_source=f"environment:{profile.api_key_env}",
                    )
                else:
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
        suffix = path.removeprefix("/api/jobs/").strip("/")
        if suffix.endswith("/llm-calls"):
            job_id = suffix.removesuffix("/llm-calls").strip("/")
            job = self.jobs.get(job_id)
            if job is None or job.kind != "run":
                self._send_error(HTTPStatus.NOT_FOUND, "run job not found")
            else:
                self._llm_calls_path_get(
                    self.jobs.artifact_root / "runs" / job_id / "llm_calls.jsonl"
                )
            return
        job = self.jobs.get(suffix)
        if job is None:
            self._send_error(HTTPStatus.NOT_FOUND, "job not found")
        else:
            self._send_json({"job": job.payload()})

    def _run_get(self, path: str) -> None:
        suffix = path.removeprefix("/api/runs/").strip("/")
        if suffix.endswith("/llm-calls"):
            run_id = suffix.removesuffix("/llm-calls").strip("/")
            self._llm_calls_get(run_id)
            return
        report = suffix.endswith("/report")
        run_id = suffix.removesuffix("/report").strip("/")
        summary = self.jobs.catalog.get_run(run_id)
        if summary is None:
            self._send_error(HTTPStatus.NOT_FOUND, "run not found")
        elif report:
            manifest, spec = _artifact_context(self.jobs.artifact_root, "runs", run_id)
            self._send_html(render_html_report(summary, manifest=manifest, spec=spec))
        else:
            manifest, spec = _artifact_context(self.jobs.artifact_root, "runs", run_id)
            self._send_json(
                {
                    "summary": summary.model_dump(mode="json"),
                    "view": run_view_model(summary, manifest, spec).model_dump(mode="json"),
                }
            )

    def _llm_calls_get(self, run_id: str) -> None:
        path = self.jobs.artifact_root / "runs" / run_id / "llm_calls.jsonl"
        run_exists = (self.jobs.artifact_root / "runs" / run_id).is_dir()
        if not run_exists:
            self._send_error(HTTPStatus.NOT_FOUND, "run not found")
            return
        self._llm_calls_path_get(path)

    def _llm_calls_path_get(self, path: Path) -> None:
        pagination = _pagination(parse_qs(urlparse(self.path).query), default_limit=50)
        if pagination.limit > 100:
            raise ValueError("limit must be less than or equal to 100")
        calls = []
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    calls.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._send_json(
            {
                "calls": calls[pagination.offset : pagination.offset + pagination.limit],
                "total": len(calls),
                "limit": pagination.limit,
                "offset": pagination.offset,
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
            trials = self.jobs.catalog.list_trials(study_id, pagination.limit, pagination.offset)
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
            manifest, spec = _artifact_context(self.jobs.artifact_root, "studies", study_id)
            self._send_html(render_html_report(summary, manifest=manifest, spec=spec))
        else:
            manifest, spec = _artifact_context(self.jobs.artifact_root, "studies", study_id)
            self._send_json(
                {
                    "summary": summary.model_dump(mode="json"),
                    "view": study_view_model(summary, manifest, spec).model_dump(mode="json"),
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
        asset = self.static_assets.get(name)
        if asset is None:
            self._send_error(HTTPStatus.NOT_FOUND, "asset not found")
            return
        body, content_type = asset
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
    return _pack_payload_from_pack(load_domain_pack(pack_id))


def _pack_payload_from_pack(pack: LoadedDomainPack) -> dict[str, Any]:
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
        "composition": {
            "population": [item.model_dump(mode="json") for item in pack.spec.population],
            "archetypes": {
                name: item.model_dump(mode="json") for name, item in pack.spec.archetypes.items()
            },
            "visibility_profiles": sorted(pack.spec.visibility_profiles),
            "visibility_profile_details": {
                name: item.model_dump(mode="json")
                for name, item in pack.spec.visibility_profiles.items()
            },
            "outcome_channels": list(pack.spec.outcome_space.channels),
        },
        "policy_templates": _policy_templates(),
        "population_templates": _population_template_payload(pack),
        "mechanics_flow": (
            pack.manifest.report.mechanics_flow.model_dump(mode="json")
            if pack.manifest.report.mechanics_flow is not None
            else None
        ),
    }


def _load_static_assets() -> dict[str, tuple[bytes, str]]:
    static_root = Path(__file__).parent / "static"
    return {
        path.name: (
            path.read_bytes(),
            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )
        for path in static_root.iterdir()
        if path.is_file()
    }


def _population_template_payload(pack: LoadedDomainPack) -> list[dict[str, Any]]:
    declared = pack.manifest.population_templates
    if not declared:
        counts = {item.archetype: item.count for item in pack.spec.population}
        declared = [
            {
                "id": archetype_id,
                "label": archetype_id.replace("_", " ").title(),
                "description": "Canonical domain-pack population.",
                "archetype": archetype_id,
                "count": counts[archetype_id],
            }
            for archetype_id in counts
        ]
    result = []
    for item in declared:
        payload = item if isinstance(item, dict) else item.model_dump(mode="json")
        archetype_id = str(payload["archetype"])
        result.append(
            {
                "id": payload["id"],
                "label": payload["label"],
                "description": payload.get("description", ""),
                "archetype_id": archetype_id,
                "count": payload["count"],
                **pack.spec.archetypes[archetype_id].model_dump(mode="json"),
            }
        )
    return result


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


_POLICY_FIELDS: dict[PolicyKind, dict[str, tuple[float | None, float | None, float]]] = {
    PolicyKind.DETERMINISTIC: {},
    PolicyKind.STOCHASTIC_WEIGHTED: {},
    PolicyKind.EPSILON_GREEDY: {
        "exploration_rate": (0.0, 1.0, 0.01),
        "initial_value": (None, None, 0.1),
    },
    PolicyKind.UCB: {
        "exploration_coefficient": (0.0, None, 0.1),
        "initial_value": (None, None, 0.1),
    },
    PolicyKind.GAUSSIAN_THOMPSON: {
        "prior_variance": (0.000001, None, 0.1),
        "initial_value": (None, None, 0.1),
    },
    PolicyKind.CONTEXTUAL: {
        "exploration_rate": (0.0, 1.0, 0.01),
        "learning_rate": (0.0, 1.0, 0.01),
    },
    PolicyKind.Q_LEARNING: {
        "exploration_rate": (0.0, 1.0, 0.01),
        "learning_rate": (0.0, 1.0, 0.01),
        "discount_factor": (0.0, 1.0, 0.01),
    },
    PolicyKind.LLM: {},
}


def _policy_templates() -> list[dict[str, Any]]:
    labels = {
        PolicyKind.DETERMINISTIC: "Deterministic",
        PolicyKind.STOCHASTIC_WEIGHTED: "Weighted stochastic",
        PolicyKind.EPSILON_GREEDY: "Epsilon-greedy bandit",
        PolicyKind.UCB: "UCB bandit",
        PolicyKind.GAUSSIAN_THOMPSON: "Gaussian Thompson bandit",
        PolicyKind.CONTEXTUAL: "Contextual bandit",
        PolicyKind.Q_LEARNING: "Simple Q-learning",
        PolicyKind.LLM: "LLM agent",
    }
    defaults = {
        "exploration_rate": 0.1,
        "exploration_coefficient": 1.0,
        "prior_variance": 1.0,
        "learning_rate": 0.1,
        "discount_factor": 0.9,
    }
    return [
        {
            "policy": policy.value,
            "label": labels[policy],
            "fields": [
                {
                    "id": name,
                    "minimum": bounds[0],
                    "maximum": bounds[1],
                    "step": bounds[2],
                    "default": defaults.get(name, 0.0),
                }
                for name, bounds in fields.items()
            ],
        }
        for policy, fields in _POLICY_FIELDS.items()
    ]


def _configure_population_pack(pack, payload: dict[str, Any]):
    overrides = payload.get("population_overrides")
    if not overrides:
        return pack
    raw = pack.spec.model_dump(mode="python", by_alias=True)
    archetypes: dict[str, Any] = {}
    population = []
    for override in overrides:
        policy = PolicyKind(override["policy"])
        visibility = str(override["visibility_profile"])
        if visibility not in raw["visibility_profiles"]:
            raise ValueError(f"unknown visibility profile: {visibility}")
        unknown_channels = set(override.get("scalarizer", {})) - set(
            raw["outcome_space"]["channels"]
        )
        if unknown_channels:
            raise ValueError(f"scalarizer references unknown channels: {sorted(unknown_channels)}")
        config = dict(override.get("policy_config") or {})
        allowed = _POLICY_FIELDS[policy]
        if policy is PolicyKind.DETERMINISTIC:
            preferences = config.get("preferences", [])
            if not isinstance(preferences, list) or not all(
                isinstance(item, str) and item in raw["actions"]["all"] for item in preferences
            ):
                raise ValueError("deterministic preferences must reference declared actions")
            unknown = set(config) - {"preferences"}
        else:
            unknown = set(config) - set(allowed)
        if unknown:
            raise ValueError(f"unsupported {policy.value} policy fields: {sorted(unknown)}")
        for name, value in config.items():
            if name == "preferences":
                continue
            if not isinstance(value, int | float):
                raise ValueError(f"policy field {name} must be numeric")
            minimum, maximum, _ = allowed[name]
            if minimum is not None and value < minimum:
                raise ValueError(f"policy field {name} is below its minimum")
            if maximum is not None and value > maximum:
                raise ValueError(f"policy field {name} is above its maximum")
        archetype_id = str(override["archetype_id"])
        archetype = {
            "policy": policy.value,
            "role": override.get("role", "agent"),
            "visibility_profile": visibility,
            "scalarizer": dict(override.get("scalarizer") or {}),
            "policy_config": config,
            "initial_state": override.get("initial_state"),
            "initial_resources": dict(override.get("initial_resources") or {}),
            "attributes": dict(override.get("attributes") or {}),
        }
        if policy is PolicyKind.LLM:
            archetype["llm"] = dict(override["llm"])
        archetypes[archetype_id] = archetype
        population.append({"archetype": archetype_id, "count": int(override["count"])})
    raw["archetypes"] = archetypes
    raw["population"] = population
    spec = IncentiveSpec.model_validate(raw)

    generated = []
    ranges = payload.get("composition_parameter_ranges") or {}
    by_id = {item["archetype_id"]: item for item in overrides}
    for parameter_id, bounds in ranges.items():
        parts = parameter_id.split(".")
        if len(parts) not in {3, 4} or parts[0] != "composition":
            raise ValueError(f"invalid composition parameter id: {parameter_id}")
        archetype_id, field = parts[1], ".".join(parts[2:])
        override = by_id.get(archetype_id)
        if override is None:
            raise ValueError(f"unknown composition archetype: {archetype_id}")
        if field == "count":
            target = ParameterTarget(
                entity=ParameterEntity.POPULATION,
                entity_id=archetype_id,
                field=["count"],
            )
            step = 1
            parameter_type = ParameterType.INTEGER
            default = int(override["count"])
        elif field.startswith("policy_config."):
            config_name = field.removeprefix("policy_config.")
            policy = PolicyKind(override["policy"])
            if config_name not in _POLICY_FIELDS[policy]:
                raise ValueError(f"policy field {config_name} cannot be optimized")
            step = _POLICY_FIELDS[policy][config_name][2]
            target = ParameterTarget(
                entity=ParameterEntity.ARCHETYPE,
                entity_id=archetype_id,
                field=["policy_config", config_name],
            )
            parameter_type = ParameterType.FLOAT
            default = float(override.get("policy_config", {}).get(config_name, 0.0))
        elif field == "llm.temperature" and PolicyKind(override["policy"]) is PolicyKind.LLM:
            target = ParameterTarget(
                entity=ParameterEntity.ARCHETYPE,
                entity_id=archetype_id,
                field=["llm", "temperature"],
            )
            step = 0.1
            parameter_type = ParameterType.FLOAT
            default = float(override["llm"].get("temperature", 0.0))
        else:
            raise ValueError(f"composition field {field} cannot be optimized")
        generated.append(
            GuidedParameter(
                id=parameter_id,
                label=parameter_id.replace(".", " "),
                description="Run-scoped population parameter.",
                type=parameter_type,
                default=default,
                minimum=bounds["minimum"],
                maximum=bounds["maximum"],
                step=step,
                optimizable=True,
                target=target,
            )
        )
    manifest = pack.manifest.model_copy(
        update={"parameters": [*pack.manifest.parameters, *generated]}
    )
    return replace(pack, spec=spec, manifest=manifest)


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


def _pagination(query: dict[str, list[str]], *, default_limit: int = 100) -> PaginationQuery:
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


def _artifact_context(
    root: Path, kind: str, artifact_id: str
) -> tuple[DomainPackManifest | None, IncentiveSpec | None]:
    directory = root / kind / artifact_id
    spec = None
    manifest = None
    spec_path = directory / "spec.json"
    if spec_path.exists():
        try:
            spec = IncentiveSpec.model_validate_json(spec_path.read_text())
        except (OSError, ValidationError):
            spec = None
    for name in ("domain_pack_manifest.json", "pack.json"):
        path = directory / name
        if not path.exists():
            continue
        try:
            manifest = DomainPackManifest.model_validate_json(path.read_text())
        except (OSError, ValidationError):
            manifest = None
        break
    return manifest, spec
