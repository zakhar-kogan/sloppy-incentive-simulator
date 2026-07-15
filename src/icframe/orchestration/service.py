from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icframe.artifacts import ArtifactLifecycle
from icframe.catalog import Catalog
from icframe.core import LoadedDomainPack, load_domain_pack
from icframe.core.compiler import runtime_hash
from icframe.domain.incentive_spec import DomainPackManifest, IncentiveSpec
from icframe.domain.run import (
    ExecutionProvenance,
    PlannerKind,
    RunConfig,
    RunStatus,
    StudyConfig,
    StudySummary,
    TrialRecord,
)
from icframe.orchestration.backends import (
    ExecutionBackend,
    LocalExecutionBackend,
    NebiusJobsBackend,
)
from icframe.orchestration.bundles import import_artifact_bundle, verify_artifact_bundle
from icframe.orchestration.models import (
    BackendJobRef,
    BackendJobState,
    JobHandle,
    RunShardRequest,
    StudyShardRequest,
    StudyShardResult,
    WorkerRequest,
)
from icframe.planning import StudyPlan, create_study_plan, pack_fingerprint
from icframe.profiles import ProfileRegistry, apply_llm_profile, load_profiles
from icframe.study import _best_trial, _pareto_trials


class JobCoordinator:
    """Durable local control plane for local and remote ICFRAME jobs."""

    def __init__(
        self,
        artifact_root: str | Path = ".artifacts/icframe",
        *,
        profiles: ProfileRegistry | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = self.artifact_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.profiles = profiles or load_profiles()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="icframe-orchestration",
            daemon=True,
        )
        self._thread.start()
        self._backends: dict[str, ExecutionBackend] = {}
        self._cancelled: set[str] = set()
        self._futures: dict[str, Future[Any]] = {}
        self.sync_jobs()

    def submit_run(
        self,
        pack: LoadedDomainPack | str | Path,
        config: RunConfig,
        *,
        backend_profile: str = "local",
        llm_profile: str | None = None,
    ) -> JobHandle:
        loaded = pack if isinstance(pack, LoadedDomainPack) else load_domain_pack(pack)
        self.profiles.execution_profile(backend_profile)
        if llm_profile:
            loaded = apply_llm_profile(loaded, self.profiles.llm_profile(llm_profile))
        run_id = config.run_id or f"run_{uuid.uuid4().hex[:12]}"
        request = RunShardRequest(
            logical_job_id=run_id,
            shard_id=run_id,
            pack_id=loaded.id,
            pack_hash=pack_fingerprint(loaded),
            runtime_hash=runtime_hash(loaded.spec, loaded.hook_hash),
            effective_manifest=loaded.manifest.model_dump(mode="json"),
            effective_spec=loaded.spec.model_dump(mode="json", by_alias=True),
            seed=config.seed,
            parameters=config.parameters,
            retention=config.retention or loaded.spec.observability.retention,
            sample_every_steps=config.sample_every_steps,
            llm_profile=llm_profile,
            llm_config=self._worker_llm_config(llm_profile),
        )
        handle = JobHandle(
            id=run_id,
            kind="run",
            backend_profile=backend_profile,
            artifact_root=str(self.artifact_root),
            planned_trials=1,
            shard_count=1,
        )
        directory = self._start_job(handle, request=request.model_dump(mode="json"))
        _write_json(directory / "request.json", request.model_dump(mode="json"))
        self._schedule(handle.id, self._execute_run(handle, request))
        return handle

    def submit_study(
        self,
        pack: LoadedDomainPack | str | Path,
        config: StudyConfig,
        *,
        backend_profile: str = "local",
        llm_profile: str | None = None,
    ) -> JobHandle:
        loaded = pack if isinstance(pack, LoadedDomainPack) else load_domain_pack(pack)
        profile = self.profiles.execution_profile(backend_profile)
        if config.planner not in {PlannerKind.MATRIX, PlannerKind.RANDOM}:
            raise ValueError("submitted studies require planner=matrix or planner=random")
        if config.live_llm.enabled and not llm_profile:
            raise ValueError("live LLM studies require an LLM profile")
        if llm_profile:
            loaded = apply_llm_profile(loaded, self.profiles.llm_profile(llm_profile))
        effective_config = config.model_copy(
            update={
                "study_id": config.study_id or f"study_{uuid.uuid4().hex[:12]}",
                "artifact_root": self.artifact_root,
            }
        )
        plan = create_study_plan(loaded, effective_config)
        shards = [plan.trials] if config.live_llm.enabled else plan.shard(profile.shard_size)
        handle = JobHandle(
            id=plan.study_id,
            kind="study",
            backend_profile=backend_profile,
            artifact_root=str(self.artifact_root),
            planned_trials=len(plan.trials),
            shard_count=len(shards),
        )
        directory = self._start_job(
            handle,
            request={
                "pack": loaded.id,
                "config": effective_config.model_dump(mode="json"),
                "llm_profile": llm_profile,
            },
        )
        _write_json(directory / "plan.json", plan.model_dump(mode="json"))
        _write_json(directory / "config.json", effective_config.model_dump(mode="json"))
        _write_json(
            directory / "effective-pack.json",
            loaded.manifest.model_dump(mode="json"),
        )
        _write_json(
            directory / "effective-spec.json",
            loaded.spec.model_dump(mode="json", by_alias=True),
        )
        study_dir = self.artifact_root / "studies" / plan.study_id
        ArtifactLifecycle.start(
            study_dir,
            {
                "study_id": plan.study_id,
                "pack_id": loaded.id,
                "mode": effective_config.mode.value,
                "planner": plan.planner.value,
                "planner_seed": plan.planner_seed,
                "objectives": effective_config.objectives,
                "parameters": effective_config.parameters,
                "seeds": effective_config.seeds,
                "plan_hash": plan.canonical_hash,
                "planned_trials": len(plan.trials),
                "backend_profile": backend_profile,
            },
            files={
                "spec.json": loaded.spec.model_dump(mode="json", by_alias=True),
                "pack.json": loaded.manifest.model_dump(mode="json"),
                "plan.json": plan.model_dump(mode="json"),
            },
        )
        self._schedule(
            handle.id,
            self._execute_study(handle, loaded, effective_config, plan, llm_profile),
        )
        return handle

    def get_job(self, job_id: str) -> JobHandle | None:
        path = self.jobs_root / job_id / "manifest.json"
        return JobHandle.model_validate_json(path.read_text()) if path.exists() else None

    def cancel_job(self, job_id: str) -> JobHandle | None:
        handle = self.get_job(job_id)
        if handle is None:
            return None
        self._cancelled.add(job_id)
        self._schedule(f"cancel:{job_id}", self._cancel_remote_jobs(handle))
        return self._set_handle(handle, cancel_requested=True)

    def sync_jobs(self) -> list[JobHandle]:
        handles = []
        for path in sorted(self.jobs_root.glob("*/manifest.json")):
            try:
                handle = JobHandle.model_validate_json(path.read_text())
            except (OSError, ValueError):
                continue
            handles.append(handle)
            if handle.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                continue
            profile = self.profiles.execution.get(handle.backend_profile)
            if profile is None or profile.type == "local":
                self._set_handle(
                    handle,
                    status=RunStatus.INTERRUPTED,
                    error="local execution was interrupted by coordinator restart",
                )
                continue
            if handle.id not in self._futures:
                self._schedule(handle.id, self._resume(handle))
        return handles

    def close(self) -> None:
        for future in self._futures.values():
            if not future.done():
                future.cancel()
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if close is not None:
                asyncio.run_coroutine_threadsafe(close(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)

    async def _execute_run(self, handle: JobHandle, request: RunShardRequest) -> None:
        started = time.perf_counter()
        try:
            self._set_handle(handle, status=RunStatus.RUNNING)
            bundle, refs = await self._execute_request(handle, request)
            verify_artifact_bundle(bundle)
            self._set_handle(handle, artifact_import_state="importing")
            validated = self._import_run_result(handle, request, bundle, refs)
            self._set_handle(
                handle,
                status=validated.status,
                completed_trials=1,
                remote_job_ids=validated.execution.remote_job_ids,
                artifact_import_state="imported",
            )
            self._event(handle.id, "completed", duration_seconds=time.perf_counter() - started)
        except asyncio.CancelledError:
            self._set_handle(handle, status=RunStatus.CANCELLED)
        except Exception as exc:
            self._set_handle(
                handle,
                status=RunStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _execute_study(
        self,
        handle: JobHandle,
        pack: LoadedDomainPack,
        config: StudyConfig,
        plan: StudyPlan,
        llm_profile: str | None,
    ) -> None:
        started = time.perf_counter()
        profile = self.profiles.execution_profile(handle.backend_profile)
        semaphore = asyncio.Semaphore(1 if config.live_llm.enabled else profile.max_in_flight)

        async def execute(index: int, trials) -> tuple[StudyShardResult, list[BackendJobRef]]:
            shard_id = f"{plan.study_id}-shard-{index:04d}"
            request = StudyShardRequest(
                logical_job_id=plan.study_id,
                shard_id=shard_id,
                plan_hash=plan.canonical_hash,
                pack_id=pack.id,
                pack_hash=plan.pack_hash,
                runtime_hash=plan.runtime_hash,
                effective_manifest=pack.manifest.model_dump(mode="json"),
                effective_spec=pack.spec.model_dump(mode="json", by_alias=True),
                trials=trials,
                llm_profile=llm_profile,
                llm_config=self._worker_llm_config(llm_profile),
                live_llm=config.live_llm,
            )
            request_path = self.jobs_root / handle.id / "shards" / shard_id / "request.json"
            _write_json(request_path, request.model_dump(mode="json"))
            collected = request_path.with_name("collected")
            if (collected / "result.json").exists():
                result = StudyShardResult.model_validate_json(
                    (collected / "result.json").read_text()
                )
                return result, []
            async with semaphore:
                bundle, refs = await self._execute_request(handle, request)
            verify_artifact_bundle(bundle)
            self._set_handle(handle, artifact_import_state="importing")
            import_artifact_bundle(bundle, collected, expected_logical_id=shard_id)
            result = StudyShardResult.model_validate_json((collected / "result.json").read_text())
            if result.plan_hash != plan.canonical_hash:
                raise ValueError("collected shard belongs to a different study plan")
            return result, refs

        try:
            self._set_handle(handle, status=RunStatus.RUNNING)
            results = await asyncio.gather(
                *[
                    execute(index, trials)
                    for index, trials in enumerate(
                        [plan.trials]
                        if config.live_llm.enabled
                        else plan.shard(profile.shard_size)
                    )
                ]
            )
            records: dict[int, TrialRecord] = {}
            refs: list[BackendJobRef] = []
            for result, shard_refs in results:
                refs.extend(shard_refs)
                for record in result.trials:
                    if record.number in records and records[record.number] != record:
                        raise ValueError(f"conflicting results for trial {record.number}")
                    records[record.number] = record
            ordered = [records[number] for number in sorted(records)]
            retained_ids, retained_refs = await self._execute_retained_runs(
                handle,
                pack,
                config,
                ordered,
                llm_profile,
            )
            refs.extend(retained_refs)
            self._finalize_study(
                handle,
                pack,
                config,
                plan,
                ordered,
                refs,
                retained_ids,
                started,
            )
        except asyncio.CancelledError:
            self._finalize_cancelled_study(handle, config, plan)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._finalize_failed_study(handle, config, plan, error)

    async def _execute_request(
        self,
        handle: JobHandle,
        request: WorkerRequest,
    ) -> tuple[Path, list[BackendJobRef]]:
        profile = self.profiles.execution_profile(handle.backend_profile)
        backend = self._backend(handle.backend_profile)
        directory = self.jobs_root / handle.id / "shards" / request.shard_id
        directory.mkdir(parents=True, exist_ok=True)
        refs: list[BackendJobRef] = []
        for attempt in range(1, profile.max_attempts + 1):
            if handle.id in self._cancelled:
                raise asyncio.CancelledError
            ref_path = directory / f"attempt-{attempt}.json"
            if ref_path.exists():
                ref = BackendJobRef.model_validate_json(ref_path.read_text())
            else:
                try:
                    ref = await backend.submit(request, attempt=attempt)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt == profile.max_attempts or not _retryable(error):
                        raise
                    self._record_retry(handle, request.shard_id, attempt + 1, error)
                    continue
                _write_json(ref_path, ref.model_dump(mode="json"))
                current = self.get_job(handle.id) or handle
                remote_ids = list(current.remote_job_ids)
                if ref.backend != "local" and ref.id not in remote_ids:
                    remote_ids.append(ref.id)
                    self._set_handle(handle, remote_job_ids=remote_ids)
                self._event(handle.id, "submitted", shard=request.shard_id, provider_job=ref.id)
            refs.append(ref)
            while True:
                if handle.id in self._cancelled:
                    await backend.cancel(ref)
                    raise asyncio.CancelledError
                ref = await backend.inspect(ref)
                _write_json(ref_path, ref.model_dump(mode="json"))
                if ref.state in {
                    BackendJobState.COMPLETED,
                    BackendJobState.FAILED,
                    BackendJobState.CANCELLED,
                }:
                    break
                await asyncio.sleep(profile.poll_seconds if ref.backend != "local" else 0.05)
            if ref.state is BackendJobState.COMPLETED:
                bundle = directory / f"attempt-{attempt}.tar.gz"
                try:
                    return await backend.collect(ref, bundle), refs
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt == profile.max_attempts or not _retryable(error):
                        raise
                    self._record_retry(handle, request.shard_id, attempt + 1, error)
                    continue
            if ref.state is BackendJobState.CANCELLED:
                raise asyncio.CancelledError
            if attempt == profile.max_attempts or not _retryable(ref.error):
                raise RuntimeError(ref.error or f"shard {request.shard_id} failed")
            self._record_retry(handle, request.shard_id, attempt + 1, ref.error)
        raise RuntimeError(f"shard {request.shard_id} exhausted retries")

    def _finalize_study(
        self,
        handle: JobHandle,
        pack: LoadedDomainPack,
        config: StudyConfig,
        plan: StudyPlan,
        records: list[TrialRecord],
        refs: list[BackendJobRef],
        retained_run_ids: list[str],
        started: float,
    ) -> None:
        study_dir = self.artifact_root / "studies" / plan.study_id
        (study_dir / "trials.jsonl").write_text(
            "".join(f"{record.model_dump_json()}\n" for record in records)
        )
        feasible = [record for record in records if record.feasible and record.state == "complete"]
        best_trial = _best_trial(pack, config, feasible)
        pareto_trials = _pareto_trials(pack, config, feasible)
        remote_ids = list((self.get_job(handle.id) or handle).remote_job_ids)
        for ref in refs:
            if ref.backend != "local" and ref.id not in remote_ids:
                remote_ids.append(ref.id)
        artifacts = {
            "manifest": str(study_dir / "manifest.json"),
            "summary": str(study_dir / "summary.json"),
            "trials": str(study_dir / "trials.jsonl"),
            "spec": str(study_dir / "spec.json"),
            "pack": str(study_dir / "pack.json"),
            "plan": str(study_dir / "plan.json"),
        }
        summary = StudySummary(
            study_id=plan.study_id,
            pack_id=pack.id,
            mode=config.mode,
            status=RunStatus.COMPLETED,
            objectives=config.objectives,
            parameters=config.parameters,
            seeds=config.seeds,
            trial_count=len(records),
            trials=records[:200],
            best_trial=best_trial,
            pareto_trials=pareto_trials[:200],
            retained_run_ids=retained_run_ids,
            duration_seconds=time.perf_counter() - started,
            artifacts=artifacts,
            llm_calls=sum(record.llm_calls for record in records),
            estimated_llm_cost_usd=sum(
                record.estimated_llm_cost_usd or 0.0 for record in records
            ),
            execution=ExecutionProvenance(
                backend=self.profiles.execution_profile(handle.backend_profile).type,
                backend_profile=handle.backend_profile,
                planner=plan.planner.value,
                planned_trials=len(plan.trials),
                completed_trials=len(records),
                shard_count=(
                    1
                    if config.live_llm.enabled
                    else math.ceil(
                        len(plan.trials)
                        / self.profiles.execution_profile(handle.backend_profile).shard_size
                    )
                ),
                remote_job_ids=remote_ids,
                retry_count=(self.get_job(handle.id) or handle).retry_count,
                artifact_import_state="imported",
            ),
        )
        _write_json(study_dir / "summary.json", summary.model_dump(mode="json"))
        ArtifactLifecycle(study_dir / "manifest.json").complete(
            trial_count=len(records),
            best_trial=best_trial,
            pareto_trials=pareto_trials,
            retained_run_ids=retained_run_ids,
            artifacts=artifacts,
        )
        catalog = Catalog(self.artifact_root)
        catalog.upsert_study(summary)
        catalog.replace_trials(plan.study_id, records)
        self._set_handle(
            handle,
            status=RunStatus.COMPLETED,
            completed_trials=len(records),
            remote_job_ids=remote_ids,
            artifact_import_state="imported",
        )

    def _finalize_cancelled_study(
        self,
        handle: JobHandle,
        config: StudyConfig,
        plan: StudyPlan,
    ) -> None:
        study_dir = self.artifact_root / "studies" / plan.study_id
        records = []
        for path in sorted((self.jobs_root / handle.id / "shards").glob("*/collected/result.json")):
            records.extend(StudyShardResult.model_validate_json(path.read_text()).trials)
        records = sorted(
            {record.number: record for record in records}.values(),
            key=lambda item: item.number,
        )
        (study_dir / "trials.jsonl").write_text(
            "".join(f"{record.model_dump_json()}\n" for record in records)
        )
        artifacts = {
            "manifest": str(study_dir / "manifest.json"),
            "summary": str(study_dir / "summary.json"),
            "trials": str(study_dir / "trials.jsonl"),
            "spec": str(study_dir / "spec.json"),
            "pack": str(study_dir / "pack.json"),
            "plan": str(study_dir / "plan.json"),
        }
        summary = StudySummary(
            study_id=plan.study_id,
            pack_id=plan.pack_id,
            mode=config.mode,
            status=RunStatus.CANCELLED,
            objectives=config.objectives,
            parameters=config.parameters,
            seeds=config.seeds,
            trial_count=len(records),
            trials=records[:200],
            llm_calls=sum(record.llm_calls for record in records),
            estimated_llm_cost_usd=sum(
                record.estimated_llm_cost_usd or 0.0 for record in records
            ),
            artifacts=artifacts,
            execution=ExecutionProvenance(
                backend=self.profiles.execution_profile(handle.backend_profile).type,
                backend_profile=handle.backend_profile,
                planner=plan.planner.value,
                planned_trials=len(plan.trials),
                completed_trials=len(records),
                shard_count=handle.shard_count,
                retry_count=(self.get_job(handle.id) or handle).retry_count,
                artifact_import_state="partial",
                remote_job_ids=(self.get_job(handle.id) or handle).remote_job_ids,
            ),
        )
        _write_json(study_dir / "summary.json", summary.model_dump(mode="json"))
        ArtifactLifecycle(study_dir / "manifest.json").cancel(
            trial_count=len(records),
            artifacts=artifacts,
        )
        catalog = Catalog(self.artifact_root)
        catalog.upsert_study(summary)
        catalog.replace_trials(plan.study_id, records)
        self._set_handle(
            handle,
            status=RunStatus.CANCELLED,
            completed_trials=len(records),
            artifact_import_state="partial",
        )

    def _finalize_failed_study(
        self,
        handle: JobHandle,
        config: StudyConfig,
        plan: StudyPlan,
        error: str,
    ) -> None:
        study_dir = self.artifact_root / "studies" / plan.study_id
        records = []
        for path in sorted((self.jobs_root / handle.id / "shards").glob("*/collected/result.json")):
            records.extend(StudyShardResult.model_validate_json(path.read_text()).trials)
        records = sorted(
            {record.number: record for record in records}.values(),
            key=lambda item: item.number,
        )
        (study_dir / "trials.jsonl").write_text(
            "".join(f"{record.model_dump_json()}\n" for record in records)
        )
        artifacts = {
            "manifest": str(study_dir / "manifest.json"),
            "summary": str(study_dir / "summary.json"),
            "trials": str(study_dir / "trials.jsonl"),
            "spec": str(study_dir / "spec.json"),
            "pack": str(study_dir / "pack.json"),
            "plan": str(study_dir / "plan.json"),
        }
        current = self.get_job(handle.id) or handle
        summary = StudySummary(
            study_id=plan.study_id,
            pack_id=plan.pack_id,
            mode=config.mode,
            status=RunStatus.FAILED,
            objectives=config.objectives,
            parameters=config.parameters,
            seeds=config.seeds,
            trial_count=len(records),
            trials=records[:200],
            error=error,
            artifacts=artifacts,
            llm_calls=sum(record.llm_calls for record in records),
            estimated_llm_cost_usd=sum(
                record.estimated_llm_cost_usd or 0.0 for record in records
            ),
            execution=ExecutionProvenance(
                backend=self.profiles.execution_profile(handle.backend_profile).type,
                backend_profile=handle.backend_profile,
                planner=plan.planner.value,
                planned_trials=len(plan.trials),
                completed_trials=len(records),
                shard_count=handle.shard_count,
                remote_job_ids=current.remote_job_ids,
                retry_count=current.retry_count,
                artifact_import_state="partial",
            ),
        )
        _write_json(study_dir / "summary.json", summary.model_dump(mode="json"))
        ArtifactLifecycle(study_dir / "manifest.json").fail(
            error,
            trial_count=len(records),
            artifacts=artifacts,
        )
        catalog = Catalog(self.artifact_root)
        catalog.upsert_study(summary)
        catalog.replace_trials(plan.study_id, records)
        self._set_handle(
            handle,
            status=RunStatus.FAILED,
            error=error,
            completed_trials=len(records),
            artifact_import_state="partial",
        )

    async def _execute_retained_runs(
        self,
        handle: JobHandle,
        pack: LoadedDomainPack,
        config: StudyConfig,
        records: list[TrialRecord],
        llm_profile: str | None,
    ) -> tuple[list[str], list[BackendJobRef]]:
        feasible = [record for record in records if record.feasible and record.state == "complete"]
        winner_number = _best_trial(pack, config, feasible)
        winner = next(
            (record for record in feasible if record.number == winner_number),
            None,
        )
        variants: list[tuple[str, dict[str, Any]]] = [("baseline", {})]
        if config.mode.value == "single" and winner is not None:
            variants.append(("winner", dict(winner.parameters)))
        retained: list[str] = []
        refs: list[BackendJobRef] = []
        used_calls = sum(record.llm_calls for record in records)
        used_cost = sum(record.estimated_llm_cost_usd or 0.0 for record in records)
        for label, parameters in variants:
            for seed_index, seed in enumerate(config.seeds):
                if handle.id in self._cancelled:
                    raise asyncio.CancelledError
                budget = config.live_llm
                if budget.enabled:
                    remaining_calls = int(budget.max_calls) - used_calls
                    remaining_cost = float(budget.max_cost_usd) - used_cost
                    if remaining_calls < 1 or remaining_cost <= 0:
                        return retained, refs
                    budget = budget.model_copy(
                        update={
                            "max_calls": remaining_calls,
                            "max_cost_usd": remaining_cost,
                        }
                    )
                run_id = f"{handle.id}-{label}-{seed_index:03d}"
                request = RunShardRequest(
                    logical_job_id=run_id,
                    shard_id=f"retained-{label}-{seed_index:03d}",
                    pack_id=pack.id,
                    pack_hash=pack_fingerprint(pack),
                    runtime_hash=runtime_hash(pack.spec, pack.hook_hash),
                    effective_manifest=pack.manifest.model_dump(mode="json"),
                    effective_spec=pack.spec.model_dump(mode="json", by_alias=True),
                    seed=seed,
                    parameters=parameters,
                    llm_profile=llm_profile,
                    llm_config=self._worker_llm_config(llm_profile),
                    live_llm=budget,
                )
                run_dir = self.artifact_root / "runs" / run_id
                if (run_dir / "summary.json").exists():
                    retained.append(run_id)
                    continue
                bundle, run_refs = await self._execute_request(handle, request)
                verify_artifact_bundle(bundle)
                summary = self._import_run_result(handle, request, bundle, run_refs)
                refs.extend(run_refs)
                retained.append(run_id)
                used_calls += summary.llm_calls
                used_cost += summary.estimated_llm_cost_usd or 0.0
        return retained, refs

    def _import_run_result(
        self,
        handle: JobHandle,
        request: RunShardRequest,
        bundle: Path,
        refs: list[BackendJobRef],
    ):
        from icframe.domain.run import RunSummary

        run_dir = self.artifact_root / "runs" / request.logical_job_id
        if not run_dir.exists():
            import_artifact_bundle(bundle, run_dir, expected_logical_id=request.shard_id)
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["artifacts"] = _artifact_paths(run_dir)
        summary["duration_seconds"] = float(summary.get("duration_seconds", 0.0))
        summary["execution"] = {
            "backend": self.profiles.execution_profile(handle.backend_profile).type,
            "backend_profile": handle.backend_profile,
            "planned_trials": 1,
            "completed_trials": 1,
            "shard_count": 1,
            "remote_job_ids": [ref.id for ref in refs if ref.backend != "local"],
            "retry_count": (self.get_job(handle.id) or handle).retry_count,
            "artifact_import_state": "imported",
        }
        _write_json(summary_path, summary)
        validated = RunSummary.model_validate_json(json.dumps(summary))
        Catalog(self.artifact_root).upsert_run(validated)
        return validated

    async def _cancel_remote_jobs(self, handle: JobHandle) -> None:
        backend = self._backend(handle.backend_profile)
        for path in (self.jobs_root / handle.id / "shards").glob("*/attempt-*.json"):
            ref = BackendJobRef.model_validate_json(path.read_text())
            if ref.state in {BackendJobState.QUEUED, BackendJobState.RUNNING}:
                try:
                    await backend.cancel(ref)
                except Exception:
                    continue

    async def _resume(self, handle: JobHandle) -> None:
        request_path = self.jobs_root / handle.id / "request.json"
        if handle.kind == "run":
            request = RunShardRequest.model_validate_json(request_path.read_text())
            await self._execute_run(handle, request)
            return
        payload = json.loads(request_path.read_text())
        pack = load_domain_pack(payload["pack"])
        directory = self.jobs_root / handle.id
        pack = replace(
            pack,
            manifest=DomainPackManifest.model_validate_json(
                (directory / "effective-pack.json").read_text()
            ),
            spec=IncentiveSpec.model_validate_json(
                (directory / "effective-spec.json").read_text()
            ),
        )
        config = StudyConfig.model_validate_json(
            (self.jobs_root / handle.id / "config.json").read_text()
        )
        plan = StudyPlan.model_validate_json((self.jobs_root / handle.id / "plan.json").read_text())
        await self._execute_study(handle, pack, config, plan, payload.get("llm_profile"))

    def _backend(self, name: str) -> ExecutionBackend:
        if name in self._backends:
            return self._backends[name]
        profile = self.profiles.execution_profile(name)
        backend: ExecutionBackend
        if profile.type == "local":
            backend = LocalExecutionBackend(self.jobs_root / ".local-exchange")
        else:
            backend = NebiusJobsBackend(profile, llm_profiles=self.profiles.llm)
        self._backends[name] = backend
        return backend

    def _worker_llm_config(self, name: str | None) -> dict[str, object] | None:
        if not name:
            return None
        payload = self.profiles.llm_profile(name).model_dump(mode="json")
        payload.pop("remote_secret", None)
        return payload

    def _start_job(self, handle: JobHandle, *, request: dict[str, object]) -> Path:
        directory = self.jobs_root / handle.id
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory / "manifest.json", handle.model_dump(mode="json"))
        _write_json(directory / "request.json", request)
        self._event(handle.id, "created", backend_profile=handle.backend_profile)
        return directory

    def _set_handle(self, handle: JobHandle, **updates: object) -> JobHandle:
        current = self.get_job(handle.id) or handle
        updated = current.model_copy(update=updates)
        _write_json(self.jobs_root / handle.id / "manifest.json", updated.model_dump(mode="json"))
        return updated

    def _event(self, job_id: str, event: str, **fields: object) -> None:
        payload = {
            "time": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        path = self.jobs_root / job_id / "orchestration.jsonl"
        with path.open("a") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def _record_retry(
        self,
        handle: JobHandle,
        shard_id: str,
        attempt: int,
        error: str | None,
    ) -> None:
        self._event(handle.id, "retry", shard=shard_id, attempt=attempt, error=error)
        current = self.get_job(handle.id) or handle
        self._set_handle(handle, retry_count=current.retry_count + 1)

    def _schedule(self, key: str, coroutine) -> None:
        self._futures[key] = asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


_COORDINATORS: dict[Path, JobCoordinator] = {}
_COORDINATORS_LOCK = threading.Lock()


def _coordinator(artifact_root: str | Path) -> JobCoordinator:
    root = Path(artifact_root).resolve()
    with _COORDINATORS_LOCK:
        if root not in _COORDINATORS:
            _COORDINATORS[root] = JobCoordinator(root)
        return _COORDINATORS[root]


def submit_run(
    pack: LoadedDomainPack | str | Path,
    config: RunConfig,
    *,
    backend_profile: str = "local",
    llm_profile: str | None = None,
) -> JobHandle:
    return _coordinator(config.artifact_root).submit_run(
        pack,
        config,
        backend_profile=backend_profile,
        llm_profile=llm_profile,
    )


def submit_study(
    pack: LoadedDomainPack | str | Path,
    config: StudyConfig,
    *,
    backend_profile: str = "local",
    llm_profile: str | None = None,
) -> JobHandle:
    return _coordinator(config.artifact_root).submit_study(
        pack,
        config,
        backend_profile=backend_profile,
        llm_profile=llm_profile,
    )


def get_job(job_id: str, *, artifact_root: str | Path = ".artifacts/icframe") -> JobHandle | None:
    return _coordinator(artifact_root).get_job(job_id)


def cancel_job(
    job_id: str,
    *,
    artifact_root: str | Path = ".artifacts/icframe",
) -> JobHandle | None:
    return _coordinator(artifact_root).cancel_job(job_id)


def sync_jobs(*, artifact_root: str | Path = ".artifacts/icframe") -> list[JobHandle]:
    return _coordinator(artifact_root).sync_jobs()


def _retryable(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    terminal = ("auth", "permission", "invalid", "schema", "checksum", "pack hash")
    if any(value in lowered for value in terminal):
        return False
    transient = (
        "transient",
        "preempt",
        "transport",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "service unavailable",
        "internal infrastructure",
    )
    return any(value in lowered for value in transient)


def _artifact_paths(directory: Path) -> dict[str, str]:
    paths = {
        "manifest": str(directory / "manifest.json"),
        "spec": str(directory / "spec.json"),
        "domain_pack_manifest": str(directory / "domain_pack_manifest.json"),
        "summary": str(directory / "summary.json"),
    }
    for path in sorted(directory.glob("*.jsonl")):
        paths[path.stem] = str(path)
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
