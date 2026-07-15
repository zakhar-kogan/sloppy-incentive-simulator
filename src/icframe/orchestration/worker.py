from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from icframe.core import load_domain_pack, run_experiment
from icframe.core.compiler import runtime_hash
from icframe.domain.incentive_spec import DomainPackManifest, IncentiveSpec
from icframe.domain.run import RunConfig, RunStatus, TrialRecord
from icframe.orchestration.bundles import create_artifact_bundle, write_completion_marker
from icframe.orchestration.models import (
    RunShardRequest,
    RunShardResult,
    StudyShardRequest,
    StudyShardResult,
    WorkerRequest,
)
from icframe.planning import pack_fingerprint
from icframe.profiles import LLMProfile, llm_client_for_profile, load_profiles
from icframe.study import _BudgetedLLMClient, evaluate_trial


def execute_study_shard(
    request: StudyShardRequest,
    output_dir: str | Path,
) -> Path:
    pack = _request_pack(request)
    if pack_fingerprint(pack) != request.pack_hash:
        raise ValueError("worker domain pack hash does not match the submitted study plan")
    llm_client = None
    if request.llm_profile:
        llm_client = llm_client_for_profile(_request_llm_profile(request))
    if request.live_llm.enabled:
        if llm_client is None:
            raise ValueError("live LLM shard requires an LLM profile")
        llm_client = _BudgetedLLMClient(
            llm_client,
            max_calls=int(request.live_llm.max_calls),
            max_cost_usd=float(request.live_llm.max_cost_usd),
        )

    records = []
    for trial in request.trials:
        try:
            record = evaluate_trial(
                pack,
                trial.number,
                trial.parameters,
                trial.seeds,
                trial.objectives,
                llm_client,
            )
        except Exception as exc:
            record = TrialRecord(
                number=trial.number,
                parameters=trial.parameters,
                seeds=[],
                objective_values={},
                feasible=False,
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(record)

    result = StudyShardResult(
        logical_job_id=request.logical_job_id,
        shard_id=request.shard_id,
        plan_hash=request.plan_hash,
        status=RunStatus.COMPLETED,
        trials=records,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "artifact.tar.gz"
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        (stage / "request.json").write_text(request.model_dump_json(indent=2))
        (stage / "result.json").write_text(result.model_dump_json(indent=2))
        (stage / "trials.jsonl").write_text(
            "".join(f"{record.model_dump_json()}\n" for record in records)
        )
        create_artifact_bundle(stage, bundle, logical_id=request.shard_id)
    write_completion_marker(bundle, output / "complete.json")
    return bundle


def execute_run_shard(request: RunShardRequest, output_dir: str | Path) -> Path:
    pack = _request_pack(request)
    if pack_fingerprint(pack) != request.pack_hash:
        raise ValueError("worker domain pack hash does not match the submitted run")
    llm_client = None
    if request.llm_profile:
        llm_client = llm_client_for_profile(_request_llm_profile(request))
    if request.live_llm.enabled:
        if llm_client is None:
            raise ValueError("live LLM run requires an LLM profile")
        llm_client = _BudgetedLLMClient(
            llm_client,
            max_calls=int(request.live_llm.max_calls),
            max_cost_usd=float(request.live_llm.max_cost_usd),
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        artifact_root = Path(temporary) / "artifacts"
        summary = run_experiment(
            pack,
            RunConfig(
                run_id=request.logical_job_id,
                seed=request.seed,
                parameters=request.parameters,
                retention=request.retention,
                sample_every_steps=request.sample_every_steps,
                artifact_root=artifact_root,
            ),
            llm_client=llm_client,
        )
        run_dir = artifact_root / "runs" / summary.run_id
        result = RunShardResult(
            logical_job_id=request.logical_job_id,
            shard_id=request.shard_id,
            status=summary.status,
            summary=summary,
        )
        (run_dir / "worker-result.json").write_text(result.model_dump_json(indent=2))
        bundle = create_artifact_bundle(
            run_dir,
            output / "artifact.tar.gz",
            logical_id=request.shard_id,
        )
    write_completion_marker(bundle, output / "complete.json")
    return bundle


def execute_worker_request(request: WorkerRequest, output_dir: str | Path) -> Path:
    if isinstance(request, RunShardRequest):
        return execute_run_shard(request, output_dir)
    return execute_study_shard(request, output_dir)


def execute_study_shard_file(
    request_path: str | Path,
    output_dir: str | Path,
) -> Path:
    payload = json.loads(Path(request_path).read_text())
    request: WorkerRequest = (
        RunShardRequest.model_validate(payload)
        if payload.get("kind") == "run"
        else StudyShardRequest.model_validate(payload)
    )
    output = Path(output_dir)
    temporary = output.with_name(f".{output.name}.working")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        bundle = execute_worker_request(request, temporary)
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
        return output / bundle.name
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def completion_payload(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _request_pack(request: WorkerRequest):
    pack = load_domain_pack(request.pack_id)
    if request.effective_manifest is not None:
        pack = replace(
            pack,
            manifest=DomainPackManifest.model_validate_json(
                json.dumps(request.effective_manifest)
            ),
        )
    if request.effective_spec is not None:
        pack = replace(
            pack,
            spec=IncentiveSpec.model_validate_json(json.dumps(request.effective_spec)),
        )
    if pack_fingerprint(pack) != request.pack_hash:
        raise ValueError("worker domain pack hash does not match the submitted request")
    if runtime_hash(pack.spec, pack.hook_hash) != request.runtime_hash:
        raise ValueError("worker runtime hash does not match the submitted request")
    return pack


def _request_llm_profile(request: WorkerRequest) -> LLMProfile:
    if request.llm_config is not None:
        return LLMProfile.model_validate_json(json.dumps(request.llm_config))
    return load_profiles().llm_profile(str(request.llm_profile))
