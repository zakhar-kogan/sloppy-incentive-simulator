from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from icframe.core.compiler import compile_runtime
from icframe.core.engine import RuntimeEngine
from icframe.core.observer import NoopObserver
from icframe.core.packs import load_domain_pack
from icframe.domain.incentive_spec import IncentiveSpec, PolicyKind
from icframe.domain.run import RunSummary
from icframe.llm import _RecordedLLMClient
from icframe.version import __version__


def replay_run(
    run: str | Path,
    *,
    artifact_root: str | Path = ".artifacts/icframe",
) -> RunSummary:
    run_dir = _resolve_run_dir(run, Path(artifact_root))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    original = RunSummary.model_validate_json((run_dir / "summary.json").read_text())
    spec = IncentiveSpec.model_validate_json((run_dir / "spec.json").read_text())
    if manifest.get("runtime_version") != __version__:
        raise RuntimeError("runtime version changed; exact replay is unavailable")
    pack_source = manifest.get("pack_path")
    if pack_source and Path(pack_source).exists():
        pack = load_domain_pack(pack_source)
    else:
        pack = load_domain_pack(manifest["pack_id"])
    if pack.hook_hash != manifest["hook_hash"]:
        raise RuntimeError("domain hook hash changed; exact replay is unavailable")
    effective = replace(pack, spec=spec)
    plan = compile_runtime(effective)
    if plan.runtime_hash != manifest["runtime_hash"]:
        raise RuntimeError("compiled runtime changed; exact replay is unavailable")
    if plan.trusted_evaluation_hash != manifest["trusted_evaluation_hash"]:
        raise RuntimeError("trusted evaluation changed; exact replay is unavailable")
    llm_client = (
        _RecordedLLMClient(run_dir / "llm_calls.jsonl")
        if (run_dir / "llm_calls.jsonl").exists()
        else None
    )
    engine = RuntimeEngine(
        plan,
        run_id=original.run_id,
        seed=original.seed,
        llm_client=llm_client,
        observer=NoopObserver(),
        retention=original.retention,
        sample_every_steps=manifest.get("sample_every_steps"),
        parameters=original.parameters,
    )
    started = time.perf_counter()
    external_path = run_dir / "external_actions.jsonl"
    if external_path.exists():
        for agent in engine.world.agents.values():
            agent.policy_kind = PolicyKind.EXTERNAL
            engine.policies[agent.id] = None
        actions_by_step: dict[int, dict[str, tuple[str, str | None]]] = {}
        for line in external_path.read_text().splitlines():
            payload = json.loads(line)
            actions_by_step.setdefault(int(payload["step"]), {})[payload["agent_id"]] = (
                payload["action"],
                payload.get("target_id"),
            )
        for step in range(1, original.steps_completed + 1):
            if step not in actions_by_step:
                raise RuntimeError(f"replay data is missing external actions for step {step}")
            engine.step_external(actions_by_step[step])
    else:
        for _ in range(original.steps_completed):
            engine.step_internal()
    replayed = engine.summary(
        status=original.status,
        duration_seconds=time.perf_counter() - started,
    )
    if _deterministic_payload(replayed) != _deterministic_payload(original):
        raise RuntimeError("replay diverged from the persisted run summary")
    return replayed


def _deterministic_payload(summary: RunSummary) -> dict[str, object]:
    """Return every deterministic result field promised by exact replay."""
    return summary.model_dump(
        mode="json",
        exclude={
            "status",
            "duration_seconds",
            "error",
            "artifacts",
        },
    )


def _resolve_run_dir(run: str | Path, root: Path) -> Path:
    requested = Path(run)
    if requested.exists():
        return requested
    path = root / "runs" / str(run)
    if not path.exists():
        raise FileNotFoundError(f"run artifacts not found: {run}")
    return path
