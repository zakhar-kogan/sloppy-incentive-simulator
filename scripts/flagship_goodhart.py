from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from icframe import PlannerKind, StudyConfig, load_domain_pack, run_study, submit_study
from icframe.catalog import Catalog
from icframe.domain.run import RunStatus, StudyMode
from icframe.orchestration import get_job
from icframe.planning import StudyPlan, apply_study_preset


def config(root: Path, study_id: str) -> StudyConfig:
    pack = load_domain_pack("software_organization")
    preset = next(item for item in pack.manifest.study.presets if item.id == "goodhart_audit")
    return StudyConfig(
        study_id=study_id,
        mode=StudyMode.PARETO,
        objectives=list(preset.objectives),
        parameters=list(preset.parameter_matrix),
        seeds=list(preset.seeds),
        artifact_root=root,
        planner=PlannerKind.MATRIX,
        parameter_matrix=dict(preset.parameter_matrix),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-profile",
        choices=["local", "nebius", "both"],
        default="local",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path(".artifacts/icframe"))
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    pack = load_domain_pack("software_organization")
    preset = next(item for item in pack.manifest.study.presets if item.id == "goodhart_audit")
    pack = apply_study_preset(pack, preset)
    def execute(profile: str):
        study_id = f"flagship-{profile}-{int(time.time())}"
        study_config = config(args.artifact_root, study_id)
        if profile == "local":
            return run_study(pack, study_config)
        handle = submit_study(
            pack,
            study_config,
            backend_profile=profile,
        )
        while handle.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
            time.sleep(args.poll_seconds)
            handle = get_job(handle.id, artifact_root=args.artifact_root)
            if handle is None:
                raise RuntimeError("job manifest disappeared")
        if handle.status is not RunStatus.COMPLETED:
            raise RuntimeError(handle.error or f"job ended as {handle.status.value}")
        summary = Catalog(args.artifact_root).get_study(handle.id)
        if summary is None:
            raise RuntimeError("completed study was not imported into the catalog")
        return summary

    if args.execution_profile == "both":
        local = execute("local")
        remote = execute("nebius")
        local_plan = StudyPlan.model_validate_json(Path(local.artifacts["plan"]).read_text())
        remote_plan = StudyPlan.model_validate_json(Path(remote.artifacts["plan"]).read_text())
        local_trials = Catalog(args.artifact_root).list_trials(local.study_id, 10_000, 0)
        remote_trials = Catalog(args.artifact_root).list_trials(remote.study_id, 10_000, 0)
        print(
            json.dumps(
                {
                    "plan_hash": local_plan.canonical_hash,
                    "same_plan": local_plan.canonical_hash == remote_plan.canonical_hash,
                    "identical_trials": local_trials == remote_trials,
                    "local": local.model_dump(mode="json"),
                    "remote": remote.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    summary = execute(args.execution_profile)

    payload = {
        "study_id": summary.study_id,
        "status": summary.status.value,
        "plan": summary.artifacts.get("plan"),
        "backend": summary.execution.model_dump(mode="json"),
        "trials": summary.trial_count,
        "pareto_trials": summary.pareto_trials,
        "retained_runs": summary.retained_run_ids,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
