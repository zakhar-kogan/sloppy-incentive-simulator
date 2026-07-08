from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from icframe.constraints import ConstraintReport, validate_constraints
from icframe.domain.incentive_spec import IncentiveSpec, load_incentive_spec
from icframe.runtime.incentive import SimulationTrace, run_incentive_simulation


def run_incentive_spec_file(path: str | Path, seed: int | None = None) -> SimulationTrace:
    spec = load_incentive_spec(path)
    report = validate_constraints(spec)
    if not report.ok:
        messages = ", ".join(f"{problem.subject}:{problem.code}" for problem in report.problems)
        raise ValueError(f"IncentiveSpec constraint validation failed: {messages}")
    return run_incentive_simulation(spec, seed=seed)


def persist_incentive_run(
    output_dir: str | Path,
    spec: IncentiveSpec,
    trace: SimulationTrace,
    constraint_report: ConstraintReport | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = constraint_report or validate_constraints(spec)
    artifacts = {
        "spec": output_path / "incentive_spec.json",
        "trace": output_path / "trace.json",
        "constraints": output_path / "constraints.json",
        "summary": output_path / "summary.json",
    }
    artifacts["spec"].write_text(spec.model_dump_json(indent=2, by_alias=True))
    artifacts["trace"].write_text(trace.model_dump_json(indent=2))
    artifacts["constraints"].write_text(report.model_dump_json(indent=2))
    artifacts["summary"].write_text(json.dumps(_summary_payload(trace), indent=2, sort_keys=True))
    return {name: str(path) for name, path in artifacts.items()}


def _summary_payload(trace: SimulationTrace) -> dict[str, object]:
    action_counts = Counter(event.action for event in trace.events)
    tag_counts: Counter[str] = Counter()
    for event in trace.events:
        tag_counts.update(event.tags)
    return {
        "run_id": trace.run_id,
        "spec_name": trace.spec_name,
        "seed": trace.seed,
        "event_count": len(trace.events),
        "action_counts": dict(sorted(action_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "metric_results": trace.metric_results,
    }
