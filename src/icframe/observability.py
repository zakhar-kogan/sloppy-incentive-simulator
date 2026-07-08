from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icframe.domain.incentive_spec import IncentiveSpec, ObservabilityStream


def stable_trace_id(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return f"trace_{digest}"


class JsonlObserver:
    """Write v0.3 observability artifacts without coupling runtime mechanics to IO."""

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.started_at: str | None = None
        self.spec: IncentiveSpec | None = None
        self.run_id: str | None = None
        self.trace_id: str | None = None
        self.seed: int | None = None

    def start_run(self, spec: IncentiveSpec, *, run_id: str, trace_id: str, seed: int) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = _utc_now()
        self.spec = spec
        self.run_id = run_id
        self.trace_id = trace_id
        self.seed = seed
        _write_json(
            self.artifact_dir / "run_manifest.json",
            {
                "spec_name": spec.spec.name,
                "spec_version": spec.spec.version,
                "spec_hash": _spec_hash(spec),
                "run_id": run_id,
                "trace_id": trace_id,
                "seed": seed,
                "runtime_version": "icframe-v0.3-core",
                "dependency_versions": {},
                "started_at": self.started_at,
                "completed_at": None,
                "artifact_paths": self._artifact_paths(),
                "redaction_policy": spec.observability.redaction.model_dump(mode="json"),
                "replay_policy": spec.observability.replay.model_dump(mode="json"),
            },
        )

    def record_observation(self, observation: Any) -> None:
        self._write_stream(
            ObservabilityStream.OBSERVATIONS,
            "observations.jsonl",
            observation.model_dump(mode="json", exclude_none=True),
        )

    def record_policy_decision(self, decision: Any) -> None:
        self._write_stream(
            ObservabilityStream.POLICY_DECISIONS,
            "policy_decisions.jsonl",
            decision.model_dump(mode="json", exclude_none=True),
        )

    def record_constraint_explanation(self, explanation: Any) -> None:
        self._write_stream(
            ObservabilityStream.CONSTRAINTS,
            "constraint_explanations.jsonl",
            explanation.model_dump(mode="json", exclude_none=True),
        )

    def record_llm_call(self, record: Any) -> None:
        self._write_stream(
            ObservabilityStream.LLM_CALLS,
            "llm_calls.jsonl",
            record.model_dump(mode="json", exclude_none=True),
        )

    def record_event(self, event: Any) -> None:
        self._write_stream(
            ObservabilityStream.EVENTS,
            "trace.jsonl",
            event.model_dump(mode="json", exclude_none=True),
        )

    def finish_run(self, trace: Any) -> None:
        if self.spec is None:
            raise RuntimeError("observer.finish_run called before start_run")
        manifest_path = self.artifact_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest["completed_at"] = _utc_now()
        manifest["metric_results"] = trace.metric_results
        _write_json(manifest_path, manifest)
        self._write_metrics(trace.metric_results)
        self._write_memory(trace.final_agent_state)

    def _write_stream(
        self,
        stream: ObservabilityStream,
        filename: str,
        payload: dict[str, Any],
    ) -> None:
        if self.spec is None:
            raise RuntimeError("observer stream write called before start_run")
        if not self.spec.observability.enabled:
            return
        if stream not in self.spec.observability.streams:
            return
        _append_jsonl(self.artifact_dir / filename, payload)

    def _write_metrics(self, metric_results: dict[str, float]) -> None:
        if self.spec is None or ObservabilityStream.METRICS not in self.spec.observability.streams:
            return
        path = self.artifact_dir / "metrics.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["metric_id", "name", "value"])
            writer.writeheader()
            for name, value in sorted(metric_results.items()):
                writer.writerow(
                    {
                        "metric_id": f"metric_{name}",
                        "name": name,
                        "value": value,
                    }
                )

    def _write_memory(self, final_agent_state: dict[str, Any]) -> None:
        if self.spec is None or ObservabilityStream.MEMORY not in self.spec.observability.streams:
            return
        payload = {agent_id: agent.memory for agent_id, agent in sorted(final_agent_state.items())}
        _write_json(self.artifact_dir / "agent_memory.json", payload)

    def _artifact_paths(self) -> dict[str, str]:
        names = [
            "run_manifest.json",
            "trace.jsonl",
            "observations.jsonl",
            "policy_decisions.jsonl",
            "constraint_explanations.jsonl",
            "llm_calls.jsonl",
            "metrics.csv",
            "agent_memory.json",
        ]
        return {name: str(self.artifact_dir / name) for name in names}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [json.loads(line) for line in file_path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(payload, sort_keys=True))
        file.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _spec_hash(spec: IncentiveSpec) -> str:
    payload = spec.model_dump(mode="json", exclude_none=True)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
