from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel

from icframe.domain.incentive_spec import (
    IncentiveSpec,
    PolicyKind,
    PromptCapture,
    ResponseCapture,
    RetentionProfile,
)
from icframe.domain.run import Checkpoint, RunSummary

from .core.types import Observation, PolicyDecision, RuntimeEvent
from .version import __version__


class ArtifactLifecycle:
    """Own atomic creation and terminal transitions for an artifact manifest."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)

    @classmethod
    def start(
        cls,
        directory: str | Path,
        manifest: dict[str, Any],
        *,
        files: dict[str, Any] | None = None,
    ) -> ArtifactLifecycle:
        artifact_dir = Path(directory)
        artifact_dir.mkdir(parents=True, exist_ok=False)
        lifecycle = cls(artifact_dir / "manifest.json")
        try:
            for name, payload in (files or {}).items():
                _write_json(artifact_dir / name, payload)
            initial = dict(manifest)
            initial.update(
                {
                    "status": "running",
                    "started_at": initial.get("started_at") or _now(),
                    "completed_at": None,
                }
            )
            _write_json(lifecycle.manifest_path, initial)
        except Exception:
            # The directory was created exclusively above, so cleanup cannot
            # remove another invocation's artifacts.
            shutil.rmtree(artifact_dir)
            raise
        return lifecycle

    def complete(self, **fields: Any) -> bool:
        return self.transition("completed", **fields)

    def fail(self, error: str, **fields: Any) -> bool:
        return self.transition("failed", error=error, **fields)

    def cancel(self, **fields: Any) -> bool:
        return self.transition("cancelled", **fields)

    def interrupt(self, **fields: Any) -> bool:
        return self.transition("interrupted", **fields)

    def transition(self, status: str, **fields: Any) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            payload = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        payload.update(fields)
        payload["status"] = status
        payload["completed_at"] = fields.get("completed_at") or _now()
        _write_json(self.manifest_path, payload)
        return True


class ArtifactObserver:
    """Stream bounded run diagnostics to an immutable artifact directory."""

    def __init__(
        self,
        root: str | Path,
        run_id: str,
        retention: RetentionProfile,
    ) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.retention = retention
        self.run_dir = self.root / "runs" / run_id
        self.spec: IncentiveSpec | None = None
        self._streams: dict[str, TextIO] = {}
        self._sample_every_steps = 1
        self._last_step_events: list[RuntimeEvent] = []
        self._recorded_event_ids: set[str] = set()
        self._lifecycle: ArtifactLifecycle | None = None

    def cancelled(self) -> bool:
        return False

    def start(self, context: dict[str, Any]) -> None:
        self.spec = IncentiveSpec.model_validate(context["spec"])
        self._sample_every_steps = int(context["sample_every_steps"])
        self._lifecycle = ArtifactLifecycle.start(
            self.run_dir,
            {
                "run_id": self.run_id,
                "pack_id": context["pack_id"],
                "pack_path": context.get("pack_path"),
                "seed": context["seed"],
                "retention": context["retention"],
                "hook_hash": context["hook_hash"],
                "runtime_hash": context["runtime_hash"],
                "trusted_evaluation_hash": context["trusted_evaluation_hash"],
                "parameters": context["parameters"],
                "sample_every_steps": context["sample_every_steps"],
                "runtime_version": __version__,
                "replayable": True,
            },
            files={"spec.json": context["spec"]},
        )

    def observation(self, value: Observation) -> None:
        if self.retention is RetentionProfile.AUDIT:
            self._append("observations.jsonl", value)

    def decision(self, value: PolicyDecision) -> None:
        should_record = self.retention is RetentionProfile.AUDIT or (
            self.retention is RetentionProfile.EXPERIMENT
            and (value.llm_call is not None or value.policy is PolicyKind.EXTERNAL or value.failure)
        )
        if should_record:
            payload = dataclasses.asdict(value)
            if payload.get("llm_call"):
                payload["llm_call"] = self._redact_llm(payload["llm_call"])
            self._append("decisions.jsonl", payload)
        if value.llm_call is not None:
            self._append("llm_calls.jsonl", self._redact_llm(value.llm_call))
        if value.policy is PolicyKind.EXTERNAL:
            self._append(
                "external_actions.jsonl",
                {
                    "step": value.step,
                    "agent_id": value.agent_id,
                    "action": value.action,
                    "target_id": value.target_id,
                },
            )

    def event(self, value: RuntimeEvent) -> None:
        if self.spec is None:
            raise RuntimeError("artifact observer has not started")
        exceptional = bool(value.violations or value.enforced or value.detected)
        diagnostic_step = (
            value.step == 1
            or value.step == self.spec.experiment.steps
            or value.step % self._sample_every_steps == 0
        )
        if not self._last_step_events or self._last_step_events[0].step != value.step:
            self._last_step_events = []
        self._last_step_events.append(value)
        if self.retention is RetentionProfile.AUDIT or (
            self.retention is RetentionProfile.EXPERIMENT and (exceptional or diagnostic_step)
        ):
            self._append("events.jsonl", value)
            if self.retention is RetentionProfile.EXPERIMENT:
                self._recorded_event_ids.add(value.event_id)
        if self.retention is RetentionProfile.AUDIT or (
            self.retention is RetentionProfile.EXPERIMENT and exceptional
        ):
            self._append(
                "constraints.jsonl",
                {
                    "event_id": value.event_id,
                    "step": value.step,
                    "actor_id": value.actor_id,
                    "transition_id": value.transition_id,
                    "availability": value.availability,
                    "norm_status": value.norm_status,
                    "reasons": value.explanation_reasons,
                    "violations": value.violations,
                    "remediation_actions": value.remediation_actions,
                },
            )

    def checkpoint(self, value: Checkpoint) -> None:
        if self.retention is not RetentionProfile.TRAINING:
            self._append("checkpoints.jsonl", value)

    def finish(self, value: RunSummary) -> None:
        if self.retention is RetentionProfile.EXPERIMENT:
            for event in self._last_step_events:
                if event.event_id not in self._recorded_event_ids:
                    self._append("events.jsonl", event)
        for stream in self._streams.values():
            stream.flush()
            stream.close()
        self._streams.clear()
        artifacts = {
            "manifest": str(self.run_dir / "manifest.json"),
            "spec": str(self.run_dir / "spec.json"),
            "summary": str(self.run_dir / "summary.json"),
        }
        for path in sorted(self.run_dir.glob("*.jsonl")):
            artifacts[path.stem] = str(path)
        value.artifacts = artifacts
        _write_json(self.run_dir / "summary.json", value.model_dump(mode="json"))
        if self._lifecycle is None:
            raise RuntimeError("artifact observer has not started")
        fields = {
            "replayable": value.replayable,
            "replay_reason": value.replay_reason,
            "metrics": value.metrics,
            "objectives": value.objectives,
            "feasible": value.feasible,
            "artifacts": artifacts,
        }
        if value.status.value == "failed":
            transitioned = self._lifecycle.fail(value.error or "run failed", **fields)
        elif value.status.value == "cancelled":
            transitioned = self._lifecycle.cancel(**fields)
        elif value.status.value == "interrupted":
            transitioned = self._lifecycle.interrupt(**fields)
        else:
            transitioned = self._lifecycle.complete(**fields)
        if not transitioned:
            raise RuntimeError("artifact manifest could not be terminalized")
        from .catalog import Catalog

        Catalog(self.root).upsert_run(value)

    def _append(self, filename: str, value: Any) -> None:
        stream = self._streams.get(filename)
        if stream is None:
            stream = (self.run_dir / filename).open("a")
            self._streams[filename] = stream
        stream.write(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")))
        stream.write("\n")

    def _redact_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.spec is None:
            raise RuntimeError("artifact observer has not started")
        redaction = self.spec.observability.redaction
        result = {
            "id": payload.get("id"),
            "request_hash": payload.get("request_hash"),
            "parsed": payload.get("parsed", {}),
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "prompt_tokens": payload.get("prompt_tokens", 0),
            "completion_tokens": payload.get("completion_tokens", 0),
            "total_tokens": payload.get("total_tokens", 0),
            "estimated_cost": payload.get("estimated_cost", 0.0),
            "error": payload.get("error"),
        }
        prompt = str(payload.get("prompt", ""))
        content = str(payload.get("content", ""))
        if redaction.prompt_capture is PromptCapture.HASH:
            result["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()
        elif redaction.prompt_capture is PromptCapture.FULL:
            result["prompt"] = prompt
        if redaction.response_capture in {
            ResponseCapture.PARSED_AND_HASH,
            ResponseCapture.FULL,
        }:
            result["response_hash"] = hashlib.sha256(content.encode()).hexdigest()
        if redaction.response_capture is ResponseCapture.FULL:
            result["content"] = content
        return result


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    temporary.replace(path)


def update_manifest(
    path: str | Path,
    status: str,
    *,
    error: str | None = None,
    completed_at: str | None = None,
) -> bool:
    """Atomically move an existing artifact manifest to a terminal state."""
    fields: dict[str, Any] = {}
    if error is not None:
        fields["error"] = error
    if completed_at is not None:
        fields["completed_at"] = completed_at
    return ArtifactLifecycle(path).transition(status, **fields)


def _now() -> str:
    return datetime.now(UTC).isoformat()
