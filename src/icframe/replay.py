from __future__ import annotations

import json
from pathlib import Path

from icframe.domain.incentive_spec import IncentiveSpec
from icframe.llm import RecordedLLMClient
from icframe.runtime.incentive import SimulationTrace, run_incentive_simulation


def replay_incentive_run(
    spec: IncentiveSpec,
    artifact_dir: str | Path,
    *,
    use_recorded_llm: bool = True,
) -> SimulationTrace:
    """Replay a run from its manifest seed and optional recorded LLM calls."""

    path = Path(artifact_dir)
    manifest = json.loads((path / "run_manifest.json").read_text())
    llm_client = None
    if use_recorded_llm and (path / "llm_calls.jsonl").exists():
        llm_client = RecordedLLMClient(
            path / "llm_calls.jsonl",
            fail_on_missing=bool(
                manifest.get("replay_policy", {}).get("fail_on_missing_replay_call", True)
            ),
        )
    return run_incentive_simulation(spec, seed=int(manifest["seed"]), llm_client=llm_client)
