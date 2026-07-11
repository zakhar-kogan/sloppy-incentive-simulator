from __future__ import annotations

import json

import pytest

from icframe import (
    RunConfig,
    compile_runtime,
    load_domain_pack,
    replay_run,
    run_experiment,
)
from icframe.catalog import Catalog
from icframe.core.engine import RuntimeEngine
from icframe.core.observer import NoopObserver
from icframe.domain.incentive_spec import RetentionProfile


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_audit_metrics_recompute_and_replay(tmp_path) -> None:
    summary = run_experiment(
        "public_goods",
        RunConfig(
            seed=7,
            parameters={"steps": 4},
            retention=RetentionProfile.AUDIT,
            artifact_root=tmp_path,
        ),
    )
    run_dir = tmp_path / "runs" / summary.run_id
    events = _lines(run_dir / "events.jsonl")
    recomputed = sum(event["global_outcome"].get("latent.social_welfare", 0.0) for event in events)
    assert recomputed == pytest.approx(summary.metrics["social_welfare"])
    assert len(_lines(run_dir / "observations.jsonl")) == summary.event_count
    assert len(_lines(run_dir / "decisions.jsonl")) == summary.event_count
    assert len(_lines(run_dir / "constraints.jsonl")) == len(events)
    assert replay_run(run_dir).metrics == summary.metrics


def test_retention_profiles_keep_bounded_artifacts(tmp_path) -> None:
    experiment = run_experiment(
        "public_goods",
        RunConfig(
            seed=7,
            parameters={"steps": 1_000},
            retention=RetentionProfile.EXPERIMENT,
            artifact_root=tmp_path,
        ),
    )
    run_dir = tmp_path / "runs" / experiment.run_id
    assert len(experiment.checkpoints) <= 201
    assert len(_lines(run_dir / "events.jsonl")) < experiment.event_count

    training = run_experiment(
        "delayed_reward_learning",
        RunConfig(
            seed=11,
            parameters={"steps": 500},
            retention=RetentionProfile.TRAINING,
            artifact_root=tmp_path,
        ),
    )
    training_dir = tmp_path / "runs" / training.run_id
    assert training.checkpoints == []
    assert list(training_dir.glob("*.jsonl")) == []


def test_llm_calls_are_redacted_replayable_and_hidden(
    tmp_path,
    deterministic_llm_client,
) -> None:
    engine = RuntimeEngine(
        compile_runtime(load_domain_pack("software_organization")),
        run_id="hidden-information-check",
        seed=19,
        observer=NoopObserver(),
    )
    llm_observation = engine.observe("llm_engineer_000")
    assert all(candidate.visible_outcomes == {} for candidate in llm_observation.candidates)
    assert all(candidate.visible_sanctions == {} for candidate in llm_observation.candidates)

    summary = run_experiment(
        "software_organization",
        RunConfig(
            seed=19,
            parameters={"steps": 2},
            retention=RetentionProfile.AUDIT,
            artifact_root=tmp_path,
        ),
        llm_client=deterministic_llm_client("refactor_core"),
    )
    run_dir = tmp_path / "runs" / summary.run_id
    calls = _lines(run_dir / "llm_calls.jsonl")
    assert len(calls) == 2
    assert all(
        "prompt" not in call and call["parsed"]["action"] == "refactor_core" for call in calls
    )
    assert replay_run(run_dir).objectives == summary.objectives


def test_catalog_is_rebuildable_from_authoritative_files(tmp_path) -> None:
    run = run_experiment(
        "public_goods",
        RunConfig(seed=7, parameters={"steps": 1}, artifact_root=tmp_path),
    )
    catalog = Catalog(tmp_path)
    catalog.path.unlink()
    rebuilt = Catalog(tmp_path)
    assert rebuilt.rebuild() == {"runs": 1, "studies": 0}
    assert rebuilt.get_run(run.run_id) == run
