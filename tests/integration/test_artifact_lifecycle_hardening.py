from __future__ import annotations

import json
import shutil

import pytest

from icframe.adapters.pettingzoo import PettingZooParallelIncentiveEnv
from icframe.catalog import Catalog
from icframe.core.engine import run_experiment
from icframe.domain.incentive_spec import RetentionProfile
from icframe.domain.run import RunConfig, StudyConfig, StudyMode
from icframe.replay import replay_run
from icframe.study import run_study


def test_run_id_collision_never_changes_existing_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "fixed"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "manifest.json"
    sentinel.write_text('{"sentinel": true}')

    with pytest.raises(FileExistsError):
        run_experiment("public_goods", RunConfig(run_id="fixed", artifact_root=tmp_path))

    assert sentinel.read_text() == '{"sentinel": true}'
    assert list(run_dir.iterdir()) == [sentinel]


def test_custom_pack_path_is_replayable(tmp_path) -> None:
    from icframe.core.packs import builtin_pack_root

    pack_dir = tmp_path / "custom-pack"
    shutil.copytree(builtin_pack_root() / "public_goods", pack_dir)
    summary = run_experiment(
        pack_dir,
        RunConfig(
            seed=7,
            parameters={"steps": 8},
            sample_every_steps=3,
            artifact_root=tmp_path,
        ),
    )

    replayed = replay_run(tmp_path / "runs" / summary.run_id)
    assert replayed.agents == summary.agents
    assert replayed.checkpoints == summary.checkpoints


def test_replay_detects_agent_or_count_divergence(tmp_path) -> None:
    summary = run_experiment(
        "public_goods",
        RunConfig(seed=7, parameters={"steps": 2}, artifact_root=tmp_path),
    )
    summary_path = tmp_path / "runs" / summary.run_id / "summary.json"
    payload = json.loads(summary_path.read_text())
    payload["action_counts"][next(iter(payload["action_counts"]))] += 1
    summary_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="replay diverged"):
        replay_run(summary_path.parent)


def test_replay_rejects_a_different_runtime_version(tmp_path) -> None:
    summary = run_experiment(
        "public_goods",
        RunConfig(seed=7, parameters={"steps": 2}, artifact_root=tmp_path),
    )
    manifest_path = tmp_path / "runs" / summary.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_version"] = "0.3.0"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="runtime version changed"):
        replay_run(manifest_path.parent)


def test_catalog_rebuild_rolls_back_as_one_transaction(tmp_path, monkeypatch) -> None:
    summary = run_experiment(
        "public_goods",
        RunConfig(seed=7, parameters={"steps": 2}, artifact_root=tmp_path),
    )
    catalog = Catalog(tmp_path)

    def fail_upsert(summary, *, _connection=None):
        raise RuntimeError("index write failed")

    monkeypatch.setattr(catalog, "upsert_run", fail_upsert)
    with pytest.raises(RuntimeError, match="index write failed"):
        catalog.rebuild()

    assert catalog.get_run(summary.run_id) is not None


def test_pettingzoo_close_finalizes_an_early_episode(tmp_path) -> None:
    env = PettingZooParallelIncentiveEnv(
        "public_goods",
        artifact_root=tmp_path,
        retention=RetentionProfile.AUDIT,
        run_id="early-close",
    )
    env.reset(seed=7)
    env.close()

    manifest = json.loads((tmp_path / "runs" / "early-close" / "manifest.json").read_text())
    assert manifest["status"] == "cancelled"
    assert env.last_summary.status.value == "cancelled"


def test_failed_study_is_terminalized(tmp_path, monkeypatch) -> None:
    def fail_baseline(*args, **kwargs):
        raise RuntimeError("baseline exploded")

    monkeypatch.setattr("icframe.study.run_experiment", fail_baseline)
    config = StudyConfig(
        study_id="failed-study",
        mode=StudyMode.SINGLE,
        objectives=["social_welfare"],
        seeds=[7],
        trials=1,
        artifact_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="baseline exploded"):
        run_study("public_goods", config)

    manifest = json.loads(
        (tmp_path / "studies" / "failed-study" / "manifest.json").read_text()
    )
    summary = json.loads(
        (tmp_path / "studies" / "failed-study" / "summary.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["completed_at"]
    assert summary["status"] == "failed"
