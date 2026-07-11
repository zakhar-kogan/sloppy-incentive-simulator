from __future__ import annotations

from pettingzoo.test import api_test, parallel_api_test

from icframe.adapters import PettingZooAECIncentiveEnv, PettingZooParallelIncentiveEnv
from icframe.domain.incentive_spec import RetentionProfile
from icframe.replay import replay_run


def test_official_pettingzoo_api_suites() -> None:
    parallel_api_test(PettingZooParallelIncentiveEnv("public_goods"), num_cycles=20)
    api_test(
        PettingZooAECIncentiveEnv("delayed_reward_learning"),
        num_cycles=20,
        verbose_progress=False,
    )


def test_external_actions_are_retained_and_replayable(tmp_path) -> None:
    env = PettingZooParallelIncentiveEnv(
        "public_goods",
        artifact_root=tmp_path,
        retention=RetentionProfile.AUDIT,
        run_id="external-run",
    )
    observations, _ = env.reset(seed=7)
    while env.agents:
        actions = {
            agent: int(observations[agent]["action_mask"].nonzero()[0][0]) for agent in env.agents
        }
        observations, *_ = env.step(actions)
    assert (tmp_path / "runs" / "external-run" / "external_actions.jsonl").exists()
    assert replay_run(tmp_path / "runs" / "external-run").metrics == env.last_summary.metrics
