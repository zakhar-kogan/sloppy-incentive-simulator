from __future__ import annotations

import json

import pytest

from icframe.core import RuntimeEngine, compile_runtime, load_domain_pack
from icframe.core.observer import NoopObserver
from icframe.core.packs import apply_parameters
from icframe.domain.incentive_spec import RetentionProfile


def _engine(pack_id: str, seed: int, **parameters) -> RuntimeEngine:
    pack = apply_parameters(load_domain_pack(pack_id), parameters)
    return RuntimeEngine(
        compile_runtime(pack),
        run_id=f"test-{pack_id}-{seed}",
        seed=seed,
        observer=NoopObserver(),
        retention=RetentionProfile.TRAINING,
    )


def test_parameter_overrides_must_align_with_declared_step() -> None:
    pack = load_domain_pack("delayed_reward_learning")

    apply_parameters(pack, {"epsilon": 0.24})
    with pytest.raises(ValueError, match="epsilon does not align with its step"):
        apply_parameters(pack, {"epsilon": 0.245})


def test_fixed_and_random_schedules_are_reproducible() -> None:
    for pack_id in ("delayed_reward_learning", "software_organization"):
        left = _engine(pack_id, 83, steps=12).run()
        right = _engine(pack_id, 83, steps=12).run()
        assert left.metrics == right.metrics
        assert left.action_counts == right.action_counts
        assert [agent.policy_state for agent in left.agents] == [
            agent.policy_state for agent in right.agents
        ]


def test_parallel_actions_observe_one_snapshot_and_combine_adds() -> None:
    engine = _engine("public_goods", 7, steps=1)
    starting = {
        agent_id: agent.resources["balance"] for agent_id, agent in engine.world.agents.items()
    }
    result = engine.step_external(
        {agent_id: ("contribute", None) for agent_id in engine.world.agents}
    )
    assert {observation.step for observation in result.observations} == {1}
    assert all(
        observation.resources["balance"] == starting[observation.agent_id]
        for observation in result.observations
    )
    assert all(value == pytest.approx(0.6) for value in result.rewards.values())
    assert all(
        agent.resources["balance"] == starting[agent_id] + 0.6
        for agent_id, agent in engine.world.agents.items()
    )


def test_observations_projects_current_state_without_resetting_runtime() -> None:
    engine = _engine("delayed_reward_learning", 11, steps=10)
    engine.step_internal()

    observations = engine.observations()

    assert engine.world.step == 1
    assert set(observations) == set(engine.world.agents)
    assert {observation.step for observation in observations.values()} == {1}


def test_training_memory_is_bounded_by_state_space_not_turns() -> None:
    engine = _engine("delayed_reward_learning", 11, steps=2_000)
    summary = engine.run()
    assert summary.steps_completed == 2_000
    assert summary.checkpoints == []
    assert all(len(agent.history) <= 4 for agent in engine.world.agents.values())
    assert all(len(json.dumps(agent.policy_state)) < 10_000 for agent in summary.agents)


def test_trusted_evaluation_cannot_change_after_compilation() -> None:
    engine = _engine("public_goods", 7, steps=1)
    engine.step_internal()
    engine.plan.spec.evaluation.objectives["trusted_score"].metric = "exploit_rate"
    try:
        engine.summary()
    except RuntimeError as exc:
        assert "trusted evaluation changed" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("mutated trusted evaluation was accepted")
