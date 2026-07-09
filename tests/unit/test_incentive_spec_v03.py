from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from icframe.adapters import (
    PettingZooAECIncentiveEnv,
    PettingZooIncentiveEnv,
    PettingZooParallelIncentiveEnv,
)
from icframe.constraints import explain_transition_availability
from icframe.domain.incentive_spec import IncentiveSpec, load_incentive_spec
from icframe.llm import (
    AgnoPolicyAdapter,
    FakeLLMClient,
    LiteLLMClient,
    LLMRequest,
    LLMResponse,
)
from icframe.observability import JsonlObserver, read_jsonl
from icframe.replay import replay_incentive_run
from icframe.runtime.incentive import choose_action, run_incentive_simulation

SPEC_PATH = "incentive_spec_example_tokenmaxxing_v0_3.toml"
LEARNING_SPEC_PATH = "incentive_spec_example_learning_v0_3.toml"


def test_v03_fixture_loads_with_observability_config() -> None:
    spec = load_incentive_spec(SPEC_PATH)

    assert spec.spec.version == "0.3"
    assert spec.observability.enabled
    assert spec.archetypes["llm_worker"].llm is not None
    assert spec.archetypes["llm_worker"].llm.backend == "mock"


def test_jsonl_artifacts_are_emitted_and_symbolic_replay_matches(tmp_path) -> None:
    spec = load_incentive_spec(SPEC_PATH)
    observer = JsonlObserver(tmp_path)
    llm_client = FakeLLMClient("misreport_activity")

    trace = run_incentive_simulation(spec, seed=7, observer=observer, llm_client=llm_client)
    replayed = replay_incentive_run(spec, tmp_path)

    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "trace.jsonl").exists()
    assert (tmp_path / "observations.jsonl").exists()
    assert (tmp_path / "policy_decisions.jsonl").exists()
    assert (tmp_path / "constraint_explanations.jsonl").exists()
    assert (tmp_path / "llm_calls.jsonl").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "agent_memory.json").exists()
    assert read_jsonl(tmp_path / "policy_decisions.jsonl")
    assert [event.canonical_json() for event in trace.events] == [
        event.canonical_json() for event in replayed.events
    ]


def test_policy_decisions_are_logged_for_symbolic_and_bandit_policies() -> None:
    spec = load_incentive_spec("incentive_spec_example_tokenmaxxing.toml")
    trace = run_incentive_simulation(spec, seed=3)

    backends = {decision.policy_backend for decision in trace.policy_decisions}

    assert "epsilon_greedy_bandit" in backends
    assert "scripted" in backends
    assert all(decision.observation_id for decision in trace.policy_decisions)


def test_public_choose_action_records_bandit_decision() -> None:
    spec = load_incentive_spec("incentive_spec_example_tokenmaxxing.toml")
    trace = run_incentive_simulation(spec, seed=1)
    policy = spec.archetypes["proxy_maximizer"].policy

    decision = choose_action(
        policy,
        observation=trace.observations[0],
        action_space=["real_work", "misreport_activity"],
        memory={},
        rng=random.Random(1),
        estimated_scalar_rewards={"real_work": 1.0, "misreport_activity": 5.0},
        behavior={"exploration_rate": 0.0},
    )

    assert policy == "epsilon_greedy_bandit"
    assert decision.chosen_action == "misreport_activity"
    assert decision.estimated_scalar_rewards["misreport_activity"] == 5.0


def test_epsilon_greedy_exploits_learned_memory_values() -> None:
    spec = load_incentive_spec("incentive_spec_example_tokenmaxxing.toml")
    trace = run_incentive_simulation(spec, seed=1)
    policy = spec.archetypes["proxy_maximizer"].policy

    decision = choose_action(
        policy,
        observation=trace.observations[0],
        action_space=["real_work", "misreport_activity"],
        memory={
            "policy": {
                "value_estimates": {
                    "real_work": 9.0,
                    "misreport_activity": 1.0,
                }
            }
        },
        rng=random.Random(1),
        behavior={"exploration_rate": 0.0},
    )

    assert decision.chosen_action == "real_work"
    assert decision.estimated_scalar_rewards["real_work"] == 9.0


def test_ucb_explores_unvisited_actions_then_exploits() -> None:
    spec = _single_state_learning_spec("ucb_bandit", steps=4)

    trace = run_incentive_simulation(spec, seed=2)
    actions = [event.action for event in trace.events[:3]]
    memory = trace.final_agent_state["learner_000"].memory["policy"]

    assert actions[:2] == ["low", "high"]
    assert actions[2] == "high"
    assert memory["action_counts"]["low"] == 1
    assert memory["action_counts"]["high"] == 3


def test_thompson_sampling_is_deterministic_and_updates_posterior() -> None:
    spec = _single_state_learning_spec("thompson_sampling_bandit", steps=4)

    first = run_incentive_simulation(spec, seed=3)
    second = run_incentive_simulation(spec, seed=3)
    posterior = first.final_agent_state["learner_000"].memory["policy"]["thompson"]

    assert first.canonical_json() == second.canonical_json()
    assert posterior
    assert sum(stats["count"] for stats in posterior.values()) == 4


def test_contextual_bandit_updates_action_feature_weights() -> None:
    spec = _single_state_learning_spec("contextual_bandit", steps=3)

    trace = run_incentive_simulation(spec, seed=4)
    memory = trace.final_agent_state["learner_000"].memory["policy"]
    contextual_deltas = [
        decision.policy_state_delta.get("contextual")
        for decision in trace.policy_decisions
        if decision.policy_state_delta.get("contextual")
    ]

    assert contextual_deltas
    assert memory["contextual"]["weights"]
    assert "latent." not in json.dumps(memory["contextual"])


def test_q_learning_updates_q_table_deterministically_with_future_value() -> None:
    spec = load_incentive_spec(LEARNING_SPEC_PATH)

    first = run_incentive_simulation(spec, seed=11)
    second = run_incentive_simulation(spec, seed=11)
    q_values = first.final_agent_state["q_learner_000"].memory["policy"]["q_values"]

    assert first.canonical_json() == second.canonical_json()
    assert q_values["setup"]["harvest"] > 0.0
    assert q_values["start"]["invest"] > 0.0
    assert any(decision.policy_state_delta.get("q_learning") for decision in first.policy_decisions)


def test_mocked_llm_policy_chooses_valid_action_without_latent_prompt_leak() -> None:
    spec = load_incentive_spec(SPEC_PATH)
    llm_client = FakeLLMClient("misreport_activity")

    trace = run_incentive_simulation(spec, seed=7, llm_client=llm_client)

    assert trace.policy_decisions[0].chosen_action == "misreport_activity"
    assert trace.llm_calls[0].parsed_response["action"] == "misreport_activity"
    assert "latent.goal_value" not in llm_client.requests[0].prompt


def test_malformed_llm_output_is_logged_and_rejected() -> None:
    class BadLLMClient:
        def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                content=json.dumps({"not_action": "misreport_activity"}),
                parsed={"not_action": "misreport_activity"},
                provider=request.provider,
                model=request.model,
            )

    spec = load_incentive_spec(SPEC_PATH)

    trace = run_incentive_simulation(spec, seed=7, llm_client=BadLLMClient())

    assert trace.policy_decisions[0].failure_mode == "malformed_llm_output"
    assert trace.policy_decisions[0].chosen_action is None


def test_litellm_client_uses_completion_json_mode(monkeypatch) -> None:
    litellm = pytest.importorskip("litellm")
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"action": "real_work", "rationale": "patched-litellm"})
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7),
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    response = LiteLLMClient().complete(
        LLMRequest(
            llm_call_id="llm_test",
            policy_decision_id="decision_test",
            provider="litellm",
            model="openai/test-model",
            system_prompt="Choose JSON.",
            prompt="Pick an action.",
        )
    )

    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["messages"][0]["role"] == "system"
    assert response.parsed["action"] == "real_work"
    assert response.total_tokens == 7


def test_litellm_client_logs_malformed_json(monkeypatch) -> None:
    litellm = pytest.importorskip("litellm")

    def fake_completion(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    response = LiteLLMClient().complete(
        LLMRequest(
            llm_call_id="llm_bad",
            policy_decision_id="decision_bad",
            provider="litellm",
            model="openai/test-model",
            prompt="Pick an action.",
        )
    )

    assert response.content == "not-json"
    assert response.parsed == {}


def test_agno_policy_adapter_uses_supplied_agent() -> None:
    class FakeAgnoAgent:
        def run(self, prompt: str):
            assert "Pick" in prompt
            return SimpleNamespace(content=json.dumps({"action": "real_work"}))

    response = AgnoPolicyAdapter(FakeAgnoAgent()).complete(
        LLMRequest(
            llm_call_id="agno_call",
            policy_decision_id="agno_decision",
            provider="agno",
            model="agno/fake-agent",
            prompt="Pick an action.",
        )
    )

    assert response.provider == "agno"
    assert response.parsed["action"] == "real_work"


def test_clingo_explanation_includes_forbidden_obligatory_violated_and_blocked() -> None:
    spec = load_incentive_spec(SPEC_PATH)

    forbidden = explain_transition_availability(
        spec,
        actor_id="llm_worker_000",
        state="working",
        action="misreport_activity",
    )
    obligatory = explain_transition_availability(
        spec,
        actor_id="llm_worker_000",
        state="reported_high_activity",
        action="remediate",
    )
    payload = spec.model_dump(mode="python", by_alias=True)
    payload["transitions"][0]["availability"] = "hard_blocked"
    blocked_spec = IncentiveSpec.model_validate(payload)
    blocked = explain_transition_availability(
        blocked_spec,
        actor_id="llm_worker_000",
        state="working",
        action="real_work",
    )

    assert forbidden.available
    assert forbidden.violations == ["forbidden_action"]
    assert forbidden.remediation_actions == ["remediate"]
    assert obligatory.obligations == ["remediate"]
    assert blocked.blocked
    assert not blocked.available


def test_pettingzoo_adapter_exposes_action_masks_and_observations() -> None:
    spec = load_incentive_spec(SPEC_PATH)
    env = PettingZooIncentiveEnv(spec)
    observations = env.reset(seed=7)

    mask = env.action_mask("llm_worker_000")

    assert observations["llm_worker_000"].visible_actions
    assert len(mask) == len(spec.actions.all)
    assert mask[spec.actions.all.index("real_work")] == 1
    assert mask[spec.actions.all.index("remediate")] == 0


def test_real_pettingzoo_aec_adapter_exposes_rewards_and_infos() -> None:
    pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    spec = load_incentive_spec(SPEC_PATH)
    env = PettingZooAECIncentiveEnv(spec)

    observations = env.reset(seed=7)
    agent = env.agent_selection
    assert agent is not None
    action_index = spec.actions.all.index("real_work")
    env.step(action_index)

    assert observations[agent]["action_mask"][action_index] == 1
    assert env.rewards[agent] > 0.0
    assert env.infos[agent]["scalar_reward"] == env.rewards[agent]
    assert "outcome_vector" in env.infos[agent]


def test_pettingzoo_aec_last_and_norm_infos_for_possible_violation() -> None:
    pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    spec = load_incentive_spec(SPEC_PATH)
    env = PettingZooAECIncentiveEnv(spec)

    env.reset(seed=7)
    agent = env.agent_selection
    assert agent is not None
    action_index = spec.actions.all.index("misreport_activity")
    env.step(action_index)
    observation, reward, terminated, truncated, _info = env.last()

    assert observation is not None
    assert not terminated
    assert not truncated
    assert reward == 0.0
    assert env.infos[agent]["norm_status"] == "forbidden"
    assert env.infos[agent]["violations"] == ["forbidden_action"]
    assert "outcome_vector" in env.infos[agent]


def test_pettingzoo_masks_hard_blocked_actions_with_clingo() -> None:
    spec = load_incentive_spec(SPEC_PATH)
    payload = spec.model_dump(mode="python", by_alias=True)
    payload["transitions"][0]["availability"] = "hard_blocked"
    blocked = IncentiveSpec.model_validate(payload)
    env = PettingZooIncentiveEnv(blocked)
    env.reset(seed=7)

    mask = env.action_mask("llm_worker_000")

    assert mask[blocked.actions.all.index("real_work")] == 0
    assert mask[blocked.actions.all.index("misreport_activity")] == 1


def test_real_pettingzoo_parallel_adapter_exposes_vector_rewards() -> None:
    pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    spec = load_incentive_spec(SPEC_PATH)
    env = PettingZooParallelIncentiveEnv(spec)
    observations, infos = env.reset(seed=7)

    actions = {
        agent: spec.actions.all.index("real_work")
        for agent, observation in observations.items()
        if observation["action_mask"][spec.actions.all.index("real_work")]
    }
    _, rewards, _, _, step_infos = env.step(actions)

    assert infos
    assert rewards["llm_worker_000"] > 0.0
    assert step_infos["llm_worker_000"]["scalar_reward"] == rewards["llm_worker_000"]
    assert "outcome_vector" in step_infos["llm_worker_000"]


def _single_state_learning_spec(policy: str, *, steps: int) -> IncentiveSpec:
    spec = load_incentive_spec(LEARNING_SPEC_PATH)
    payload = spec.model_dump(mode="python", by_alias=True)
    payload["spec"]["name"] = f"{policy}_reference"
    payload["experiment"]["steps"] = steps
    payload["states"] = {"initial_global": "start", "all": ["start"]}
    payload["actions"] = {"all": ["low", "high"]}
    payload["archetypes"] = {
        "learner": {
            "policy": policy,
            "role": "learner",
            "visibility_profile": "full",
            "scalarizer": {"agent.personal_payoff": 1.0},
            "behavior": {
                "exploration_rate": 0.0,
                "learning_rate": 0.25,
                "ucb_exploration_coefficient": 0.1,
            },
        }
    }
    payload["population"] = [{"archetype": "learner", "count": 1}]
    payload["transitions"] = [
        {
            "id": "low",
            "from": "start",
            "action": "low",
            "to": "start",
            "availability": "hard_available",
            "norm_status": "permitted",
            "tags": ["low"],
            "effects": {"observed.reward_signal": 1.0, "agent.personal_payoff": 1.0},
        },
        {
            "id": "high",
            "from": "start",
            "action": "high",
            "to": "start",
            "availability": "hard_available",
            "norm_status": "permitted",
            "tags": ["high"],
            "effects": {"observed.reward_signal": 3.0, "agent.personal_payoff": 3.0},
        },
    ]
    return IncentiveSpec.model_validate(payload)
