from __future__ import annotations

from typing import ClassVar

from icframe.constraints import explain_transition_availability
from icframe.domain.incentive_spec import Availability, IncentiveSpec, PolicyBackend
from icframe.runtime.incentive import (
    AgentRuntimeState,
    Observation,
    PolicyDecision,
    _event_id,
    _execute_transition,
    _expand_population,
    _observation_id,
    _RuntimeWorld,
    compile_observation,
)


class PettingZooIncentiveEnv:
    """Small PettingZoo-shaped adapter without importing PettingZoo at runtime.

    The class exposes the pieces MARL trainers need first: agent IDs, observations,
    action masks from transition availability, scalar rewards, and vector outcomes in
    infos. A full AEC/Parallel subclass can wrap this once PettingZoo is installed.
    """

    metadata: ClassVar[dict[str, str]] = {"name": "icframe_incentive_v0_3"}

    def __init__(self, spec: IncentiveSpec) -> None:
        self.spec = spec
        self.possible_agents = list(_expand_population(spec))
        self.agents = list(self.possible_agents)
        self._agent_state: dict[str, AgentRuntimeState] = {}
        self.rewards: dict[str, float] = {}
        self.infos: dict[str, dict[str, object]] = {}
        self.world: _RuntimeWorld | None = None

    def reset(self, seed: int | None = None) -> dict[str, Observation]:
        import random

        run_seed = seed if seed is not None else self.spec.experiment.seeds[0]
        self._agent_state = _expand_population(self.spec)
        self.agents = list(self._agent_state)
        self.rewards = {agent_id: 0.0 for agent_id in self.agents}
        self.infos = {agent_id: {} for agent_id in self.agents}
        self.world = _RuntimeWorld(
            spec=self.spec,
            rng=random.Random(run_seed),
            seed=run_seed,
            run_id=f"{self.spec.spec.name}:{run_seed}:pettingzoo",
            trace_id=f"pettingzoo_{run_seed}",
            agents=self._agent_state,
        )
        return {
            agent_id: compile_observation(self.spec, self.world, agent)
            for agent_id, agent in self._agent_state.items()
        }

    def action_mask(self, agent_id: str) -> list[int]:
        if agent_id not in self._agent_state:
            return [0 for _ in self.spec.actions.all]
        state = self._agent_state[agent_id].current_state
        available_actions = _available_actions(self.spec, agent_id, state)
        return [1 if action in available_actions else 0 for action in self.spec.actions.all]

    def action_space(self, agent_id: str) -> list[str]:
        mask = self.action_mask(agent_id)
        return [
            action
            for action, is_available in zip(self.spec.actions.all, mask, strict=True)
            if is_available
        ]

    def observe(self, agent_id: str) -> Observation:
        if self.world is None:
            self.reset()
        assert self.world is not None
        return compile_observation(self.spec, self.world, self._agent_state[agent_id])


try:
    from pettingzoo import AECEnv as _AECEnv
    from pettingzoo import ParallelEnv as _ParallelEnv
except ImportError:  # pragma: no cover - exercised by base installs without the extra
    _AECEnv = object
    _ParallelEnv = object


class PettingZooAECIncentiveEnv(_AECEnv):
    """Optional PettingZoo AEC wrapper backed by the IncentiveSpec runtime."""

    metadata: ClassVar[dict[str, str]] = {
        "name": "icframe_incentive_v0_3_aec",
        "render_modes": [],
    }

    def __init__(self, spec: IncentiveSpec) -> None:
        if _AECEnv is object:
            raise RuntimeError("install icframe[marl] to use PettingZooAECIncentiveEnv")
        from gymnasium import spaces

        self.spec = spec
        self.possible_agents = list(_expand_population(spec))
        self.agents: list[str] = []
        self.agent_selection: str | None = None
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, object]] = {}
        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "action_mask": spaces.MultiBinary(len(spec.actions.all)),
                    "state_index": spaces.Discrete(len(spec.states.all)),
                }
            )
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: spaces.Discrete(len(spec.actions.all)) for agent_id in self.possible_agents
        }
        self._agent_state: dict[str, AgentRuntimeState] = {}
        self.world: _RuntimeWorld | None = None
        self._agent_cursor = 0
        self._turn_count = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> dict[str, dict[str, object]]:
        import random

        del options
        run_seed = seed if seed is not None else self.spec.experiment.seeds[0]
        self._agent_state = _expand_population(self.spec)
        self.agents = list(self._agent_state)
        self.agent_selection = self.agents[0] if self.agents else None
        self.rewards = {agent_id: 0.0 for agent_id in self.agents}
        self._cumulative_rewards = {agent_id: 0.0 for agent_id in self.agents}
        self.terminations = {agent_id: False for agent_id in self.agents}
        self.truncations = {agent_id: False for agent_id in self.agents}
        self.infos = {
            agent_id: {"action_mask": self.action_mask(agent_id)} for agent_id in self.agents
        }
        self._agent_cursor = 0
        self._turn_count = 0
        self.world = _RuntimeWorld(
            spec=self.spec,
            rng=random.Random(run_seed),
            seed=run_seed,
            run_id=f"{self.spec.spec.name}:{run_seed}:pettingzoo-aec",
            trace_id=f"pettingzoo_aec_{run_seed}",
            agents=self._agent_state,
        )
        return {agent_id: self.observe(agent_id) for agent_id in self.agents}

    def observe(self, agent: str) -> dict[str, object]:
        state_index = 0
        if agent in self._agent_state:
            state_index = self.spec.states.all.index(self._agent_state[agent].current_state)
        return {
            "action_mask": self.action_mask(agent),
            "state_index": state_index,
        }

    def action_mask(self, agent_id: str) -> list[int]:
        if agent_id not in self._agent_state:
            return [0 for _ in self.spec.actions.all]
        state = self._agent_state[agent_id].current_state
        available_actions = _available_actions(self.spec, agent_id, state)
        return [1 if action in available_actions else 0 for action in self.spec.actions.all]

    def last(
        self,
        observe: bool = True,
    ) -> tuple[dict[str, object] | None, float, bool, bool, dict[str, object]]:
        if self.agent_selection is None:
            return None, 0.0, True, True, {}
        agent_id = self.agent_selection
        observation = self.observe(agent_id) if observe else None
        return (
            observation,
            self.rewards.get(agent_id, 0.0),
            self.terminations.get(agent_id, False),
            self.truncations.get(agent_id, False),
            self.infos.get(agent_id, {}),
        )

    def step(self, action: int | None) -> None:
        if self.world is None:
            self.reset()
        if self.agent_selection is None or not self.agents:
            return
        assert self.world is not None
        agent_id = self.agent_selection
        self.rewards = {current: 0.0 for current in self.agents}
        if (
            action is not None
            and not self.terminations[agent_id]
            and not self.truncations[agent_id]
        ):
            self._apply_action(agent_id, action)
        self._turn_count += 1
        max_turns = self.spec.experiment.steps * max(len(self.agents), 1)
        if self._turn_count >= max_turns:
            self.truncations = {current: True for current in self.agents}
        self._advance_agent()

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _apply_action(self, agent_id: str, action: int) -> None:
        assert self.world is not None
        if action < 0 or action >= len(self.spec.actions.all):
            self.infos[agent_id] = {"invalid_action": action}
            return
        action_name = self.spec.actions.all[action]
        agent = self._agent_state[agent_id]
        transition = next(
            (
                item
                for item in self.spec.transitions
                if item.from_state == agent.current_state
                and item.action == action_name
                and item.availability is not Availability.HARD_BLOCKED
            ),
            None,
        )
        if transition is None:
            self.infos[agent_id] = {"invalid_action": action_name}
            return
        step = self._turn_count + 1
        observation = compile_observation(self.spec, self.world, agent, step=step, turn_index=1)
        decision = PolicyDecision(
            run_id=self.world.run_id,
            trace_id=self.world.trace_id,
            step=step,
            observation_id=observation.observation_id or _observation_id(step, 1, agent_id),
            policy_decision_id=f"decision_pettingzoo_{step:04d}_{agent_id}",
            agent_id=agent_id,
            policy_backend=PolicyBackend.PETTINGZOO_EXTERNAL,
            candidate_actions=[
                candidate.action
                for candidate in self.spec.transitions
                if candidate.from_state == agent.current_state
            ],
            chosen_action=action_name,
        )
        explanation = explain_transition_availability(
            self.spec,
            actor_id=agent_id,
            state=agent.current_state,
            action=action_name,
            constraint_id=f"constraint_pettingzoo_{step:04d}_{agent_id}",
            policy_decision_id=decision.policy_decision_id,
        )
        if not explanation.available:
            self.infos[agent_id] = {
                "constraint_id": explanation.constraint_id,
                "blocked": explanation.blocked,
                "norm_status": explanation.norm_status.value,
                "violations": explanation.violations,
                "remediation_actions": explanation.remediation_actions,
            }
            return
        event = _execute_transition(
            self.world,
            agent,
            transition,
            step,
            explanation,
            observation,
            decision,
            1,
        )
        reward = event.scalar_rewards.get(agent_id, 0.0)
        self.rewards[agent_id] = reward
        self._cumulative_rewards[agent_id] += reward
        self.infos[agent_id] = {
            "event_id": _event_id(step, 1, agent_id),
            "constraint_id": event.constraint_id,
            "norm_status": explanation.norm_status.value,
            "violations": explanation.violations,
            "remediation_actions": explanation.remediation_actions,
            "outcome_vector": event.final_outcome_vector,
            "scalar_reward": reward,
            "action_mask": self.action_mask(agent_id),
        }

    def _advance_agent(self) -> None:
        if not self.agents:
            self.agent_selection = None
            return
        self._agent_cursor = (self._agent_cursor + 1) % len(self.agents)
        self.agent_selection = self.agents[self._agent_cursor]


class PettingZooParallelIncentiveEnv(_ParallelEnv):
    """Optional PettingZoo Parallel wrapper for simultaneous external policies."""

    metadata: ClassVar[dict[str, str]] = {
        "name": "icframe_incentive_v0_3_parallel",
        "render_modes": [],
    }

    def __init__(self, spec: IncentiveSpec) -> None:
        if _ParallelEnv is object:
            raise RuntimeError("install icframe[marl] to use PettingZooParallelIncentiveEnv")
        from gymnasium import spaces

        self.spec = spec
        self.possible_agents = list(_expand_population(spec))
        self.agents: list[str] = []
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, object]] = {}
        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "action_mask": spaces.MultiBinary(len(spec.actions.all)),
                    "state_index": spaces.Discrete(len(spec.states.all)),
                }
            )
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: spaces.Discrete(len(spec.actions.all)) for agent_id in self.possible_agents
        }
        self._agent_state: dict[str, AgentRuntimeState] = {}
        self.world: _RuntimeWorld | None = None
        self._turn_count = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        import random

        del options
        run_seed = seed if seed is not None else self.spec.experiment.seeds[0]
        self._agent_state = _expand_population(self.spec)
        self.agents = list(self._agent_state)
        self.rewards = {agent_id: 0.0 for agent_id in self.agents}
        self.terminations = {agent_id: False for agent_id in self.agents}
        self.truncations = {agent_id: False for agent_id in self.agents}
        self.infos = {
            agent_id: {"action_mask": self.action_mask(agent_id)} for agent_id in self.agents
        }
        self._turn_count = 0
        self.world = _RuntimeWorld(
            spec=self.spec,
            rng=random.Random(run_seed),
            seed=run_seed,
            run_id=f"{self.spec.spec.name}:{run_seed}:pettingzoo-parallel",
            trace_id=f"pettingzoo_parallel_{run_seed}",
            agents=self._agent_state,
        )
        observations = {agent_id: self.observe(agent_id) for agent_id in self.agents}
        return observations, self.infos

    def observe(self, agent: str) -> dict[str, object]:
        state_index = 0
        if agent in self._agent_state:
            state_index = self.spec.states.all.index(self._agent_state[agent].current_state)
        return {
            "action_mask": self.action_mask(agent),
            "state_index": state_index,
        }

    def action_mask(self, agent_id: str) -> list[int]:
        if agent_id not in self._agent_state:
            return [0 for _ in self.spec.actions.all]
        state = self._agent_state[agent_id].current_state
        available_actions = _available_actions(self.spec, agent_id, state)
        return [1 if action in available_actions else 0 for action in self.spec.actions.all]

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, dict[str, object]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, object]],
    ]:
        if self.world is None:
            self.reset()
        assert self.world is not None
        self.rewards = {agent_id: 0.0 for agent_id in self.agents}
        self._turn_count += 1
        for turn_index, (agent_id, action) in enumerate(sorted(actions.items()), start=1):
            if agent_id not in self.agents:
                continue
            self._apply_action(agent_id, action, self._turn_count, turn_index)
        max_turns = self.spec.experiment.steps
        if self._turn_count >= max_turns:
            self.truncations = {agent_id: True for agent_id in self.agents}
        observations = {agent_id: self.observe(agent_id) for agent_id in self.agents}
        return observations, self.rewards, self.terminations, self.truncations, self.infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _apply_action(self, agent_id: str, action: int, step: int, turn_index: int) -> None:
        assert self.world is not None
        if action < 0 or action >= len(self.spec.actions.all):
            self.infos[agent_id] = {"invalid_action": action}
            return
        action_name = self.spec.actions.all[action]
        agent = self._agent_state[agent_id]
        transition = next(
            (
                item
                for item in self.spec.transitions
                if item.from_state == agent.current_state
                and item.action == action_name
                and item.availability is not Availability.HARD_BLOCKED
            ),
            None,
        )
        if transition is None:
            self.infos[agent_id] = {"invalid_action": action_name}
            return
        observation = compile_observation(
            self.spec,
            self.world,
            agent,
            step=step,
            turn_index=turn_index,
        )
        decision = PolicyDecision(
            run_id=self.world.run_id,
            trace_id=self.world.trace_id,
            step=step,
            observation_id=(
                observation.observation_id or _observation_id(step, turn_index, agent_id)
            ),
            policy_decision_id=f"decision_parallel_{step:04d}_{turn_index:04d}_{agent_id}",
            agent_id=agent_id,
            policy_backend=PolicyBackend.PETTINGZOO_EXTERNAL,
            candidate_actions=[
                candidate.action
                for candidate in self.spec.transitions
                if candidate.from_state == agent.current_state
            ],
            chosen_action=action_name,
        )
        explanation = explain_transition_availability(
            self.spec,
            actor_id=agent_id,
            state=agent.current_state,
            action=action_name,
            constraint_id=f"constraint_parallel_{step:04d}_{turn_index:04d}_{agent_id}",
            policy_decision_id=decision.policy_decision_id,
        )
        if not explanation.available:
            self.infos[agent_id] = {
                "constraint_id": explanation.constraint_id,
                "blocked": explanation.blocked,
                "norm_status": explanation.norm_status.value,
                "violations": explanation.violations,
            }
            return
        event = _execute_transition(
            self.world,
            agent,
            transition,
            step,
            explanation,
            observation,
            decision,
            turn_index,
        )
        reward = event.scalar_rewards.get(agent_id, 0.0)
        self.rewards[agent_id] = reward
        self.infos[agent_id] = {
            "event_id": event.event_id,
            "constraint_id": event.constraint_id,
            "norm_status": explanation.norm_status.value,
            "violations": explanation.violations,
            "remediation_actions": explanation.remediation_actions,
            "outcome_vector": event.final_outcome_vector,
            "scalar_reward": reward,
            "action_mask": self.action_mask(agent_id),
        }


def _available_actions(spec: IncentiveSpec, agent_id: str, state: str) -> set[str]:
    available_actions: set[str] = set()
    for action in spec.actions.all:
        explanation = explain_transition_availability(
            spec,
            actor_id=agent_id,
            state=state,
            action=action,
        )
        if explanation.available:
            available_actions.add(action)
    return available_actions
