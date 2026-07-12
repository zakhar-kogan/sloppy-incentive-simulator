from __future__ import annotations

import uuid
from pathlib import Path
from typing import ClassVar, TypeAlias

from icframe.artifacts import ArtifactObserver
from icframe.core.compiler import RuntimePlan, compile_runtime
from icframe.core.engine import RuntimeEngine
from icframe.core.observer import NoopObserver
from icframe.core.packs import LoadedDomainPack, load_domain_pack
from icframe.core.types import StepResult
from icframe.domain.incentive_spec import PolicyKind, RetentionProfile, ScheduleMode
from icframe.domain.run import RunStatus

try:  # Kept entirely outside the base install.
    import numpy as np
    from gymnasium import spaces
    from pettingzoo import AECEnv, ParallelEnv
except ImportError:  # pragma: no cover - exercised by base-install tests
    np = None
    spaces = None
    AECEnv = object  # type: ignore[assignment,misc]
    ParallelEnv = object  # type: ignore[assignment,misc]


PackSource: TypeAlias = str | Path | LoadedDomainPack | RuntimePlan
ExternalAction: TypeAlias = tuple[str, str | None]


def _require_marl() -> None:
    if spaces is None or np is None or AECEnv is object:
        raise RuntimeError("install icframe[marl] to use the PettingZoo adapters")


def _plan(source: PackSource) -> RuntimePlan:
    if isinstance(source, RuntimePlan):
        return source
    if isinstance(source, LoadedDomainPack):
        return compile_runtime(source)
    return compile_runtime(load_domain_pack(source))


class _ExternalRuntime:
    def _configure(
        self,
        source: PackSource,
        *,
        artifact_root: str | Path | None,
        retention: RetentionProfile,
        run_id: str | None,
    ) -> None:
        _require_marl()
        self.plan = _plan(source)
        self.engine: RuntimeEngine | None = None
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.retention = retention
        self.requested_run_id = run_id
        self.last_summary = None
        self.possible_agents = _agent_ids(self.plan)
        populations = _agent_populations(self.plan)
        self._options: dict[str, tuple[ExternalAction, ...]] = {
            agent_id: _action_options(self.plan, agent_id, populations)
            for agent_id in self.possible_agents
        }
        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "action_mask": spaces.MultiBinary(len(self._options[agent_id])),
                    "state_index": spaces.Discrete(len(self.plan.spec.states.all)),
                }
            )
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: spaces.Discrete(len(self._options[agent_id]))
            for agent_id in self.possible_agents
        }

    def _reset_engine(self, seed: int | None) -> None:
        self._finalize_active()
        self.last_summary = None
        run_seed = seed if seed is not None else self.plan.spec.experiment.seeds[0]
        run_id = self.requested_run_id or f"run_{uuid.uuid4().hex[:12]}"
        observer = (
            ArtifactObserver(self.artifact_root, run_id, self.retention)
            if self.artifact_root is not None
            else NoopObserver()
        )
        self.engine = RuntimeEngine(
            self.plan,
            run_id=run_id,
            seed=run_seed,
            observer=observer,
            retention=self.retention,
        )
        for agent_id, agent in self.engine.world.agents.items():
            agent.policy_kind = PolicyKind.EXTERNAL
            self.engine.policies[agent_id] = None

    def _finalize_active(self) -> None:
        if self.engine is None or self.last_summary is not None:
            return
        self.last_summary = self.engine.finish_session(status=RunStatus.CANCELLED)
        self.engine = None

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _observation(self, agent_id: str) -> dict[str, object]:
        if self.engine is None or agent_id not in self.engine.world.agents:
            return {
                "action_mask": np.zeros(len(self._options[agent_id]), dtype=np.int8),
                "state_index": 0,
            }
        observation = self.engine.observe(agent_id)
        valid = {(item.action, item.target_id) for item in observation.candidates}
        return {
            "action_mask": np.asarray(
                [int(option in valid) for option in self._options[agent_id]],
                dtype=np.int8,
            ),
            "state_index": self.plan.spec.states.all.index(observation.state),
        }

    def _external_action(self, agent_id: str, action: int) -> ExternalAction:
        if action < 0 or action >= len(self._options[agent_id]):
            raise ValueError(f"action {action} is outside the action space for {agent_id}")
        return self._options[agent_id][action]


class PettingZooParallelIncentiveEnv(_ExternalRuntime, ParallelEnv):
    """PettingZoo Parallel API backed directly by the atomic ICFRAME engine."""

    metadata: ClassVar[dict[str, object]] = {
        "name": "icframe_incentive_v0_4_parallel",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        source: PackSource,
        *,
        artifact_root: str | Path | None = None,
        retention: RetentionProfile = RetentionProfile.TRAINING,
        run_id: str | None = None,
    ) -> None:
        self._configure(
            source,
            artifact_root=artifact_root,
            retention=retention,
            run_id=run_id,
        )
        if self.plan.spec.experiment.schedule is not ScheduleMode.PARALLEL_SIMULTANEOUS:
            raise ValueError("the Parallel adapter requires schedule=parallel_simultaneous")
        self.agents: list[str] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        del options
        self._reset_engine(seed)
        self.agents = list(self.possible_agents)
        observations = {agent: self._observation(agent) for agent in self.agents}
        return observations, {agent: {} for agent in self.agents}

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
        if self.engine is None:
            raise RuntimeError("reset() must be called before step()")
        acting_agents = list(self.agents)
        missing = set(acting_agents) - set(actions)
        if missing:
            raise ValueError(f"parallel action map is missing agents: {sorted(missing)}")
        selected = {
            agent: self._external_action(agent, int(actions[agent])) for agent in acting_agents
        }
        result = self.engine.step_external(selected)
        finished = result.terminated or self.engine.world.step >= self.plan.spec.experiment.steps
        rewards = {agent: result.rewards.get(agent, 0.0) for agent in acting_agents}
        terminations = {agent: result.terminated for agent in acting_agents}
        truncations = {agent: finished and not result.terminated for agent in acting_agents}
        infos = _step_infos(result, acting_agents)
        observations = {agent: self._observation(agent) for agent in acting_agents}
        if finished:
            self.last_summary = self.engine.finish_session()
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._finalize_active()


class PettingZooAECIncentiveEnv(_ExternalRuntime, AECEnv):
    """PettingZoo AEC API backed by the sequential ICFRAME engine."""

    metadata: ClassVar[dict[str, object]] = {
        "name": "icframe_incentive_v0_4_aec",
        "render_modes": [],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        source: PackSource,
        *,
        artifact_root: str | Path | None = None,
        retention: RetentionProfile = RetentionProfile.TRAINING,
        run_id: str | None = None,
    ) -> None:
        self._configure(
            source,
            artifact_root=artifact_root,
            retention=retention,
            run_id=run_id,
        )
        if self.plan.spec.experiment.schedule is ScheduleMode.PARALLEL_SIMULTANEOUS:
            raise ValueError("the AEC adapter requires a sequential schedule")
        self.agents: list[str] = []
        self.agent_selection = ""
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, object]] = {}
        self._cycle_order: list[str] = []
        self._cycle_cursor = 0
        self._cycle_actions: dict[str, ExternalAction] = {}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> None:
        del options
        self._reset_engine(seed)
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._begin_cycle()

    def observe(self, agent: str) -> dict[str, object]:
        return self._observation(agent)

    def step(self, action: int | None) -> None:
        if not self.agents:
            return
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError("live agents require an action")
        assert self.engine is not None
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        self._cycle_actions[agent] = self._external_action(agent, int(action))
        self._cycle_cursor += 1
        if self._cycle_cursor < len(self._cycle_order):
            self.agent_selection = self._cycle_order[self._cycle_cursor]
            return

        result = self.engine.step_external(self._cycle_actions)

        self.rewards.update(result.rewards)
        self.infos.update(_step_infos(result, self.agents))
        self._accumulate_rewards()
        finished = result.terminated or self.engine.world.step >= self.plan.spec.experiment.steps
        if finished:
            self.last_summary = self.engine.finish_session()
            self.terminations = {current: result.terminated for current in self.agents}
            self.truncations = {current: not result.terminated for current in self.agents}
            self._deads_step_first()
            return

        self._begin_cycle()

    def _begin_cycle(self) -> None:
        self._cycle_order = list(self.agents)
        self._cycle_cursor = 0
        self._cycle_actions = {}
        self.agent_selection = self._cycle_order[0] if self._cycle_order else ""

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._finalize_active()


def _agent_ids(plan: RuntimePlan) -> list[str]:
    return [
        f"{entry.archetype}_{index:03d}"
        for entry in plan.spec.population
        for index in range(entry.count)
    ]


def _agent_populations(plan: RuntimePlan) -> dict[str, str]:
    return {
        f"{entry.archetype}_{index:03d}": entry.archetype
        for entry in plan.spec.population
        for index in range(entry.count)
    }


def _action_options(
    plan: RuntimePlan,
    agent_id: str,
    populations: dict[str, str],
) -> tuple[ExternalAction, ...]:
    result: list[ExternalAction] = []
    seen: set[ExternalAction] = set()
    for transition in plan.transitions:
        if not transition.requires_target:
            option = (transition.action, None)
            if option not in seen:
                seen.add(option)
                result.append(option)
            continue
        for target_id, population in populations.items():
            if target_id == agent_id:
                continue
            if transition.target_populations and population not in transition.target_populations:
                continue
            option = (transition.action, target_id)
            if option not in seen:
                seen.add(option)
                result.append(option)
    if not result:
        raise ValueError(f"agent {agent_id} has no addressable action options")
    return tuple(result)


def _step_infos(
    result: StepResult,
    agents: list[str],
) -> dict[str, dict[str, object]]:
    events_by_actor = {event.actor_id: event for event in result.events if event.counts_as_action}
    infos = {}
    for agent in agents:
        event = events_by_actor.get(agent)
        infos[agent] = (
            {
                "event_id": event.event_id,
                "action": event.action,
                "target_id": event.target_id,
                "outcome_vector": dict(event.outcomes_by_agent.get(agent, {})),
                "violations": list(event.violations),
                "enforced": event.enforced,
            }
            if event is not None
            else {"failure": "invalid_or_unavailable_action"}
        )
    return infos
