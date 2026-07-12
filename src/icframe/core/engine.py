from __future__ import annotations

import math
import random
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import (
    Availability,
    ConstraintOperator,
    GraphVisibility,
    NormStatus,
    Operation,
    RetentionProfile,
    ScheduleMode,
)
from icframe.domain.run import (
    AgentResult,
    Checkpoint,
    ConstraintResult,
    RunConfig,
    RunStatus,
    RunSummary,
)
from icframe.llm import LLMClient

from .compiler import RuntimePlan, compile_runtime, trusted_evaluation_hash
from .execution import TransitionExecutor
from .hooks import CommitContext, HookResult, InitContext, ResolvedStatePatch, StepContext
from .metrics import OnlineMetrics
from .observer import NoopObserver, RunObserver
from .packs import LoadedDomainPack, apply_parameters
from .policies import Policy
from .projection import ObservationProjector
from .types import (
    AgentState,
    CompiledTransition,
    Observation,
    PolicyChoice,
    PolicyDecision,
    PolicyFeedback,
    RuntimeEvent,
    StepResult,
    WorldSnapshot,
    WorldState,
)


class RuntimeEngine:
    """Compiled, bounded-memory IncentiveSpec execution engine."""

    def __init__(
        self,
        plan: RuntimePlan,
        *,
        run_id: str,
        seed: int,
        llm_client: LLMClient | None = None,
        observer: RunObserver | None = None,
        retention: RetentionProfile | None = None,
        sample_every_steps: int | None = None,
        parameters: dict[str, Scalar] | None = None,
    ) -> None:
        self.plan = plan
        self.run_id = run_id
        self.seed = seed
        self.rng = random.Random(seed)
        self.observer = observer or NoopObserver()
        self.retention = retention or plan.spec.observability.retention
        self.sample_every_steps = (
            sample_every_steps
            or plan.spec.observability.sample_every_steps
            or max(
                1,
                math.ceil(plan.spec.experiment.steps / plan.spec.observability.max_checkpoints),
            )
        )
        self.metrics = OnlineMetrics(plan.metric_reducers)
        self.checkpoints: list[Checkpoint] = []
        self.policies: dict[str, Policy | None] = {}
        self.llm_client = llm_client
        self.llm_calls = 0
        self.estimated_llm_cost = 0.0
        self.replayable = True
        self.replay_reason: str | None = None
        self.parameters = dict(parameters or {})
        self._session_started = False
        self._session_finished = False
        self._session_started_at = 0.0
        self._observation_projector = ObservationProjector(plan)
        self._transition_executor = TransitionExecutor()
        self.world = self._initial_world()
        self._initialize_policies()
        self._apply_hook_result(
            self.plan.hooks.initialize(
                InitContext(
                    snapshot=self.world.snapshot(),
                    hook_config=dict(self.plan.spec.hook_config),
                    rng=self.rng,
                )
            )
        )

    def observations(self) -> dict[str, Observation]:
        return {agent_id: self.observe(agent_id) for agent_id in self.world.agents}

    def observe(
        self,
        agent_id: str,
        *,
        snapshot: WorldSnapshot | None = None,
    ) -> Observation:
        snapshot = snapshot or self.world.snapshot()
        return self._observation_projector.project(
            run_id=self.run_id,
            agent_id=agent_id,
            snapshot=snapshot,
            agent_state=self.world.agents[agent_id],
        )

    def action_mask(self, agent_id: str) -> list[int]:
        actions = {item.action for item in self.observe(agent_id).candidates}
        return [int(action in actions) for action in self.plan.spec.actions.all]

    def run(self) -> RunSummary:
        self.start_session()
        error = None
        status = RunStatus.COMPLETED
        try:
            while self.world.step < self.plan.spec.experiment.steps:
                if self.observer.cancelled():
                    status = RunStatus.CANCELLED
                    break
                result = self.step_internal()
                if result.terminated:
                    break
        except Exception as exc:
            status = RunStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"
        summary = self.finish_session(status=status, error=error)
        if error is not None:
            raise RuntimeError(error)
        return summary

    def start_session(self) -> None:
        if self._session_started:
            return
        started_at = time.perf_counter()
        self.observer.start(
            {
                "run_id": self.run_id,
                "pack_id": self.plan.pack_id,
                "pack_path": self.plan.pack_path,
                "spec": self.plan.spec.model_dump(mode="json", by_alias=True),
                "seed": self.seed,
                "hook_hash": self.plan.hook_hash,
                "runtime_hash": self.plan.runtime_hash,
                "trusted_evaluation_hash": self.plan.trusted_evaluation_hash,
                "parameters": self.parameters,
                "retention": self.retention.value,
                "sample_every_steps": self.sample_every_steps,
            }
        )
        # Starting the observer is the transactional boundary. In particular, a
        # pre-existing artifact directory must fail before this engine becomes
        # finishable and can overwrite that directory.
        self._session_started_at = started_at
        self._session_started = True

    def finish_session(
        self,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        error: str | None = None,
    ) -> RunSummary:
        if self._session_finished:
            raise RuntimeError("runtime session is already finished")
        self.start_session()
        summary = self.summary(
            status=status,
            duration_seconds=time.perf_counter() - self._session_started_at,
            error=error,
        )
        self.observer.finish(summary)
        self._session_finished = True
        return summary

    def step_internal(self) -> StepResult:
        external_agents = [agent_id for agent_id, policy in self.policies.items() if policy is None]
        if external_agents:
            raise RuntimeError(
                f"external policies require adapter actions: {sorted(external_agents)}"
            )
        return self._step(actions=None)

    def step_external(
        self,
        actions: dict[str, tuple[str, str | None]],
    ) -> StepResult:
        self.start_session()
        missing = set(self.world.agents) - set(actions)
        if missing:
            raise ValueError(f"external step is missing agents: {sorted(missing)}")
        return self._step(actions=actions)

    def summary(
        self,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        duration_seconds: float = 0.0,
        error: str | None = None,
    ) -> RunSummary:
        if trusted_evaluation_hash(self.plan.spec) != self.plan.trusted_evaluation_hash:
            raise RuntimeError("trusted evaluation changed after runtime compilation")
        metric_values = self.metrics.snapshot()
        objectives = {
            name: metric_values[objective.metric]
            for name, objective in self.plan.spec.evaluation.objectives.items()
        }
        constraints = []
        for constraint in self.plan.constraint_templates:
            value = metric_values[constraint.metric]
            passed = (
                value <= constraint.threshold
                if constraint.operator is ConstraintOperator.LE
                else value >= constraint.threshold
            )
            constraints.append(
                ConstraintResult(
                    metric=constraint.metric,
                    value=value,
                    threshold=constraint.threshold,
                    operator=constraint.operator.value,
                    passed=passed,
                )
            )
        agents = []
        for agent_id, agent in sorted(self.world.agents.items()):
            policy = self.policies[agent_id]
            agents.append(
                AgentResult(
                    id=agent.id,
                    archetype=agent.archetype,
                    role=agent.role,
                    state=agent.state,
                    resources=dict(agent.resources),
                    policy=agent.policy_kind.value,
                    policy_state=policy.snapshot() if policy is not None else {},
                )
            )
        return RunSummary(
            run_id=self.run_id,
            pack_id=self.plan.pack_id,
            spec_name=self.plan.spec.spec.name,
            seed=self.seed,
            parameters=self.parameters,
            status=status,
            retention=self.retention,
            steps_planned=self.plan.spec.experiment.steps,
            steps_completed=self.world.step,
            event_count=self.metrics.event_count,
            metrics=metric_values,
            objectives=objectives,
            constraints=constraints,
            feasible=all(item.passed for item in constraints),
            action_counts=dict(sorted(self.metrics.action_counts.items())),
            tag_counts=dict(sorted(self.metrics.tag_counts.items())),
            checkpoints=self.checkpoints,
            agents=agents,
            llm_calls=self.llm_calls,
            estimated_llm_cost_usd=self.estimated_llm_cost,
            replayable=self.replayable,
            replay_reason=self.replay_reason,
            duration_seconds=duration_seconds,
            error=error,
        )

    def _step(
        self,
        actions: dict[str, tuple[str, str | None]] | None,
    ) -> StepResult:
        self.world.step += 1
        self._apply_hook_result(
            self.plan.hooks.before_step(
                StepContext(
                    snapshot=self.world.snapshot(),
                    hook_config=dict(self.plan.spec.hook_config),
                    rng=self.rng,
                )
            )
        )
        before = self.world.snapshot()
        if self.plan.spec.experiment.schedule is ScheduleMode.PARALLEL_SIMULTANEOUS:
            observations, decisions, selected = self._collect_parallel(before, actions)
            events = self._commit_parallel(before, selected, decisions)
        else:
            observations, decisions, events = self._commit_sequential(actions)

        return self._finish_committed_step(
            before=before,
            observations=observations,
            decisions=decisions,
            events=events,
        )

    def _finish_committed_step(
        self,
        *,
        before: WorldSnapshot,
        observations: list[Observation],
        decisions: list[PolicyDecision],
        events: list[RuntimeEvent],
    ) -> StepResult:
        after_transitions = self.world.snapshot()
        hook_result = self.plan.hooks.after_commit(
            CommitContext(
                before=before,
                after=after_transitions,
                events=tuple(events),
                hook_config=dict(self.plan.spec.hook_config),
                rng=self.rng,
            )
        )
        self._apply_hook_result(hook_result)
        hook_event = self._hook_event(hook_result)
        if hook_event is not None:
            events.append(hook_event)

        rewards = self._finalize_rewards(events)
        self._learn(decisions, observations, events, rewards)
        for decision in decisions:
            self.observer.decision(decision)
        for event in events:
            self.metrics.update(event)
            self.observer.event(event)
        self._update_histories(events)
        if self.retention is not RetentionProfile.TRAINING and self._should_checkpoint(
            self.world.step
        ):
            checkpoint = Checkpoint(
                step=self.world.step,
                metrics=self.metrics.snapshot(),
                action_counts=dict(sorted(self.metrics.action_counts.items())),
                tag_counts=dict(sorted(self.metrics.tag_counts.items())),
            )
            self.checkpoints.append(checkpoint)
            self.observer.checkpoint(checkpoint)

        terminated = self.plan.hooks.is_terminal(
            StepContext(
                snapshot=self.world.snapshot(),
                hook_config=dict(self.plan.spec.hook_config),
                rng=self.rng,
            )
        )
        return StepResult(
            step=self.world.step,
            events=events,
            decisions=decisions,
            observations=observations,
            rewards=rewards,
            terminated=terminated,
        )

    def _collect_parallel(
        self,
        snapshot: WorldSnapshot,
        actions: dict[str, tuple[str, str | None]] | None,
    ) -> tuple[
        list[Observation],
        list[PolicyDecision],
        list[tuple[str, CompiledTransition, str | None]],
    ]:
        observations = []
        decisions = []
        selected = []
        for agent_id in sorted(snapshot.agents):
            observation = self.observe(agent_id, snapshot=snapshot)
            decision, transition, target_id = self._decide(agent_id, observation, actions)
            observations.append(observation)
            decisions.append(decision)
            self.observer.observation(observation)
            if transition is not None:
                selected.append((agent_id, transition, target_id))
        return observations, decisions, selected

    def _commit_parallel(
        self,
        snapshot: WorldSnapshot,
        selected: list[tuple[str, CompiledTransition, str | None]],
        decisions: list[PolicyDecision],
    ) -> list[RuntimeEvent]:
        decisions_by_agent = {item.agent_id: item for item in decisions}
        events = []
        patches: list[ResolvedStatePatch] = []
        next_states: dict[str, str] = {}
        for index, (agent_id, transition, target_id) in enumerate(selected, start=1):
            event, event_patches = self._transition_executor.execute(
                run_id=self.run_id,
                step=self.world.step,
                turn_index=index,
                snapshot=snapshot,
                agent_id=agent_id,
                transition=transition,
                target_id=target_id,
                rng=self.rng,
            )
            event.scalar_rewards = {}
            events.append(event)
            patches.extend(event_patches)
            next_states[agent_id] = transition.to_state
            decisions_by_agent[agent_id].state_delta["transition"] = transition.id
        for patch in patches:
            self._apply_patch(patch)
        for agent_id, next_state in next_states.items():
            self.world.agents[agent_id].state = next_state
        return events

    def _commit_sequential(
        self,
        actions: dict[str, tuple[str, str | None]] | None,
    ) -> tuple[list[Observation], list[PolicyDecision], list[RuntimeEvent]]:
        order = list(self.world.agents)
        if self.plan.spec.experiment.schedule is ScheduleMode.SEQUENTIAL_RANDOM:
            self.rng.shuffle(order)
        observations = []
        decisions = []
        events = []
        for index, agent_id in enumerate(order, start=1):
            snapshot = self.world.snapshot()
            observation = self.observe(agent_id, snapshot=snapshot)
            decision, transition, target_id = self._decide(agent_id, observation, actions)
            observations.append(observation)
            decisions.append(decision)
            self.observer.observation(observation)
            if transition is None:
                continue
            event, patches = self._transition_executor.execute(
                run_id=self.run_id,
                step=self.world.step,
                turn_index=index,
                snapshot=snapshot,
                agent_id=agent_id,
                transition=transition,
                target_id=target_id,
                rng=self.rng,
            )
            for patch in patches:
                self._apply_patch(patch)
            self.world.agents[agent_id].state = transition.to_state
            decision.state_delta["transition"] = transition.id
            events.append(event)
        return observations, decisions, events

    def _decide(
        self,
        agent_id: str,
        observation: Observation,
        external_actions: dict[str, tuple[str, str | None]] | None,
    ) -> tuple[PolicyDecision, CompiledTransition | None, str | None]:
        agent = self.world.agents[agent_id]
        if external_actions is not None:
            action, target_id = external_actions[agent_id]
            choice = PolicyChoice(action=action, target_id=target_id, rationale="external")
        else:
            policy = self.policies[agent_id]
            if policy is None:
                choice = PolicyChoice(failure="missing_external_action")
            else:
                choice = policy.choose_action(observation, self.rng)
        decision = PolicyDecision(
            decision_id=f"decision_{self.run_id}_{self.world.step:08d}_{agent_id}",
            observation_id=observation.observation_id,
            step=self.world.step,
            agent_id=agent_id,
            policy=agent.policy_kind,
            candidate_keys=tuple(item.key for item in observation.candidates),
            action=choice.action,
            target_id=choice.target_id,
            estimated_rewards=choice.estimated_rewards,
            probability=choice.probability,
            rationale=choice.rationale,
            failure=choice.failure,
            llm_call=choice.llm_call,
        )
        if choice.llm_call is not None and choice.llm_call.get("performed", True):
            self.llm_calls += 1
            self.estimated_llm_cost += float(choice.llm_call.get("estimated_cost", 0.0) or 0.0)
        if choice.failure or choice.action is None:
            return decision, None, choice.target_id
        candidate = next(
            (
                item
                for item in observation.candidates
                if item.action == choice.action and item.target_id == choice.target_id
            ),
            None,
        )
        if candidate is None:
            decision.failure = "invalid_action"
            return decision, None, choice.target_id
        transition = self.plan.transitions_by_state_action[(observation.state, choice.action)]
        return decision, transition, choice.target_id

    def _apply_hook_result(self, result: HookResult) -> None:
        self._validate_hook_result(result)
        for patch in result.state_patches:
            self._apply_patch(patch)

    def _hook_event(self, result: HookResult) -> RuntimeEvent | None:
        if not result.outcomes_by_agent and not result.global_outcome and not result.diagnostics:
            return None
        return RuntimeEvent(
            event_id=f"hook_{self.run_id}_{self.world.step:08d}",
            step=self.world.step,
            actor_id="__domain__",
            target_id=None,
            transition_id="__after_commit__",
            action="__after_commit__",
            from_state="__domain__",
            to_state="__domain__",
            availability=Availability.HARD_AVAILABLE,
            norm_status=NormStatus.UNKNOWN,
            tags=("domain_hook",),
            outcomes_by_agent={
                agent_id: dict(outcome) for agent_id, outcome in result.outcomes_by_agent.items()
            },
            global_outcome=dict(result.global_outcome),
            counts_as_action=False,
        )

    def _validate_hook_result(self, result: HookResult) -> None:
        unknown_agents = set(result.outcomes_by_agent) - set(self.world.agents)
        if unknown_agents:
            raise ValueError(f"domain hook emitted unknown agents: {sorted(unknown_agents)}")
        channels = set(self.plan.spec.outcome_space.channels)
        unknown_channels = set(result.global_outcome) - channels
        for outcome in result.outcomes_by_agent.values():
            unknown_channels.update(set(outcome) - channels)
        if unknown_channels:
            raise ValueError(f"domain hook emitted unknown channels: {sorted(unknown_channels)}")
        for patch in result.state_patches:
            if patch.target != "__global__" and patch.target not in self.world.agents:
                raise ValueError(f"domain hook emitted unknown patch target {patch.target!r}")

    def _apply_patch(self, patch: ResolvedStatePatch) -> None:
        if patch.target == "__global__":
            target: dict[str, Any] = self.world.global_values
        else:
            agent = self.world.agents[patch.target]
            target = {"resources": agent.resources, "attributes": agent.attributes}
        _apply_path(target, patch.field, patch.operation, patch.value)

    def _finalize_rewards(self, events: list[RuntimeEvent]) -> dict[str, float]:
        totals = {agent_id: 0.0 for agent_id in self.world.agents}
        for event in events:
            rewards = {}
            for agent_id, outcome in event.outcomes_by_agent.items():
                scalarizer = self.world.agents[agent_id].scalarizer
                reward = sum(
                    scalarizer.get(channel, 0.0) * value for channel, value in outcome.items()
                )
                rewards[agent_id] = reward
                totals[agent_id] += reward
            event.scalar_rewards = rewards
        return totals

    def _learn(
        self,
        decisions: list[PolicyDecision],
        observations: list[Observation],
        events: list[RuntimeEvent],
        rewards: dict[str, float],
    ) -> None:
        observations_by_id = {item.observation_id: item for item in observations}
        events_by_actor = {item.actor_id: item for item in events if item.counts_as_action}
        for decision in decisions:
            policy = self.policies.get(decision.agent_id)
            event = events_by_actor.get(decision.agent_id)
            if policy is None or event is None or decision.action is None:
                continue
            observation = observations_by_id[decision.observation_id]
            delta = policy.learn(
                PolicyFeedback(
                    state=event.from_state,
                    action=event.action,
                    target_id=event.target_id,
                    reward=rewards.get(decision.agent_id, 0.0),
                    next_state=event.to_state,
                    observation=observation,
                )
            )
            decision.state_delta.update(delta)

    def _update_histories(self, events: list[RuntimeEvent]) -> None:
        for agent_id, state in self.world.agents.items():
            profile = self.plan.visibility[state.visibility_profile]
            if not profile.history_events or profile.graph is GraphVisibility.NONE:
                continue
            for event in events:
                local = (
                    event.actor_id == agent_id
                    or event.target_id == agent_id
                    or agent_id in event.outcomes_by_agent
                )
                if (
                    profile.graph in {GraphVisibility.LOCAL, GraphVisibility.PROMPT_ONLY}
                    and not local
                ):
                    continue
                state.history.append(
                    {
                        "step": event.step,
                        "actor_id": event.actor_id,
                        "target_id": event.target_id,
                        "action": event.action,
                        "tags": list(event.tags),
                        "own_outcome": event.outcomes_by_agent.get(agent_id, {}),
                    }
                )
            while len(state.history) > profile.history_events:
                state.history.popleft()

    def _should_checkpoint(self, step: int) -> bool:
        return (
            step == 1
            or step == self.plan.spec.experiment.steps
            or step % self.sample_every_steps == 0
        )

    def _initial_world(self) -> WorldState:
        agents = {}
        for entry in self.plan.spec.population:
            archetype = self.plan.spec.archetypes[entry.archetype]
            history_limit = self.plan.visibility[archetype.visibility_profile].history_events
            for index in range(entry.count):
                agent_id = f"{entry.archetype}_{index:03d}"
                agents[agent_id] = AgentState(
                    id=agent_id,
                    archetype=entry.archetype,
                    population=entry.archetype,
                    role=archetype.role,
                    state=archetype.initial_state or self.plan.spec.states.initial,
                    resources=dict(archetype.initial_resources),
                    attributes=dict(archetype.attributes),
                    scalarizer=dict(archetype.scalarizer),
                    policy_kind=archetype.policy,
                    visibility_profile=archetype.visibility_profile,
                    history=deque(maxlen=history_limit or None),
                )
        return WorldState(
            step=0,
            global_values=dict(self.plan.spec.states.global_values),
            agents=agents,
        )

    def _initialize_policies(self) -> None:
        for agent_id, agent in self.world.agents.items():
            factory = self.plan.policy_factories[agent.archetype]
            self.policies[agent_id] = factory.create(self.llm_client)


def run_experiment(
    pack: LoadedDomainPack | str | Path,
    config: RunConfig | None = None,
    *,
    llm_client: LLMClient | None = None,
    observer: RunObserver | None = None,
) -> RunSummary:
    loaded = load_pack_if_needed(pack)
    config = config or RunConfig()
    effective = apply_parameters(loaded, config.parameters)
    plan = compile_runtime(effective)
    seed = config.seed if config.seed is not None else plan.spec.experiment.seeds[0]
    run_id = config.run_id or f"run_{uuid.uuid4().hex[:12]}"
    retention = config.retention or plan.spec.observability.retention
    if observer is None:
        from icframe.artifacts import ArtifactObserver

        observer = ArtifactObserver(config.artifact_root, run_id, retention)
    engine = RuntimeEngine(
        plan,
        run_id=run_id,
        seed=seed,
        llm_client=llm_client,
        observer=observer,
        retention=retention,
        sample_every_steps=config.sample_every_steps,
        parameters=config.parameters,
    )
    return engine.run()


def load_pack_if_needed(pack: LoadedDomainPack | str | Path) -> LoadedDomainPack:
    if isinstance(pack, LoadedDomainPack):
        return pack
    from .packs import load_domain_pack

    return load_domain_pack(pack)


def _apply_path(
    target: dict[str, Any],
    path: tuple[str, ...],
    operation: Operation,
    value: Scalar,
) -> None:
    cursor = target
    for segment in path[:-1]:
        child = cursor.get(segment)
        if child is None:
            child = {}
            cursor[segment] = child
        if not isinstance(child, dict):
            raise ValueError(f"state path {'/'.join(path)} crosses a scalar")
        cursor = child
    leaf = path[-1]
    if operation is Operation.SET:
        cursor[leaf] = value
        return
    current = cursor.get(leaf, 0.0)
    if not isinstance(current, int | float) or not isinstance(value, int | float):
        raise ValueError(f"numeric state operation at {'/'.join(path)} has nonnumeric data")
    cursor[leaf] = current + value if operation is Operation.ADD else current * value
