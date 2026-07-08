from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from icframe.constraints.clingo import ConstraintExplanation, explain_transition_availability
from icframe.domain.base import ICFrameModel, Scalar
from icframe.domain.incentive_spec import (
    Availability,
    ConditionalEffect,
    EffectOperation,
    GraphVisibility,
    IncentiveSpec,
    MetricType,
    NormStatus,
    OutcomeVector,
    OutcomeVisibility,
    PolicyBackend,
    ScheduleMode,
    Transition,
)


class AgentRuntimeState(ICFrameModel):
    id: str
    archetype: str
    role: str
    population: str
    current_state: str
    scalarizer: dict[str, float] = Field(default_factory=dict)
    behavior: dict[str, float] = Field(default_factory=dict)
    policy: PolicyBackend
    visibility_profile: str
    memory: dict[str, Any] = Field(default_factory=dict)


class Observation(ICFrameModel):
    agent_id: str
    state: str
    visible_actions: list[str] = Field(default_factory=list)
    visible_transitions: list[str] = Field(default_factory=list)
    visible_outcomes: dict[str, OutcomeVector] = Field(default_factory=dict)
    visible_prompts: dict[str, str] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)


class SimulationEvent(ICFrameModel):
    run_id: str
    seed: int
    step: int
    actor_id: str
    actor_archetype: str
    actor_role: str
    from_state: str
    action: str
    to_state: str
    availability: Availability
    norm_status: NormStatus
    tags: list[str] = Field(default_factory=list)
    base_effects: OutcomeVector = Field(default_factory=dict)
    conditional_effects_applied: list[str] = Field(default_factory=list)
    audit_sampled: bool = False
    detected: bool = False
    enforced: bool = False
    final_outcome_vector: OutcomeVector = Field(default_factory=dict)
    scalar_rewards: dict[str, float] = Field(default_factory=dict)
    constraint_explanation: ConstraintExplanation | None = None


class SimulationTrace(ICFrameModel):
    run_id: str
    spec_name: str
    seed: int
    events: list[SimulationEvent] = Field(default_factory=list)
    final_global_state: dict[str, Scalar] = Field(default_factory=dict)
    final_agent_state: dict[str, AgentRuntimeState] = Field(default_factory=dict)
    metric_results: dict[str, float] = Field(default_factory=dict)


@dataclass
class _RuntimeWorld:
    spec: IncentiveSpec
    rng: random.Random
    seed: int
    run_id: str
    agents: dict[str, AgentRuntimeState]
    global_state: dict[str, Scalar] = field(default_factory=dict)
    learned_values: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))


def run_incentive_simulation(spec: IncentiveSpec, seed: int | None = None) -> SimulationTrace:
    run_seed = (
        seed if seed is not None else (spec.experiment.seeds[0] if spec.experiment.seeds else 0)
    )
    world = _RuntimeWorld(
        spec=spec,
        rng=random.Random(run_seed),
        seed=run_seed,
        run_id=f"{spec.spec.name}:{run_seed}",
        agents=_expand_population(spec),
        global_state={"round": 0},
    )
    events: list[SimulationEvent] = []
    for step in range(1, spec.experiment.steps + 1):
        world.global_state["round"] = step
        for agent in _scheduled_agents(world, step):
            observation = compile_observation(spec, world, agent)
            transition = _choose_transition(world, agent, observation)
            if transition is None:
                continue
            explanation = explain_transition_availability(
                spec,
                actor_id=agent.id,
                state=agent.current_state,
                action=transition.action,
            )
            if not explanation.available:
                continue
            events.append(_execute_transition(world, agent, transition, step, explanation))
    metric_results = compute_metrics(spec, events)
    return SimulationTrace(
        run_id=world.run_id,
        spec_name=spec.spec.name,
        seed=run_seed,
        events=events,
        final_global_state=world.global_state,
        final_agent_state=world.agents,
        metric_results=metric_results,
    )


def compile_observation(
    spec: IncentiveSpec,
    world: _RuntimeWorld,
    agent: AgentRuntimeState,
) -> Observation:
    profile = spec.visibility_profiles[agent.visibility_profile]
    candidates = [
        transition
        for transition in spec.transitions
        if transition.from_state == agent.current_state
        and transition.availability is not Availability.HARD_BLOCKED
    ]
    visible_transitions: list[str] = []
    visible_actions: list[str] = []
    visible_outcomes: dict[str, OutcomeVector] = {}
    visible_prompts: dict[str, str] = {}

    if profile.graph in {
        GraphVisibility.FULL_GRAPH,
        GraphVisibility.LOCAL_GRAPH,
        GraphVisibility.DISCOVERED_GRAPH,
    }:
        visible_transitions = [transition.id for transition in candidates]
        visible_actions = [transition.action for transition in candidates]
    elif profile.graph in {GraphVisibility.PROMPT_ONLY, GraphVisibility.BLACK_BOX}:
        visible_actions = [transition.action for transition in candidates]

    if profile.graph is GraphVisibility.DISCOVERED_GRAPH:
        known = set(agent.memory.get("known_transitions", ()))
        visible_transitions = [transition.id for transition in candidates if transition.id in known]

    for transition in candidates:
        outcome = _visible_outcome_vector(profile, transition.effects)
        if outcome:
            visible_outcomes[transition.id] = outcome
        if profile.prompts and transition.prompt is not None and transition.prompt.public:
            visible_prompts[transition.id] = transition.prompt.public

    return Observation(
        agent_id=agent.id,
        state=agent.current_state,
        visible_actions=visible_actions,
        visible_transitions=visible_transitions,
        visible_outcomes=visible_outcomes,
        visible_prompts=visible_prompts,
        memory=agent.memory,
    )


def compute_metrics(spec: IncentiveSpec, events: list[SimulationEvent]) -> dict[str, float]:
    results: dict[str, float] = {}
    for name, metric in spec.metrics.items():
        if metric.type in {MetricType.SUM, MetricType.MEAN}:
            values = [
                _channel_value(event.final_outcome_vector, metric.channel) for event in events
            ]
            if metric.type is MetricType.SUM:
                results[name] = sum(values)
            else:
                results[name] = sum(values) / len(values) if values else 0.0
        elif metric.type in {MetricType.DIFFERENCE, MetricType.ZSCORE_DIFFERENCE}:
            results[name] = _resolve_value(metric.proxy, events, results) - _resolve_value(
                metric.target,
                events,
                results,
            )
        elif metric.type is MetricType.RATIO:
            denominator = _resolve_value(metric.denominator, events, results)
            numerator = _resolve_value(metric.numerator, events, results)
            results[name] = numerator / denominator if denominator else 0.0
        elif metric.type is MetricType.EVENT_COUNT:
            results[name] = float(_matching_event_count(events, metric.where_tags_include))
        elif metric.type in {MetricType.EVENT_RATE, MetricType.RATE}:
            denominator = (
                len(events)
                if metric.denominator in {None, "all_actions"}
                else _resolve_value(
                    metric.denominator,
                    events,
                    results,
                )
            )
            results[name] = (
                _matching_event_count(events, metric.where_tags_include) / denominator
                if denominator
                else 0.0
            )
        else:
            results[name] = 0.0
    return results


def _expand_population(spec: IncentiveSpec) -> dict[str, AgentRuntimeState]:
    agents: dict[str, AgentRuntimeState] = {}
    for entry in spec.population:
        archetype = spec.archetypes[entry.archetype]
        for index in range(entry.count):
            agent_id = f"{entry.archetype}_{index:03d}"
            agents[agent_id] = AgentRuntimeState(
                id=agent_id,
                archetype=entry.archetype,
                role=archetype.role,
                population=entry.archetype,
                current_state=archetype.initial_state or spec.states.initial_global,
                scalarizer=archetype.scalarizer,
                behavior=archetype.behavior,
                policy=archetype.policy,
                visibility_profile=archetype.visibility_profile,
                memory={"known_transitions": ()},
            )
    return agents


def _scheduled_agents(world: _RuntimeWorld, step: int) -> list[AgentRuntimeState]:
    agents = list(world.agents.values())
    if world.spec.experiment.schedule is ScheduleMode.SEQUENTIAL_FIXED:
        return agents
    if world.spec.experiment.schedule is ScheduleMode.SEQUENTIAL_RANDOM:
        world.rng.shuffle(agents)
        return agents
    return agents


def _choose_transition(
    world: _RuntimeWorld,
    agent: AgentRuntimeState,
    observation: Observation,
) -> Transition | None:
    candidates = [
        transition
        for transition in world.spec.transitions
        if transition.from_state == agent.current_state
        and transition.availability is not Availability.HARD_BLOCKED
    ]
    if not candidates:
        return None
    if agent.policy is PolicyBackend.SCRIPTED:
        return _prefer_permitted(candidates)
    if agent.policy is PolicyBackend.EPSILON_GREEDY_BANDIT:
        exploration_rate = agent.behavior.get("exploration_rate", 0.1)
        if world.rng.random() < exploration_rate:
            return world.rng.choice(candidates)
    scored = [(transition, _expected_scalar_reward(agent, transition)) for transition in candidates]
    if agent.policy is PolicyBackend.STOCHASTIC_WEIGHTED:
        return _weighted_choice(world.rng, scored)
    return max(scored, key=lambda item: (item[1], item[0].id))[0]


def _execute_transition(
    world: _RuntimeWorld,
    agent: AgentRuntimeState,
    transition: Transition,
    step: int,
    explanation: ConstraintExplanation,
) -> SimulationEvent:
    from_state = agent.current_state
    outcome = dict(transition.effects)
    applied: list[str] = []
    for conditional in sorted(transition.conditional_effects, key=lambda item: item.priority):
        if _selector_matches(agent, conditional):
            _apply_effect_operation(outcome, conditional.effects, conditional.operation)
            if conditional.effects:
                applied.append(f"{transition.id}:conditional:{conditional.priority}")

    audit_sampled = False
    detected = False
    enforced = False
    if transition.enforcement is not None:
        audit_sampled = world.rng.random() < transition.enforcement.audit_probability
        if audit_sampled:
            detected = world.rng.random() < transition.enforcement.detection_probability
            if detected and transition.enforcement.false_negative_probability:
                detected = world.rng.random() >= transition.enforcement.false_negative_probability
            if not detected and transition.enforcement.false_positive_probability:
                detected = world.rng.random() < transition.enforcement.false_positive_probability
        if detected:
            enforced = world.rng.random() < transition.enforcement.enforcement_probability
        if enforced:
            _apply_effect_operation(
                outcome,
                transition.enforcement.sanction_if_detected,
                EffectOperation.ADD,
            )
            for conditional in sorted(
                transition.conditional_effects,
                key=lambda item: item.priority,
            ):
                if _selector_matches(agent, conditional):
                    _apply_effect_operation(
                        outcome,
                        conditional.effects_if_detected,
                        conditional.operation,
                    )
                    if conditional.effects_if_detected:
                        applied.append(f"{transition.id}:detected:{conditional.priority}")
        elif (
            transition.norm_status is NormStatus.PERMITTED
            and transition.enforcement.reward_if_compliant
        ):
            _apply_effect_operation(
                outcome,
                transition.enforcement.reward_if_compliant,
                EffectOperation.ADD,
            )

    agent.current_state = transition.to_state
    known = set(agent.memory.get("known_transitions", ()))
    known.add(transition.id)
    agent.memory["known_transitions"] = tuple(sorted(known))
    scalar_reward = _scalarize(agent.scalarizer, outcome)
    world.learned_values[agent.id][transition.id] = scalar_reward
    return SimulationEvent(
        run_id=world.run_id,
        seed=world.seed,
        step=step,
        actor_id=agent.id,
        actor_archetype=agent.archetype,
        actor_role=agent.role,
        from_state=from_state,
        action=transition.action,
        to_state=transition.to_state,
        availability=transition.availability,
        norm_status=transition.norm_status,
        tags=transition.tags,
        base_effects=transition.effects,
        conditional_effects_applied=applied,
        audit_sampled=audit_sampled,
        detected=detected,
        enforced=enforced,
        final_outcome_vector=outcome,
        scalar_rewards={agent.id: scalar_reward},
        constraint_explanation=explanation,
    )


def _visible_outcome_vector(profile, outcome: OutcomeVector) -> OutcomeVector:
    visible: OutcomeVector = {}
    for channel, value in outcome.items():
        if (
            channel.startswith("observed.")
            and profile.observed_outcomes is OutcomeVisibility.FULL_NUMERIC
        ):
            visible[channel] = value
        elif (
            channel.startswith("latent.")
            and profile.latent_outcomes is OutcomeVisibility.FULL_NUMERIC
        ):
            visible[channel] = value
        elif (
            channel.startswith("governance.")
            and profile.governance_outcomes is OutcomeVisibility.FULL_NUMERIC
        ):
            visible[channel] = value
        elif (
            channel.startswith("agent.")
            and profile.observed_outcomes is OutcomeVisibility.FULL_NUMERIC
        ):
            visible[channel] = value
        elif (
            channel.startswith("social.")
            and profile.observed_outcomes is OutcomeVisibility.FULL_NUMERIC
        ):
            visible[channel] = value
    return visible


def _expected_scalar_reward(agent: AgentRuntimeState, transition: Transition) -> float:
    return _scalarize(agent.scalarizer, transition.effects)


def _scalarize(weights: dict[str, float], outcome: OutcomeVector) -> float:
    return sum(weights.get(channel, 0.0) * value for channel, value in outcome.items())


def _prefer_permitted(candidates: list[Transition]) -> Transition:
    for status in (
        NormStatus.PERMITTED,
        NormStatus.UNKNOWN,
        NormStatus.DISCOURAGED,
        NormStatus.FORBIDDEN,
    ):
        for transition in candidates:
            if transition.norm_status is status:
                return transition
    return candidates[0]


def _weighted_choice(rng: random.Random, scored: list[tuple[Transition, float]]) -> Transition:
    minimum = min(score for _, score in scored)
    weights = [(score - minimum + 0.001) for _, score in scored]
    total = sum(weights)
    if total <= 0:
        return rng.choice([transition for transition, _ in scored])
    sample = rng.random() * total
    cursor = 0.0
    for (transition, _), weight in zip(scored, weights, strict=True):
        cursor += weight
        if sample <= cursor:
            return transition
    return scored[-1][0]


def _selector_matches(agent: AgentRuntimeState, conditional: ConditionalEffect) -> bool:
    selector = conditional.selector.actor
    if selector is None:
        return True
    if selector.role is not None and not _matches_one_or_many(agent.role, selector.role):
        return False
    if selector.archetype is not None and not _matches_one_or_many(
        agent.archetype,
        selector.archetype,
    ):
        return False
    if selector.population is not None and not _matches_one_or_many(
        agent.population,
        selector.population,
    ):
        return False
    return True


def _matches_one_or_many(value: str, expected: str | list[str]) -> bool:
    if isinstance(expected, str):
        return value == expected
    return value in expected


def _apply_effect_operation(
    outcome: OutcomeVector,
    effects: OutcomeVector,
    operation: EffectOperation,
) -> None:
    for channel, value in effects.items():
        if operation is EffectOperation.ADD:
            outcome[channel] = outcome.get(channel, 0.0) + value
        elif operation is EffectOperation.MULTIPLY:
            outcome[channel] = outcome.get(channel, 0.0) * value
        elif operation is EffectOperation.SET:
            outcome[channel] = value


def _channel_value(outcome: OutcomeVector, channel: str | None) -> float:
    return outcome.get(channel or "", 0.0)


def _resolve_value(
    reference: str | None,
    events: list[SimulationEvent],
    metric_results: dict[str, float],
) -> float:
    if reference is None:
        return 0.0
    if reference.startswith("metric."):
        return metric_results.get(reference.removeprefix("metric."), 0.0)
    if reference == "all_actions":
        return float(len(events))
    return sum(event.final_outcome_vector.get(reference, 0.0) for event in events)


def _matching_event_count(events: list[SimulationEvent], tags: list[str]) -> int:
    if not tags:
        return len(events)
    required = set(tags)
    return sum(1 for event in events if required.issubset(event.tags))
