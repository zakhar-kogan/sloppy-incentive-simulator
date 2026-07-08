from __future__ import annotations

import json
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
from icframe.llm import LLMCallRecord, LLMClient, LLMRequest, llm_call_record_from_response
from icframe.observability import stable_trace_id


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
    run_id: str | None = None
    trace_id: str | None = None
    step: int | None = None
    observation_id: str | None = None
    agent_id: str
    state: str
    visible_actions: list[str] = Field(default_factory=list)
    visible_transitions: list[str] = Field(default_factory=list)
    visible_outcomes: dict[str, OutcomeVector] = Field(default_factory=dict)
    visible_prompts: dict[str, str] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(ICFrameModel):
    run_id: str
    trace_id: str
    step: int
    observation_id: str
    policy_decision_id: str
    agent_id: str
    policy_backend: PolicyBackend
    candidate_actions: list[str] = Field(default_factory=list)
    chosen_action: str | None = None
    target_id: str | None = None
    estimated_scalar_rewards: dict[str, float] = Field(default_factory=dict)
    decision_probability: float | None = None
    rationale: str | None = None
    llm_call_id: str | None = None
    failure_mode: str | None = None
    policy_state_delta: dict[str, Any] = Field(default_factory=dict)


class SimulationEvent(ICFrameModel):
    run_id: str
    trace_id: str
    event_id: str
    observation_id: str
    policy_decision_id: str
    constraint_id: str | None = None
    llm_call_id: str | None = None
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
    artifact_refs: dict[str, str] = Field(default_factory=dict)


class SimulationTrace(ICFrameModel):
    run_id: str
    trace_id: str
    spec_name: str
    seed: int
    events: list[SimulationEvent] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    policy_decisions: list[PolicyDecision] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
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
    trace_id: str = "trace_local"
    global_state: dict[str, Scalar] = field(default_factory=dict)
    learned_values: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))


def run_incentive_simulation(
    spec: IncentiveSpec,
    seed: int | None = None,
    observer: Any | None = None,
    llm_client: LLMClient | None = None,
) -> SimulationTrace:
    run_seed = (
        seed if seed is not None else (spec.experiment.seeds[0] if spec.experiment.seeds else 0)
    )
    run_id = f"{spec.spec.name}:{run_seed}"
    world = _RuntimeWorld(
        spec=spec,
        rng=random.Random(run_seed),
        seed=run_seed,
        run_id=run_id,
        trace_id=stable_trace_id(run_id),
        agents=_expand_population(spec),
        global_state={"round": 0},
    )
    events: list[SimulationEvent] = []
    observations: list[Observation] = []
    decisions: list[PolicyDecision] = []
    llm_calls: list[LLMCallRecord] = []
    if observer is not None:
        observer.start_run(spec, run_id=world.run_id, trace_id=world.trace_id, seed=run_seed)
    for step in range(1, spec.experiment.steps + 1):
        world.global_state["round"] = step
        for turn_index, agent in enumerate(_scheduled_agents(world, step), start=1):
            observation = compile_observation(
                spec,
                world,
                agent,
                step=step,
                turn_index=turn_index,
            )
            observations.append(observation)
            if observer is not None:
                observer.record_observation(observation)
            transition, decision, llm_call = _choose_transition(
                world,
                agent,
                observation,
                llm_client=llm_client,
            )
            decisions.append(decision)
            if observer is not None:
                observer.record_policy_decision(decision)
            if llm_call is not None:
                llm_calls.append(llm_call)
                if observer is not None:
                    observer.record_llm_call(llm_call)
            if transition is None:
                continue
            explanation = explain_transition_availability(
                spec,
                actor_id=agent.id,
                state=agent.current_state,
                action=transition.action,
                constraint_id=_constraint_id(step, turn_index),
                policy_decision_id=decision.policy_decision_id,
            )
            if observer is not None:
                observer.record_constraint_explanation(explanation)
            if not explanation.available:
                continue
            event = _execute_transition(
                world,
                agent,
                transition,
                step,
                explanation,
                observation,
                decision,
                turn_index,
            )
            events.append(event)
            if observer is not None:
                observer.record_event(event)
    metric_results = compute_metrics(spec, events)
    trace = SimulationTrace(
        run_id=world.run_id,
        trace_id=world.trace_id,
        spec_name=spec.spec.name,
        seed=run_seed,
        events=events,
        observations=observations,
        policy_decisions=decisions,
        llm_calls=llm_calls,
        final_global_state=world.global_state,
        final_agent_state=world.agents,
        metric_results=metric_results,
    )
    if observer is not None:
        observer.finish_run(trace)
    return trace


def compile_observation(
    spec: IncentiveSpec,
    world: _RuntimeWorld,
    agent: AgentRuntimeState,
    *,
    step: int | None = None,
    turn_index: int = 0,
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
        run_id=world.run_id,
        trace_id=world.trace_id,
        step=step,
        observation_id=(_observation_id(step, turn_index, agent.id) if step is not None else None),
        agent_id=agent.id,
        state=agent.current_state,
        visible_actions=visible_actions,
        visible_transitions=visible_transitions,
        visible_outcomes=visible_outcomes,
        visible_prompts=visible_prompts,
        memory=agent.memory,
    )


def choose_action(
    policy: PolicyBackend,
    observation: Observation,
    action_space: list[str],
    memory: dict[str, Any],
    rng: random.Random,
    *,
    estimated_scalar_rewards: dict[str, float] | None = None,
    behavior: dict[str, float] | None = None,
) -> PolicyDecision:
    """Choose an action over visible action labels and return an auditable decision."""

    estimated = estimated_scalar_rewards or {action: 0.0 for action in action_space}
    behavior = behavior or {}
    chosen_action: str | None = None
    probability: float | None = None

    if action_space:
        if policy is PolicyBackend.EPSILON_GREEDY_BANDIT:
            exploration_rate = behavior.get("exploration_rate", 0.1)
            if rng.random() < exploration_rate:
                chosen_action = rng.choice(action_space)
                probability = exploration_rate / len(action_space)
            else:
                chosen_action = max(action_space, key=lambda action: (estimated[action], action))
                probability = 1.0 - exploration_rate
        elif policy in {
            PolicyBackend.STOCHASTIC_WEIGHTED,
            PolicyBackend.THOMPSON_SAMPLING_BANDIT,
            PolicyBackend.CONTEXTUAL_BANDIT,
            PolicyBackend.UCB_BANDIT,
        }:
            chosen_action = _weighted_action_choice(rng, action_space, estimated)
        else:
            chosen_action = max(action_space, key=lambda action: (estimated[action], action))

    return PolicyDecision(
        run_id=observation.run_id or "run_unknown",
        trace_id=observation.trace_id or "trace_unknown",
        step=observation.step or 0,
        observation_id=observation.observation_id or "observation_unknown",
        policy_decision_id=_decision_id(observation.step or 0, 0, observation.agent_id),
        agent_id=observation.agent_id,
        policy_backend=policy,
        candidate_actions=action_space,
        chosen_action=chosen_action,
        estimated_scalar_rewards=estimated,
        decision_probability=probability,
        policy_state_delta={},
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
    *,
    llm_client: LLMClient | None = None,
) -> tuple[Transition | None, PolicyDecision, LLMCallRecord | None]:
    candidates = [
        transition
        for transition in world.spec.transitions
        if transition.from_state == agent.current_state
        and transition.availability is not Availability.HARD_BLOCKED
    ]
    estimated = {
        transition.action: _expected_scalar_reward(agent, transition) for transition in candidates
    }
    decision = PolicyDecision(
        run_id=world.run_id,
        trace_id=world.trace_id,
        step=observation.step or 0,
        observation_id=observation.observation_id or "observation_unknown",
        policy_decision_id=_decision_id(
            observation.step or 0,
            _turn_index_from_observation(observation),
            agent.id,
        ),
        agent_id=agent.id,
        policy_backend=agent.policy,
        candidate_actions=[transition.action for transition in candidates],
        estimated_scalar_rewards=estimated,
    )
    if not candidates:
        decision.failure_mode = "no_available_candidates"
        return None, decision, None
    if agent.policy in {
        PolicyBackend.LLM_POLICY,
        PolicyBackend.LITELLM_POLICY,
        PolicyBackend.AGNO_POLICY,
    }:
        return _choose_transition_with_llm(
            world,
            agent,
            observation,
            candidates,
            decision,
            llm_client,
        )
    if agent.policy is PolicyBackend.SCRIPTED:
        transition = _prefer_permitted(candidates)
        decision.chosen_action = transition.action
        decision.decision_probability = 1.0
        decision.rationale = "prefer_permitted"
        return transition, decision, None
    if agent.policy is PolicyBackend.EPSILON_GREEDY_BANDIT:
        exploration_rate = agent.behavior.get("exploration_rate", 0.1)
        if world.rng.random() < exploration_rate:
            transition = world.rng.choice(candidates)
            decision.chosen_action = transition.action
            decision.decision_probability = exploration_rate / len(candidates)
            decision.rationale = "epsilon_explore"
            return transition, decision, None
    scored = [(transition, _expected_scalar_reward(agent, transition)) for transition in candidates]
    if agent.policy in {
        PolicyBackend.STOCHASTIC_WEIGHTED,
        PolicyBackend.THOMPSON_SAMPLING_BANDIT,
        PolicyBackend.CONTEXTUAL_BANDIT,
        PolicyBackend.UCB_BANDIT,
    }:
        transition = _weighted_choice(world.rng, scored)
        decision.chosen_action = transition.action
        decision.rationale = agent.policy.value
        return transition, decision, None
    transition = max(scored, key=lambda item: (item[1], item[0].id))[0]
    decision.chosen_action = transition.action
    decision.decision_probability = 1.0 - agent.behavior.get("exploration_rate", 0.0)
    decision.rationale = "max_expected_scalar_reward"
    return transition, decision, None


def _execute_transition(
    world: _RuntimeWorld,
    agent: AgentRuntimeState,
    transition: Transition,
    step: int,
    explanation: ConstraintExplanation,
    observation: Observation,
    decision: PolicyDecision,
    turn_index: int,
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
    event_id = _event_id(step, turn_index, agent.id)
    explanation.event_id = event_id
    return SimulationEvent(
        run_id=world.run_id,
        trace_id=world.trace_id,
        event_id=event_id,
        observation_id=observation.observation_id or "observation_unknown",
        policy_decision_id=decision.policy_decision_id,
        constraint_id=explanation.constraint_id,
        llm_call_id=decision.llm_call_id,
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
        artifact_refs={
            "observation": f"observations.jsonl:{observation.observation_id}",
            "policy_decision": f"policy_decisions.jsonl:{decision.policy_decision_id}",
            "constraint_explanation": f"constraint_explanations.jsonl:{explanation.constraint_id}",
        },
    )


def _choose_transition_with_llm(
    world: _RuntimeWorld,
    agent: AgentRuntimeState,
    observation: Observation,
    candidates: list[Transition],
    decision: PolicyDecision,
    llm_client: LLMClient | None,
) -> tuple[Transition | None, PolicyDecision, LLMCallRecord | None]:
    if llm_client is None:
        transition = _prefer_permitted(candidates)
        decision.chosen_action = transition.action
        decision.failure_mode = "missing_llm_client_fallback"
        decision.rationale = "fallback_prefer_permitted"
        return transition, decision, None

    archetype = world.spec.archetypes[agent.archetype]
    llm_config = archetype.llm
    if llm_config is None:
        transition = _prefer_permitted(candidates)
        decision.chosen_action = transition.action
        decision.failure_mode = "missing_llm_config_fallback"
        decision.rationale = "fallback_prefer_permitted"
        return transition, decision, None

    llm_call_id = f"llm_{decision.policy_decision_id}"
    decision.llm_call_id = llm_call_id
    request = LLMRequest(
        llm_call_id=llm_call_id,
        policy_decision_id=decision.policy_decision_id,
        provider=llm_config.backend,
        model=llm_config.model,
        system_prompt=llm_config.system_prompt,
        prompt=_prompt_from_observation(observation, candidates),
        response_schema=llm_config.response_schema,
        temperature=llm_config.temperature,
        require_json=llm_config.require_json_action,
    )
    try:
        response = llm_client.complete(request)
        action = response.parsed.get(llm_config.action_field)
        if not isinstance(action, str):
            decision.failure_mode = "malformed_llm_output"
            transition = None
        else:
            transition = next(
                (candidate for candidate in candidates if candidate.action == action),
                None,
            )
            if transition is None:
                decision.failure_mode = "llm_chose_invalid_action"
            decision.chosen_action = action
            rationale = response.parsed.get("rationale")
            if isinstance(rationale, str):
                decision.rationale = rationale
    except Exception as exc:
        response = None
        transition = None
        decision.failure_mode = f"llm_error:{type(exc).__name__}"

    if response is None:
        llm_record = LLMCallRecord(
            llm_call_id=llm_call_id,
            policy_decision_id=decision.policy_decision_id,
            provider=llm_config.backend,
            model=llm_config.model,
            request_hash=request.request_hash,
            response_hash="",
            error_type=decision.failure_mode,
            redaction_mode=world.spec.observability.redaction.mode.value,
        )
    else:
        llm_record = llm_call_record_from_response(
            request,
            response,
            redaction_mode=world.spec.observability.redaction.mode.value,
        )
    return transition, decision, llm_record


def _prompt_from_observation(
    observation: Observation,
    candidates: list[Transition],
) -> str:
    payload = {
        "state": observation.state,
        "visible_actions": observation.visible_actions,
        "visible_transitions": observation.visible_transitions,
        "visible_outcomes": observation.visible_outcomes,
        "visible_prompts": observation.visible_prompts,
        "candidate_actions": [candidate.action for candidate in candidates],
    }
    return json.dumps(payload, sort_keys=True)


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


def _weighted_action_choice(
    rng: random.Random,
    actions: list[str],
    estimated: dict[str, float],
) -> str:
    scored = [(action, estimated[action]) for action in actions]
    minimum = min(score for _, score in scored)
    weights = [(score - minimum + 0.001) for _, score in scored]
    total = sum(weights)
    if total <= 0:
        return rng.choice(actions)
    sample = rng.random() * total
    cursor = 0.0
    for (action, _), weight in zip(scored, weights, strict=True):
        cursor += weight
        if sample <= cursor:
            return action
    return actions[-1]


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


def _observation_id(step: int | None, turn_index: int, agent_id: str) -> str:
    return f"obs_{step or 0:04d}_{turn_index:04d}_{agent_id}"


def _decision_id(step: int, turn_index: int, agent_id: str) -> str:
    return f"decision_{step:04d}_{turn_index:04d}_{agent_id}"


def _constraint_id(step: int, turn_index: int) -> str:
    return f"constraint_{step:04d}_{turn_index:04d}"


def _event_id(step: int, turn_index: int, agent_id: str) -> str:
    return f"event_{step:04d}_{turn_index:04d}_{agent_id}"


def _turn_index_from_observation(observation: Observation) -> int:
    if observation.observation_id is None:
        return 0
    parts = observation.observation_id.split("_")
    if len(parts) < 3:
        return 0
    try:
        return int(parts[2])
    except ValueError:
        return 0
