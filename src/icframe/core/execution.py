from __future__ import annotations

import random

from icframe.domain.incentive_spec import EffectScope, NormStatus, Operation, OutcomeVector

from .hooks import ResolvedStatePatch
from .types import (
    CompiledEffect,
    CompiledStateUpdate,
    CompiledTransition,
    RuntimeEvent,
    WorldSnapshot,
)


class TransitionExecutor:
    """Resolve one compiled transition without owning mutable world state."""

    def execute(
        self,
        *,
        run_id: str,
        step: int,
        turn_index: int,
        snapshot: WorldSnapshot,
        agent_id: str,
        transition: CompiledTransition,
        target_id: str | None,
        rng: random.Random,
    ) -> tuple[RuntimeEvent, list[ResolvedStatePatch]]:
        outcomes_by_agent: dict[str, OutcomeVector] = {}
        global_outcome: OutcomeVector = {}
        for effect in transition.effects:
            apply_effect(
                effect,
                snapshot,
                agent_id,
                target_id,
                outcomes_by_agent,
                global_outcome,
            )

        audit_sampled = False
        detected = False
        enforced = False
        violating = transition.norm_status is NormStatus.FORBIDDEN
        if transition.enforcement is not None:
            enforcement = transition.enforcement
            audit_sampled = rng.random() < enforcement.audit_probability
            if audit_sampled:
                if violating:
                    detected = rng.random() < enforcement.detection_probability
                    if detected and enforcement.false_negative_probability:
                        detected = rng.random() >= enforcement.false_negative_probability
                elif enforcement.false_positive_probability:
                    detected = rng.random() < enforcement.false_positive_probability
            if detected:
                enforced = rng.random() < enforcement.enforcement_probability
            selected_effects = (
                enforcement.sanctions
                if enforced
                else enforcement.compliance_rewards
                if not violating
                else ()
            )
            for effect in selected_effects:
                apply_effect(
                    effect,
                    snapshot,
                    agent_id,
                    target_id,
                    outcomes_by_agent,
                    global_outcome,
                )

        patches = resolve_updates(
            transition.state_updates,
            snapshot,
            actor_id=agent_id,
            target_id=target_id,
        )
        event = RuntimeEvent(
            event_id=f"event_{run_id}_{step:08d}_{turn_index:04d}_{agent_id}",
            step=step,
            actor_id=agent_id,
            target_id=target_id,
            transition_id=transition.id,
            action=transition.action,
            from_state=snapshot.agents[agent_id].state,
            to_state=transition.to_state,
            availability=transition.availability,
            norm_status=transition.norm_status,
            tags=transition.tags,
            outcomes_by_agent=outcomes_by_agent,
            global_outcome=global_outcome,
            audit_sampled=audit_sampled,
            detected=detected,
            enforced=enforced,
            explanation_reasons=transition.explanation_reasons,
            violations=("forbidden_action",) if violating else (),
            remediation_actions=(
                transition.enforcement.remediation_actions
                if transition.enforcement is not None
                else ()
            ),
        )
        return event, patches


def apply_effect(
    effect: CompiledEffect,
    snapshot: WorldSnapshot,
    actor_id: str,
    target_id: str | None,
    outcomes_by_agent: dict[str, OutcomeVector],
    global_outcome: OutcomeVector,
) -> None:
    if effect.scope is EffectScope.GLOBAL:
        apply_vector(global_outcome, effect.values, effect.operation)
        return
    for recipient in recipients(effect.scope, effect.population, snapshot, actor_id, target_id):
        apply_vector(
            outcomes_by_agent.setdefault(recipient, {}),
            effect.values,
            effect.operation,
        )


def resolve_updates(
    updates: tuple[CompiledStateUpdate, ...],
    snapshot: WorldSnapshot,
    *,
    actor_id: str,
    target_id: str | None,
) -> list[ResolvedStatePatch]:
    patches = []
    for update in updates:
        addressed = (
            ["__global__"]
            if update.scope is EffectScope.GLOBAL
            else recipients(update.scope, update.population, snapshot, actor_id, target_id)
        )
        patches.extend(
            ResolvedStatePatch(
                target=recipient,
                field=update.field,
                operation=update.operation,
                value=update.value,
            )
            for recipient in addressed
        )
    return patches


def recipients(
    scope: EffectScope,
    population: str | None,
    snapshot: WorldSnapshot,
    actor_id: str,
    target_id: str | None,
) -> list[str]:
    if scope is EffectScope.ACTOR:
        return [actor_id]
    if scope is EffectScope.TARGET:
        if target_id is None:
            raise ValueError("target-scoped behavior requires target_id")
        return [target_id]
    if scope is EffectScope.ALL_AGENTS:
        return sorted(snapshot.agents)
    if scope is EffectScope.POPULATION:
        return sorted(
            agent_id
            for agent_id, agent in snapshot.agents.items()
            if agent.population == population
        )
    raise ValueError(f"scope {scope.value} does not address agents")


def apply_vector(
    target: OutcomeVector,
    values: tuple[tuple[str, float], ...],
    operation: Operation,
) -> None:
    for channel, value in values:
        if operation is Operation.ADD:
            target[channel] = target.get(channel, 0.0) + value
        elif operation is Operation.MULTIPLY:
            target[channel] = target.get(channel, 0.0) * value
        else:
            target[channel] = value
