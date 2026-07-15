from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from icframe.domain.incentive_spec import (
    DomainPackManifest,
    EffectScope,
    IncentiveSpec,
    MetricSpec,
    MetricType,
    Operation,
    ScheduleMode,
)

from .hooks import DomainHooks
from .packs import LoadedDomainPack
from .policies import PolicyFactory
from .types import (
    CompiledConstraint,
    CompiledEffect,
    CompiledEnforcement,
    CompiledMetric,
    CompiledStateUpdate,
    CompiledTransition,
    CompiledVisibility,
)


class CompilationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    pack_id: str
    pack_path: str
    pack_manifest: DomainPackManifest
    spec: IncentiveSpec
    hook_hash: str
    runtime_hash: str
    trusted_evaluation_hash: str
    hooks: DomainHooks
    transitions: tuple[CompiledTransition, ...]
    transitions_by_state: dict[str, tuple[CompiledTransition, ...]]
    transitions_by_state_action: dict[tuple[str, str], CompiledTransition]
    visibility: dict[str, CompiledVisibility]
    metric_order: tuple[str, ...]
    metric_reducers: tuple[CompiledMetric, ...]
    constraint_templates: tuple[CompiledConstraint, ...]
    policy_factories: dict[str, PolicyFactory]


def compile_runtime(pack: LoadedDomainPack) -> RuntimePlan:
    spec = pack.spec
    symbolic_blocked: set[str] = set()
    symbolic_reasons: dict[str, tuple[str, ...]] = {}
    if spec.symbolic.enabled:
        from icframe.symbolic import compile_symbolic

        symbolic = compile_symbolic(spec)
        symbolic_blocked = symbolic.blocked
        symbolic_reasons = symbolic.reasons

    transitions: list[CompiledTransition] = []
    by_key: dict[tuple[str, str], CompiledTransition] = {}
    by_state_mutable: dict[str, list[CompiledTransition]] = {state: [] for state in spec.states.all}
    for transition in spec.transitions:
        key = (transition.from_state, transition.action)
        if key in by_key:
            raise CompilationError(
                f"state/action pair {key!r} has multiple transitions; v0.4 requires one"
            )
        if (
            any(
                item.scope is EffectScope.TARGET
                for item in [*transition.effects, *transition.state_updates]
            )
            and not transition.requires_target
        ):
            raise CompilationError(
                f"transition {transition.id} has target-scoped behavior without requires_target"
            )
        for update in transition.state_updates:
            if update.scope is not EffectScope.GLOBAL and update.field[0] not in {
                "resources",
                "attributes",
            }:
                raise CompilationError(
                    f"agent state update in {transition.id} must begin with resources or attributes"
                )
        compiled = _compile_transition(
            transition,
            symbolic_blocked=symbolic_blocked,
            symbolic_reasons=symbolic_reasons,
        )
        transitions.append(compiled)
        by_key[key] = compiled
        by_state_mutable[compiled.from_state].append(compiled)

    if spec.experiment.schedule is ScheduleMode.PARALLEL_SIMULTANEOUS:
        _validate_parallel_updates(transitions)

    metric_order = _metric_order(spec.metrics)
    metric_reducers = tuple(_compile_metric(name, spec.metrics[name]) for name in metric_order)
    visibility = {
        name: CompiledVisibility(
            graph=profile.graph,
            outcomes=profile.outcomes,
            sanctions=profile.sanctions,
            prompts=profile.prompts,
            history_events=profile.history_events,
        )
        for name, profile in spec.visibility_profiles.items()
    }
    return RuntimePlan(
        pack_id=pack.id,
        pack_path=str(pack.path),
        pack_manifest=pack.manifest,
        spec=spec,
        hook_hash=pack.hook_hash,
        runtime_hash=runtime_hash(spec, pack.hook_hash),
        trusted_evaluation_hash=trusted_evaluation_hash(spec),
        hooks=pack.hooks,
        transitions=tuple(transitions),
        transitions_by_state={state: tuple(items) for state, items in by_state_mutable.items()},
        transitions_by_state_action=by_key,
        visibility=visibility,
        metric_order=metric_order,
        metric_reducers=metric_reducers,
        constraint_templates=tuple(
            CompiledConstraint(
                metric=item.metric,
                operator=item.operator,
                threshold=item.threshold,
            )
            for item in spec.evaluation.constraints
        ),
        policy_factories={
            name: PolicyFactory(archetype.model_copy(deep=True))
            for name, archetype in spec.archetypes.items()
        },
    )


def trusted_evaluation_hash(spec: IncentiveSpec) -> str:
    payload = {
        "metrics": spec.model_dump(mode="json")["metrics"],
        "evaluation": spec.evaluation.model_dump(mode="json"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_hash(spec: IncentiveSpec, hook_hash: str) -> str:
    payload = f"{spec.canonical_json()}:{hook_hash}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _compile_transition(
    transition,
    *,
    symbolic_blocked: set[str],
    symbolic_reasons: dict[str, tuple[str, ...]],
) -> CompiledTransition:
    reasons = [
        f"availability:{transition.availability.value}",
        f"norm:{transition.norm_status.value}",
        *symbolic_reasons.get(transition.id, ()),
    ]
    availability = transition.availability
    if transition.id in symbolic_blocked:
        from icframe.domain.incentive_spec import Availability

        availability = Availability.HARD_BLOCKED
        reasons.append("symbolic:blocked")
    enforcement = None
    if transition.enforcement is not None:
        source = transition.enforcement
        enforcement = CompiledEnforcement(
            audit_probability=source.audit_probability,
            detection_probability=source.detection_probability,
            false_positive_probability=source.false_positive_probability,
            false_negative_probability=source.false_negative_probability,
            enforcement_probability=source.enforcement_probability,
            sanctions=tuple(_compile_effect(item) for item in source.sanctions),
            compliance_rewards=tuple(_compile_effect(item) for item in source.compliance_rewards),
            remediation_actions=tuple(source.remediation_actions),
        )
    return CompiledTransition(
        id=transition.id,
        from_state=transition.from_state,
        action=transition.action,
        to_state=transition.to_state,
        availability=availability,
        norm_status=transition.norm_status,
        requires_target=transition.requires_target,
        target_populations=frozenset(transition.target_populations),
        tags=tuple(transition.tags),
        effects=tuple(_compile_effect(item) for item in transition.effects),
        state_updates=tuple(
            CompiledStateUpdate(
                scope=item.scope,
                population=item.population,
                field=tuple(item.field),
                operation=item.operation,
                value=item.value,
            )
            for item in transition.state_updates
        ),
        enforcement=enforcement,
        prompt_label=transition.prompt.label if transition.prompt else None,
        prompt_description=transition.prompt.description if transition.prompt else None,
        explanation_reasons=tuple(sorted(set(reasons))),
    )


def _compile_effect(item) -> CompiledEffect:
    return CompiledEffect(
        scope=item.scope,
        population=item.population,
        operation=item.operation,
        values=tuple(sorted(item.values.items())),
    )


def _compile_metric(name: str, item: MetricSpec) -> CompiledMetric:
    return CompiledMetric(
        name=name,
        type=item.type,
        channel=item.channel,
        scope=item.scope,
        required_tags=frozenset(item.where_tags_include),
        left=item.left,
        right=item.right,
        numerator=item.numerator,
        denominator=item.denominator,
        terms=tuple(sorted(item.terms.items())),
    )


def _validate_parallel_updates(transitions: list[CompiledTransition]) -> None:
    for transition in transitions:
        for update in transition.state_updates:
            shared = update.scope in {
                EffectScope.GLOBAL,
                EffectScope.POPULATION,
                EffectScope.ALL_AGENTS,
                EffectScope.TARGET,
            }
            if shared and update.operation is not Operation.ADD:
                path = ".".join(update.field)
                raise CompilationError(
                    f"parallel transition {transition.id} has non-commutative shared "
                    f"{update.operation.value} update at {path}; only add is supported"
                )


def _metric_order(metrics: dict[str, MetricSpec]) -> tuple[str, ...]:
    dependencies: dict[str, set[str]] = {}
    names = set(metrics)
    for name, metric in metrics.items():
        refs: set[str] = set()
        if metric.type is MetricType.DIFFERENCE:
            refs.update({metric.left, metric.right})
        elif metric.type is MetricType.RATIO:
            refs.update({metric.numerator, metric.denominator})
        elif metric.type is MetricType.WEIGHTED_SUM:
            refs.update(metric.terms)
        refs.discard(None)
        unknown = refs - names
        if unknown:
            raise CompilationError(f"metric {name} references unknown metrics: {sorted(unknown)}")
        dependencies[name] = refs

    ordered: list[str] = []
    pending = {name: set(refs) for name, refs in dependencies.items()}
    while pending:
        ready = sorted(name for name, refs in pending.items() if not refs)
        if not ready:
            raise CompilationError(f"metric dependency cycle: {sorted(pending)}")
        for name in ready:
            ordered.append(name)
            pending.pop(name)
        for refs in pending.values():
            refs.difference_update(ready)
    return tuple(ordered)
