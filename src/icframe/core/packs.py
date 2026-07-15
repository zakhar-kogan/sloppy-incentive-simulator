from __future__ import annotations

import hashlib
import importlib
import inspect
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import (
    DomainPackManifest,
    GuidedParameter,
    IncentiveSpec,
    ParameterEntity,
    ParameterType,
    load_domain_pack_manifest,
    load_incentive_spec,
)

from .hooks import DomainHooks, NoopHooks


@dataclass(frozen=True, slots=True)
class LoadedDomainPack:
    manifest: DomainPackManifest
    spec: IncentiveSpec
    path: Path
    hooks: DomainHooks
    hook_hash: str

    @property
    def id(self) -> str:
        return self.manifest.pack.id


def builtin_pack_root() -> Path:
    return Path(__file__).resolve().parent.parent / "domain_packs"


def list_domain_packs() -> list[DomainPackManifest]:
    return [
        load_domain_pack_manifest(path) for path in sorted(builtin_pack_root().glob("*/pack.toml"))
    ]


def load_domain_pack(identifier: str | Path) -> LoadedDomainPack:
    requested = Path(identifier)
    if requested.suffix.lower() == ".json":
        raise ValueError(
            "legacy JSON scenarios are unsupported; migrate the scenario into an "
            "IncentiveSpec v0.4 domain pack"
        )
    if requested.exists():
        manifest_path = requested / "pack.toml" if requested.is_dir() else requested
    else:
        manifest_path = builtin_pack_root() / str(identifier) / "pack.toml"
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"domain pack not found: {identifier}")

    manifest = load_domain_pack_manifest(manifest_path)
    spec = load_incentive_spec(manifest_path.parent / manifest.pack.spec_file)
    unknown_objectives = {
        manifest.study.single_objective,
        *manifest.study.pareto_objectives,
    } - set(spec.evaluation.objectives)
    if unknown_objectives:
        raise ValueError(
            f"domain pack {manifest.pack.id} references unknown objectives: "
            f"{sorted(unknown_objectives)}"
        )
    unknown_report_metrics = set(manifest.validation.report_metrics) - set(spec.metrics)
    if unknown_report_metrics:
        raise ValueError(
            f"domain pack {manifest.pack.id} references unknown report metrics: "
            f"{sorted(unknown_report_metrics)}"
        )
    for preset in manifest.study.presets:
        unknown_archetypes = set(preset.exclude_archetypes) - set(spec.archetypes)
        if unknown_archetypes:
            raise ValueError(
                f"study preset {preset.id} excludes unknown archetypes: "
                f"{sorted(unknown_archetypes)}"
            )
    if not set(manifest.validation.golden_seeds).issubset(spec.experiment.seeds):
        raise ValueError(
            f"domain pack {manifest.pack.id} golden seeds must be declared in experiment.seeds"
        )
    _validate_pack_presentation(manifest, spec)
    hooks = _load_hooks(manifest.pack.hook)
    return LoadedDomainPack(
        manifest=manifest,
        spec=spec,
        path=manifest_path.parent,
        hooks=hooks,
        hook_hash=_hook_hash(hooks),
    )


def _validate_pack_presentation(
    manifest: DomainPackManifest, spec: IncentiveSpec
) -> None:
    for template in manifest.population_templates:
        if template.archetype not in spec.archetypes:
            raise ValueError(
                f"population template {template.id} references unknown archetype "
                f"{template.archetype!r}"
            )
    flow = manifest.report.mechanics_flow
    if flow is None:
        return
    transitions = {item.id: item for item in spec.transitions}
    references = {
        "action": set(spec.actions.all),
        "transition": set(transitions),
        "outcome": set(spec.outcome_space.channels),
        "metric": set(spec.metrics),
        "hook": set(spec.hook_config),
        "enforcement": {
            transition_id
            for transition_id, transition in transitions.items()
            if transition.enforcement is not None
        },
    }
    for owner, evidence in [
        *[(f"node {item.id}", item.evidence) for item in flow.nodes],
        *[
            (f"edge {item.source}->{item.target}", item.evidence)
            for item in flow.edges
        ],
    ]:
        for value in evidence:
            kind, marker, reference = value.partition(":")
            if not marker or kind not in references or reference not in references[kind]:
                raise ValueError(
                    f"mechanics flow {owner} has unknown evidence reference {value!r}"
                )


def apply_parameters(
    pack: LoadedDomainPack,
    overrides: dict[str, Scalar] | None = None,
) -> LoadedDomainPack:
    parameters = {parameter.id: parameter for parameter in pack.manifest.parameters}
    unknown = set(overrides or {}) - set(parameters)
    if unknown:
        raise ValueError(f"unknown domain parameters: {sorted(unknown)}")
    values = {
        parameter.id: (overrides or {}).get(parameter.id, parameter.default)
        for parameter in pack.manifest.parameters
    }
    payload = pack.spec.model_dump(mode="python", by_alias=True)
    for parameter_id, value in values.items():
        parameter = parameters[parameter_id]
        _validate_parameter_value(parameter, value)
        target = _parameter_target(payload, parameter)
        _set_path(target, parameter.target.field, value)
    effective = IncentiveSpec.model_validate(payload)
    return replace(pack, spec=effective)


def _parameter_target(payload: dict[str, Any], parameter: GuidedParameter) -> dict[str, Any]:
    target = parameter.target
    if target.entity is ParameterEntity.EXPERIMENT:
        return payload["experiment"]
    if target.entity is ParameterEntity.HOOK_CONFIG:
        return payload["hook_config"]
    if target.entity is ParameterEntity.ARCHETYPE:
        return payload["archetypes"][target.entity_id]
    if target.entity is ParameterEntity.TRANSITION:
        return _find_by_id(payload["transitions"], "id", target.entity_id)
    if target.entity is ParameterEntity.POPULATION:
        return _find_by_id(payload["population"], "archetype", target.entity_id)
    raise AssertionError(f"unsupported parameter entity {target.entity}")


def _find_by_id(rows: list[dict[str, Any]], field: str, value: str | None) -> dict[str, Any]:
    for row in rows:
        if row.get(field) == value:
            return row
    raise ValueError(f"parameter target {field}={value!r} does not exist")


def _set_path(target: dict[str, Any], path: list[str], value: Scalar) -> None:
    cursor = target
    for segment in path[:-1]:
        child = cursor.get(segment)
        if not isinstance(child, dict):
            raise ValueError(f"parameter path {'/'.join(path)} does not resolve to an object")
        cursor = child
    leaf = path[-1]
    if leaf not in cursor:
        raise ValueError(f"parameter path {'/'.join(path)} does not exist")
    cursor[leaf] = value


def _validate_parameter_value(parameter: GuidedParameter, value: Scalar) -> None:
    if parameter.type is ParameterType.FLOAT:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"parameter {parameter.id} requires a number")
        numeric = float(value)
        if numeric < float(parameter.minimum) or numeric > float(parameter.maximum):
            raise ValueError(f"parameter {parameter.id} is outside its bounds")
        if parameter.step is not None and not _is_step_aligned(
            numeric, float(parameter.minimum), float(parameter.step)
        ):
            raise ValueError(f"parameter {parameter.id} does not align with its step")
    elif parameter.type is ParameterType.INTEGER:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"parameter {parameter.id} requires an integer")
        if value < int(parameter.minimum) or value > int(parameter.maximum):
            raise ValueError(f"parameter {parameter.id} is outside its bounds")
        if parameter.step is not None and (value - int(parameter.minimum)) % int(parameter.step):
            raise ValueError(f"parameter {parameter.id} does not align with its step")
    elif parameter.type is ParameterType.BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError(f"parameter {parameter.id} requires a boolean")
    elif parameter.type is ParameterType.CHOICE and value not in parameter.choices:
        raise ValueError(f"parameter {parameter.id} must be one of {parameter.choices}")


def _is_step_aligned(value: float, minimum: float, step: float) -> bool:
    offset = (value - minimum) / step
    return math.isclose(offset, round(offset), rel_tol=1e-9, abs_tol=1e-9)


def _load_hooks(reference: str | None) -> DomainHooks:
    if reference is None:
        return NoopHooks()
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("domain pack hook must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    hooks = factory()
    for method in ("initialize", "before_step", "after_commit", "is_terminal"):
        if not callable(getattr(hooks, method, None)):
            raise TypeError(f"domain hook is missing {method}()")
    return hooks


def _hook_hash(hooks: DomainHooks) -> str:
    try:
        source = inspect.getsource(type(hooks))
    except (OSError, TypeError):
        source = f"{type(hooks).__module__}:{type(hooks).__qualname__}"
    return hashlib.sha256(source.encode()).hexdigest()
