from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import uuid
import warnings
from dataclasses import replace
from typing import Protocol

from pydantic import Field

from icframe.core.compiler import runtime_hash
from icframe.core.packs import LoadedDomainPack
from icframe.domain.base import ICFrameModel, Scalar
from icframe.domain.incentive_spec import GuidedParameter, IncentiveSpec, ParameterType, StudyPreset
from icframe.domain.run import ParameterRange, PlannerKind, StudyConfig, StudyMode


class TrialSpec(ICFrameModel):
    number: int = Field(ge=0)
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    seeds: list[int] = Field(min_length=1)
    objectives: list[str] = Field(min_length=1)


class StudyPlan(ICFrameModel):
    schema_version: str = "1"
    study_id: str
    pack_id: str
    pack_path: str
    pack_hash: str
    runtime_hash: str
    mode: StudyMode
    planner: PlannerKind
    planner_seed: int
    trials: list[TrialSpec] = Field(min_length=1)

    @property
    def canonical_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"study_id", "pack_path"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def shard(self, size: int) -> list[list[TrialSpec]]:
        if size < 1:
            raise ValueError("shard size must be positive")
        return [self.trials[offset : offset + size] for offset in range(0, len(self.trials), size)]


class StudyPlanner(Protocol):
    kind: PlannerKind

    def plan(self, pack: LoadedDomainPack, config: StudyConfig) -> StudyPlan:
        """Create a complete, deterministic set of trial specifications."""


class MatrixPlanner:
    kind = PlannerKind.MATRIX

    def plan(self, pack: LoadedDomainPack, config: StudyConfig) -> StudyPlan:
        parameters = _parameter_models(pack, config)
        unknown = set(config.parameter_matrix) - set(parameters)
        missing = set(config.parameters) - set(config.parameter_matrix)
        if unknown:
            raise ValueError(f"matrix references unknown parameters: {sorted(unknown)}")
        if missing:
            raise ValueError(f"matrix values are missing for parameters: {sorted(missing)}")
        ordered_values = []
        for name in config.parameters:
            values = config.parameter_matrix[name]
            if not values:
                raise ValueError(f"matrix parameter {name} has no values")
            validated = [_validated_value(parameters[name], value) for value in values]
            if len({_scalar_key(value) for value in validated}) != len(validated):
                raise ValueError(f"matrix parameter {name} contains duplicate values")
            ordered_values.append(validated)
        assignments = [
            dict(zip(config.parameters, values, strict=True))
            for values in itertools.product(*ordered_values)
        ]
        return _plan(pack, config, self.kind, assignments)


class RandomPlanner:
    kind = PlannerKind.RANDOM

    def plan(self, pack: LoadedDomainPack, config: StudyConfig) -> StudyPlan:
        parameters = _parameter_models(pack, config)
        candidates = {
            name: _candidate_values(parameter, config.parameter_ranges.get(name))
            for name, parameter in parameters.items()
        }
        population = math.prod(len(values) for values in candidates.values())
        if config.trials > population:
            raise ValueError(
                f"requested {config.trials} unique trials from a space of {population}"
            )
        rng = random.Random(config.planner_seed)
        assignments: list[dict[str, Scalar]] = []
        seen: set[str] = set()
        while len(assignments) < config.trials:
            candidate = {
                name: rng.choice(candidates[name])
                for name in config.parameters
            }
            key = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            assignments.append(candidate)
        return _plan(pack, config, self.kind, assignments)


class OptunaPlanner:
    """Compatibility planner for explicit local Optuna use.

    Adaptive ask/tell execution stays in ``run_study`` for v0.5. Calling
    ``plan`` produces the deterministic startup batch only and is therefore
    intentionally not accepted by remote execution.
    """

    kind = PlannerKind.OPTUNA

    def plan(self, pack: LoadedDomainPack, config: StudyConfig) -> StudyPlan:
        warnings.warn(
            "Optuna planning is adaptive and local-only; use run_study for ask/tell execution",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - optional installation
            raise RuntimeError("install icframe[optimize] to use OptunaPlanner") from exc
        parameters = _parameter_models(pack, config)
        study = optuna.create_study(
            directions=[
                pack.spec.evaluation.objectives[name].direction.value
                for name in config.objectives
            ],
            sampler=optuna.samplers.TPESampler(seed=config.planner_seed),
        )
        assignments = []
        for _ in range(config.trials):
            trial = study.ask()
            values = {}
            for name, parameter in parameters.items():
                choices = _candidate_values(parameter, config.parameter_ranges.get(name))
                values[name] = trial.suggest_categorical(name, choices)
            assignments.append(values)
        return _plan(pack, config, self.kind, assignments)


def create_study_plan(pack: LoadedDomainPack, config: StudyConfig) -> StudyPlan:
    planner = config.planner or PlannerKind.OPTUNA
    implementation: StudyPlanner
    if planner is PlannerKind.MATRIX:
        implementation = MatrixPlanner()
    elif planner is PlannerKind.RANDOM:
        implementation = RandomPlanner()
    else:
        implementation = OptunaPlanner()
    return implementation.plan(pack, config)


def apply_study_preset(pack: LoadedDomainPack, preset: StudyPreset) -> LoadedDomainPack:
    """Materialize non-secret runtime changes declared by a named study preset."""
    if not preset.exclude_archetypes:
        return pack
    payload = pack.spec.model_dump(mode="python", by_alias=True)
    payload["population"] = [
        item
        for item in payload["population"]
        if item["archetype"] not in preset.exclude_archetypes
    ]
    if not payload["population"]:
        raise ValueError(f"study preset {preset.id} removes the complete population")
    return replace(pack, spec=IncentiveSpec.model_validate(payload))


def _plan(
    pack: LoadedDomainPack,
    config: StudyConfig,
    planner: PlannerKind,
    assignments: list[dict[str, Scalar]],
) -> StudyPlan:
    if not assignments:
        raise ValueError("study plan must contain at least one trial")
    study_id = config.study_id or f"study_{uuid.uuid4().hex[:12]}"
    return StudyPlan(
        study_id=study_id,
        pack_id=pack.id,
        pack_path=str(pack.path),
        pack_hash=pack_fingerprint(pack),
        runtime_hash=runtime_hash(pack.spec, pack.hook_hash),
        mode=config.mode,
        planner=planner,
        planner_seed=config.planner_seed,
        trials=[
            TrialSpec(
                number=number,
                parameters=parameters,
                seeds=list(config.seeds),
                objectives=list(config.objectives),
            )
            for number, parameters in enumerate(assignments)
        ],
    )


def _parameter_models(
    pack: LoadedDomainPack,
    config: StudyConfig,
) -> dict[str, GuidedParameter]:
    declared = {item.id: item for item in pack.manifest.parameters}
    unknown = set(config.parameters) - set(declared)
    if unknown:
        raise ValueError(f"unknown study parameters: {sorted(unknown)}")
    disabled = [name for name in config.parameters if not declared[name].optimizable]
    if disabled:
        raise ValueError(f"parameters are not optimizable: {disabled}")
    return {name: declared[name] for name in config.parameters}


def _candidate_values(
    parameter: GuidedParameter,
    bounds: ParameterRange | None,
) -> list[Scalar]:
    if parameter.type is ParameterType.BOOLEAN:
        return [False, True]
    if parameter.type is ParameterType.CHOICE:
        return list(parameter.choices)
    minimum = bounds.minimum if bounds is not None else parameter.minimum
    maximum = bounds.maximum if bounds is not None else parameter.maximum
    if minimum is None or maximum is None:
        raise ValueError(f"numeric parameter {parameter.id} is missing bounds")
    step = parameter.step or (1 if parameter.type is ParameterType.INTEGER else 0.01)
    count = math.floor((float(maximum) - float(minimum)) / float(step) + 1e-9) + 1
    values: list[Scalar] = []
    for index in range(count):
        raw = float(minimum) + index * float(step)
        value: Scalar = (
            round(raw) if parameter.type is ParameterType.INTEGER else round(raw, 12)
        )
        values.append(value)
    if values[-1] != maximum and math.isclose(float(values[-1]), float(maximum), abs_tol=1e-9):
        values[-1] = int(maximum) if parameter.type is ParameterType.INTEGER else float(maximum)
    return values


def _validated_value(parameter: GuidedParameter, value: Scalar) -> Scalar:
    if parameter.type is ParameterType.BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError(f"matrix parameter {parameter.id} requires boolean values")
        return value
    if parameter.type is ParameterType.CHOICE:
        if value not in parameter.choices:
            raise ValueError(f"matrix parameter {parameter.id} value {value!r} is not declared")
        return value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"matrix parameter {parameter.id} requires numeric values")
    if parameter.type is ParameterType.INTEGER and not float(value).is_integer():
        raise ValueError(f"matrix parameter {parameter.id} requires integer values")
    if value < parameter.minimum or value > parameter.maximum:
        raise ValueError(f"matrix parameter {parameter.id} value {value} is outside bounds")
    if parameter.step is not None:
        offset = (float(value) - float(parameter.minimum)) / float(parameter.step)
        if not math.isclose(offset, round(offset), abs_tol=1e-8):
            raise ValueError(f"matrix parameter {parameter.id} value {value} is off step")
    return int(value) if parameter.type is ParameterType.INTEGER else float(value)


def pack_fingerprint(pack: LoadedDomainPack) -> str:
    payload = {
        "manifest": pack.manifest.model_dump(mode="json"),
        "spec": pack.spec.model_dump(mode="json", by_alias=True),
        "hook_hash": pack.hook_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scalar_key(value: Scalar) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
