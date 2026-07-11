from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from icframe.catalog import Catalog
from icframe.core.compiler import compile_runtime, runtime_hash, trusted_evaluation_hash
from icframe.core.engine import RuntimeEngine, run_experiment
from icframe.core.observer import NoopObserver
from icframe.core.packs import LoadedDomainPack, apply_parameters, load_domain_pack
from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import (
    ConstraintOperator,
    GuidedParameter,
    ObjectiveDirection,
    ParameterType,
    RetentionProfile,
    SeedReducer,
)
from icframe.domain.run import (
    RunConfig,
    RunStatus,
    SeedResult,
    StudyConfig,
    StudyMode,
    StudySummary,
    TrialRecord,
)
from icframe.llm import LiteLLMClient, LLMClient, LLMRequest, LLMResponse


def run_study(
    pack: LoadedDomainPack | str | Path,
    config: StudyConfig,
    *,
    llm_client: LLMClient | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> StudySummary:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError("install icframe[optimize] to run optimization studies") from exc

    loaded = pack if isinstance(pack, LoadedDomainPack) else load_domain_pack(pack)
    cancelled = cancel_check or (lambda: False)
    _validate_study_config(loaded, config)
    if isinstance(llm_client, LiteLLMClient) and not config.live_llm.enabled:
        raise ValueError("live LLM studies require explicit call and cost budgets")
    if config.live_llm.enabled and llm_client is None:
        raise ValueError("live LLM studies require an LLM client")
    budgeted_client = (
        _BudgetedLLMClient(
            llm_client,
            max_calls=int(config.live_llm.max_calls),
            max_cost_usd=float(config.live_llm.max_cost_usd),
        )
        if config.live_llm.enabled and llm_client is not None
        else None
    )
    effective_llm_client = budgeted_client or llm_client
    study_id = config.study_id or f"study_{uuid.uuid4().hex[:12]}"
    study_dir = config.artifact_root / "studies" / study_id
    study_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _write_json(
        study_dir / "spec.json",
        loaded.spec.model_dump(mode="json", by_alias=True),
    )
    _write_json(
        study_dir / "pack.json",
        loaded.manifest.model_dump(mode="json"),
    )
    _write_json(
        study_dir / "manifest.json",
        {
            "study_id": study_id,
            "pack_id": loaded.id,
            "mode": config.mode.value,
            "objectives": config.objectives,
            "parameters": config.parameters,
            "parameter_ranges": {
                name: bounds.model_dump(mode="json")
                for name, bounds in config.parameter_ranges.items()
            },
            "seeds": config.seeds,
            "status": "running",
            "started_at": _now(),
            "hook_hash": loaded.hook_hash,
            "runtime_hash": runtime_hash(loaded.spec, loaded.hook_hash),
            "trusted_evaluation_hash": trusted_evaluation_hash(loaded.spec),
        },
    )

    retained_run_ids = []
    llm_calls_used = 0
    llm_cost_used = 0.0
    for seed in config.seeds:
        if cancelled() or _budget_exhausted(config, llm_calls_used, llm_cost_used):
            break
        baseline = run_experiment(
            loaded,
            RunConfig(
                seed=seed,
                retention=RetentionProfile.EXPERIMENT,
                artifact_root=config.artifact_root,
            ),
            llm_client=effective_llm_client,
        )
        retained_run_ids.append(baseline.run_id)
        llm_calls_used, llm_cost_used = _budget_usage(
            budgeted_client,
            llm_calls_used + baseline.llm_calls,
            llm_cost_used + baseline.estimated_llm_cost_usd,
        )

    directions = [
        loaded.spec.evaluation.objectives[name].direction.value for name in config.objectives
    ]
    sampler = optuna.samplers.TPESampler(
        seed=config.seeds[0],
        constraints_func=lambda frozen: frozen.user_attrs.get("constraint_values", ()),
    )
    optuna_study = optuna.create_study(
        directions=directions,
        sampler=sampler,
        study_name=study_id,
    )
    parameter_models = {
        item.id: item for item in loaded.manifest.parameters if item.id in config.parameters
    }
    records: list[TrialRecord] = []
    pack_reference = str(loaded.path)
    workers = 1 if effective_llm_client is not None else config.workers

    if workers == 1:
        for _ in range(config.trials):
            if cancelled() or _budget_exhausted(config, llm_calls_used, llm_cost_used):
                break
            trial = optuna_study.ask()
            parameters = _suggest_parameters(trial, parameter_models, config.parameter_ranges)
            try:
                record = _evaluate_trial(
                    loaded,
                    trial.number,
                    parameters,
                    config.seeds,
                    config.objectives,
                    llm_client=effective_llm_client,
                )
            except Exception as exc:
                optuna_study.tell(trial, state=optuna.trial.TrialState.FAIL)
                record = TrialRecord(
                    number=trial.number,
                    parameters=parameters,
                    seeds=[],
                    objective_values={},
                    feasible=False,
                    state="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                _tell_trial(optuna_study, trial, record, config.objectives)
            records.append(record)
            _append_trial(study_dir, record)
            llm_calls_used, llm_cost_used = _budget_usage(
                budgeted_client,
                llm_calls_used + record.llm_calls,
                llm_cost_used + record.estimated_llm_cost_usd,
            )
    else:
        _run_process_trials(
            optuna_study=optuna_study,
            pack_reference=pack_reference,
            config=config,
            parameter_models=parameter_models,
            records=records,
            study_dir=study_dir,
            cancel_check=cancelled,
        )

    feasible = [record for record in records if record.feasible and record.state == "complete"]
    best_trial = _best_trial(loaded, config, feasible)
    pareto_trials = _pareto_trials(loaded, config, feasible)
    pareto_preview = pareto_trials[:200]
    if not cancelled() and config.mode is StudyMode.SINGLE and best_trial is not None:
        winner = next(item for item in feasible if item.number == best_trial)
        for seed in config.seeds:
            if _budget_exhausted(config, llm_calls_used, llm_cost_used):
                break
            retained = run_experiment(
                loaded,
                RunConfig(
                    seed=seed,
                    parameters=winner.parameters,
                    retention=RetentionProfile.EXPERIMENT,
                    artifact_root=config.artifact_root,
                ),
                llm_client=effective_llm_client,
            )
            retained_run_ids.append(retained.run_id)
            llm_calls_used, llm_cost_used = _budget_usage(
                budgeted_client,
                llm_calls_used + retained.llm_calls,
                llm_cost_used + retained.estimated_llm_cost_usd,
            )

    preview_numbers = {record.number for record in records[:200]}
    if best_trial is not None:
        preview_numbers.add(best_trial)
    preview_numbers.update(pareto_trials[:50])
    preview = [record for record in records if record.number in preview_numbers]
    summary = StudySummary(
        study_id=study_id,
        pack_id=loaded.id,
        mode=config.mode,
        status=RunStatus.CANCELLED if cancelled() else RunStatus.COMPLETED,
        objectives=config.objectives,
        parameters=config.parameters,
        seeds=config.seeds,
        trial_count=len(records),
        trials=preview,
        best_trial=best_trial,
        pareto_trials=pareto_preview,
        retained_run_ids=retained_run_ids,
        duration_seconds=time.perf_counter() - started,
        artifacts={
            "manifest": str(study_dir / "manifest.json"),
            "summary": str(study_dir / "summary.json"),
            "trials": str(study_dir / "trials.jsonl"),
            "spec": str(study_dir / "spec.json"),
            "pack": str(study_dir / "pack.json"),
        },
    )
    _write_json(study_dir / "summary.json", summary.model_dump(mode="json"))
    manifest = json.loads((study_dir / "manifest.json").read_text())
    manifest.update(
        {
            "status": summary.status.value,
            "completed_at": _now(),
            "trial_count": len(records),
            "best_trial": best_trial,
            "pareto_trials": pareto_trials,
            "retained_run_ids": retained_run_ids,
            "artifacts": summary.artifacts,
        }
    )
    _write_json(study_dir / "manifest.json", manifest)
    catalog = Catalog(config.artifact_root)
    catalog.upsert_study(summary)
    catalog.replace_trials(study_id, records)
    return summary


def _run_process_trials(
    *,
    optuna_study,
    pack_reference: str,
    config: StudyConfig,
    parameter_models: dict[str, GuidedParameter],
    records: list[TrialRecord],
    study_dir: Path,
    cancel_check: Callable[[], bool],
) -> None:
    remaining = config.trials
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        while remaining and not cancel_check():
            batch: dict[Future, Any] = {}
            for _ in range(min(config.workers, remaining)):
                trial = optuna_study.ask()
                parameters = _suggest_parameters(
                    trial,
                    parameter_models,
                    config.parameter_ranges,
                )
                future = executor.submit(
                    _evaluate_trial,
                    pack_reference,
                    trial.number,
                    parameters,
                    config.seeds,
                    config.objectives,
                )
                batch[future] = trial
                remaining -= 1
            for completed, trial in sorted(batch.items(), key=lambda item: item[1].number):
                try:
                    record = completed.result()
                except Exception as exc:
                    import optuna

                    optuna_study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    record = TrialRecord(
                        number=trial.number,
                        parameters=dict(trial.params),
                        seeds=[],
                        objective_values={},
                        feasible=False,
                        state="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    _tell_trial(optuna_study, trial, record, config.objectives)
                records.append(record)
                _append_trial(study_dir, record)


def _evaluate_trial(
    pack_source: LoadedDomainPack | str,
    number: int,
    parameters: dict[str, Scalar],
    seeds: list[int],
    objectives: list[str],
    llm_client: LLMClient | None = None,
) -> TrialRecord:
    loaded = (
        pack_source if isinstance(pack_source, LoadedDomainPack) else load_domain_pack(pack_source)
    )
    pack = apply_parameters(loaded, parameters)
    plan = compile_runtime(pack)
    seed_results = []
    for seed in seeds:
        engine = RuntimeEngine(
            plan,
            run_id=f"trial_{number:06d}_seed_{seed}",
            seed=seed,
            llm_client=llm_client,
            observer=NoopObserver(),
            retention=RetentionProfile.TRAINING,
        )
        summary = engine.run()
        seed_results.append(
            SeedResult(
                seed=seed,
                metrics=summary.metrics,
                objectives=summary.objectives,
                feasible=summary.feasible,
                constraints=summary.constraints,
                llm_calls=summary.llm_calls,
                estimated_llm_cost_usd=summary.estimated_llm_cost_usd,
            )
        )
    objective_values = {
        name: _reduce_objective(
            [seed.objectives[name] for seed in seed_results],
            pack.spec.evaluation.objectives[name],
        )
        for name in objectives
    }
    return TrialRecord(
        number=number,
        parameters=parameters,
        seeds=seed_results,
        objective_values=objective_values,
        feasible=_constraints_pass(pack, seed_results),
        llm_calls=sum(seed.llm_calls for seed in seed_results),
        estimated_llm_cost_usd=sum(seed.estimated_llm_cost_usd for seed in seed_results),
        runtime_hash=plan.runtime_hash,
        hook_hash=plan.hook_hash,
    )


def _suggest_parameters(
    trial,
    parameters: dict[str, GuidedParameter],
    ranges,
) -> dict[str, Scalar]:
    values = {}
    for name, parameter in parameters.items():
        if parameter.type is ParameterType.FLOAT:
            bounds = ranges.get(name)
            values[name] = trial.suggest_float(
                name,
                float(bounds.minimum if bounds else parameter.minimum),
                float(bounds.maximum if bounds else parameter.maximum),
            )
        elif parameter.type is ParameterType.INTEGER:
            bounds = ranges.get(name)
            values[name] = trial.suggest_int(
                name,
                int(bounds.minimum if bounds else parameter.minimum),
                int(bounds.maximum if bounds else parameter.maximum),
            )
        elif parameter.type is ParameterType.BOOLEAN:
            values[name] = trial.suggest_categorical(name, [False, True])
        else:
            values[name] = trial.suggest_categorical(name, parameter.choices)
    return values


def _tell_trial(study, trial, record: TrialRecord, objectives: list[str]) -> None:
    trial.set_user_attr("constraint_values", [] if record.feasible else [1.0])
    values = [record.objective_values[name] for name in objectives]
    study.tell(trial, values[0] if len(values) == 1 else values)


def _reduce_objective(values: list[float], objective) -> float:
    if objective.seed_reducer is SeedReducer.MEAN:
        return statistics.fmean(values)
    if objective.seed_reducer is SeedReducer.MEDIAN:
        return statistics.median(values)
    if objective.seed_reducer is SeedReducer.WORST:
        return min(values) if objective.direction is ObjectiveDirection.MAXIMIZE else max(values)
    return _quantile(values, float(objective.quantile))


def _constraints_pass(pack: LoadedDomainPack, seeds: list[SeedResult]) -> bool:
    for constraint in pack.spec.evaluation.constraints:
        if constraint.require_all_seeds:
            if not all(
                next(
                    result.passed
                    for result in seed.constraints
                    if result.metric == constraint.metric
                )
                for seed in seeds
            ):
                return False
            continue
        value = _reduce_constraint(
            [seed.metrics[constraint.metric] for seed in seeds],
            constraint,
        )
        passed = (
            value <= constraint.threshold
            if constraint.operator is ConstraintOperator.LE
            else value >= constraint.threshold
        )
        if not passed:
            return False
    return True


def _reduce_constraint(values: list[float], constraint) -> float:
    if constraint.seed_reducer is SeedReducer.MEAN:
        return statistics.fmean(values)
    if constraint.seed_reducer is SeedReducer.MEDIAN:
        return statistics.median(values)
    if constraint.seed_reducer is SeedReducer.WORST:
        return max(values) if constraint.operator is ConstraintOperator.LE else min(values)
    return _quantile(values, float(constraint.quantile))


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _best_trial(
    pack: LoadedDomainPack,
    config: StudyConfig,
    records: list[TrialRecord],
) -> int | None:
    if config.mode is not StudyMode.SINGLE or not records:
        return None
    name = config.objectives[0]
    direction = pack.spec.evaluation.objectives[name].direction

    def objective_value(record: TrialRecord) -> float:
        return record.objective_values[name]

    selected = (
        max(records, key=objective_value)
        if direction is ObjectiveDirection.MAXIMIZE
        else min(records, key=objective_value)
    )
    return selected.number


def _pareto_trials(
    pack: LoadedDomainPack,
    config: StudyConfig,
    records: list[TrialRecord],
) -> list[int]:
    if config.mode is not StudyMode.PARETO:
        return []
    frontier = []
    for candidate in records:
        dominated = any(
            _dominates(pack, config.objectives, other, candidate)
            for other in records
            if other.number != candidate.number
        )
        if not dominated:
            frontier.append(candidate.number)
    return sorted(frontier)


def _dominates(
    pack: LoadedDomainPack,
    objectives: list[str],
    left: TrialRecord,
    right: TrialRecord,
) -> bool:
    at_least_as_good = True
    strictly_better = False
    for name in objectives:
        direction = pack.spec.evaluation.objectives[name].direction
        left_value = left.objective_values[name]
        right_value = right.objective_values[name]
        if direction is ObjectiveDirection.MAXIMIZE:
            at_least_as_good &= left_value >= right_value
            strictly_better |= left_value > right_value
        else:
            at_least_as_good &= left_value <= right_value
            strictly_better |= left_value < right_value
    return at_least_as_good and strictly_better


def _validate_study_config(pack: LoadedDomainPack, config: StudyConfig) -> None:
    objectives = set(pack.spec.evaluation.objectives)
    unknown_objectives = set(config.objectives) - objectives
    if unknown_objectives:
        raise ValueError(f"unknown study objectives: {sorted(unknown_objectives)}")
    parameters = {item.id: item for item in pack.manifest.parameters}
    unknown_parameters = set(config.parameters) - set(parameters)
    if unknown_parameters:
        raise ValueError(f"unknown study parameters: {sorted(unknown_parameters)}")
    not_optimizable = [name for name in config.parameters if not parameters[name].optimizable]
    if not_optimizable:
        raise ValueError(f"parameters are not optimizable: {not_optimizable}")
    unknown_ranges = set(config.parameter_ranges) - set(config.parameters)
    if unknown_ranges:
        raise ValueError(
            f"parameter ranges are not selected for optimization: {sorted(unknown_ranges)}"
        )
    for name, bounds in config.parameter_ranges.items():
        parameter = parameters[name]
        if parameter.type not in {ParameterType.FLOAT, ParameterType.INTEGER}:
            raise ValueError(f"parameter {name} does not accept numeric search bounds")
        if bounds.minimum < parameter.minimum or bounds.maximum > parameter.maximum:
            raise ValueError(f"parameter {name} search bounds exceed the domain-pack bounds")
    has_llm = any(
        archetype.policy.value == "llm_policy" for archetype in pack.spec.archetypes.values()
    )
    if config.live_llm.enabled and not has_llm:
        raise ValueError("live LLM budget was enabled for a pack without LLM policies")


def _budget_exhausted(config: StudyConfig, calls: int, cost: float) -> bool:
    budget = config.live_llm
    if not budget.enabled:
        return False
    return calls >= int(budget.max_calls) or cost >= float(budget.max_cost_usd)


class _BudgetedLLMClient:
    def __init__(self, client: LLMClient, *, max_calls: int, max_cost_usd: float) -> None:
        self.client = client
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.calls = 0
        self.cost = 0.0

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.calls >= self.max_calls or self.cost >= self.max_cost_usd:
            raise RuntimeError("LLM study budget exhausted")
        response = self.client.complete(request)
        self.calls += 1
        self.cost += response.estimated_cost
        if self.cost > self.max_cost_usd:
            raise RuntimeError("LLM study cost budget exhausted")
        return response


def _budget_usage(
    client: _BudgetedLLMClient | None,
    fallback_calls: int,
    fallback_cost: float,
) -> tuple[int, float]:
    if client is None:
        return fallback_calls, fallback_cost
    return client.calls, client.cost


def _append_trial(study_dir: Path, record: TrialRecord) -> None:
    with (study_dir / "trials.jsonl").open("a") as file:
        file.write(record.model_dump_json())
        file.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
