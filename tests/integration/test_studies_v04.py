from __future__ import annotations

import math

import pytest

from icframe import StudyConfig, run_study
from icframe.catalog import Catalog
from icframe.domain.incentive_spec import ObjectiveDirection, ObjectiveSpec, SeedReducer
from icframe.domain.run import LiveLLMBudget, ParameterRange, StudyMode
from icframe.study import _reduce_objective, _suggest_parameters


def test_single_and_pareto_studies_share_one_trial_contract(tmp_path) -> None:
    single = run_study(
        "delayed_reward_learning",
        StudyConfig(
            mode=StudyMode.SINGLE,
            objectives=["trusted_score"],
            parameters=["epsilon"],
            trials=3,
            seeds=[11, 17],
            workers=1,
            artifact_root=tmp_path,
        ),
    )
    assert single.best_trial is not None
    assert len(single.retained_run_ids) == 4
    assert Catalog(tmp_path).get_trial(single.study_id, 0) is not None

    pareto = run_study(
        "delayed_reward_learning",
        StudyConfig(
            mode=StudyMode.PARETO,
            objectives=["total_payoff", "harvest_rate"],
            parameters=["epsilon"],
            trials=4,
            seeds=[11],
            workers=1,
            artifact_root=tmp_path,
        ),
    )
    assert pareto.pareto_trials
    assert len(pareto.retained_run_ids) == 1


def test_process_pool_trials_are_deterministic(tmp_path) -> None:
    results = []
    for name in ("one", "two"):
        summary = run_study(
            "delayed_reward_learning",
            StudyConfig(
                study_id=name,
                mode=StudyMode.SINGLE,
                objectives=["trusted_score"],
                parameters=["epsilon"],
                trials=4,
                seeds=[11],
                workers=2,
                artifact_root=tmp_path,
            ),
        )
        results.append(
            [(trial.number, trial.parameters, trial.objective_values) for trial in summary.trials]
        )
    assert results[0] == results[1]


def test_study_uses_requested_parameter_range(tmp_path) -> None:
    summary = run_study(
        "delayed_reward_learning",
        StudyConfig(
            mode=StudyMode.SINGLE,
            objectives=["trusted_score"],
            parameters=["epsilon"],
            parameter_ranges={"epsilon": ParameterRange(minimum=0.2, maximum=0.25)},
            trials=3,
            seeds=[11],
            workers=1,
            artifact_root=tmp_path,
        ),
    )
    assert all(0.2 <= trial.parameters["epsilon"] <= 0.25 for trial in summary.trials)
    assert all(
        math.isclose(
            trial.parameters["epsilon"] * 100,
            round(trial.parameters["epsilon"] * 100),
        )
        for trial in summary.trials
    )


def test_study_suggestions_forward_declared_numeric_steps() -> None:
    from icframe import load_domain_pack

    pack = load_domain_pack("delayed_reward_learning")
    parameters = {item.id: item for item in pack.manifest.parameters}

    class RecordingTrial:
        def __init__(self) -> None:
            self.calls = []

        def suggest_float(self, name, minimum, maximum, *, step):
            self.calls.append((name, minimum, maximum, step))
            return minimum

        def suggest_int(self, name, minimum, maximum, *, step):
            self.calls.append((name, minimum, maximum, step))
            return minimum

    trial = RecordingTrial()
    _suggest_parameters(
        trial,
        {"epsilon": parameters["epsilon"], "steps": parameters["steps"]},
        {},
    )

    assert trial.calls == [
        ("epsilon", 0.0, 1.0, 0.01),
        ("steps", 10, 100000, 1),
    ]


def test_study_parameter_ranges_must_align_with_declared_step(tmp_path) -> None:
    with pytest.raises(ValueError, match="search bounds do not align with its step"):
        run_study(
            "delayed_reward_learning",
            StudyConfig(
                mode=StudyMode.SINGLE,
                objectives=["trusted_score"],
                parameters=["epsilon"],
                parameter_ranges={"epsilon": ParameterRange(minimum=0.205, maximum=0.25)},
                trials=1,
                seeds=[11],
                workers=1,
                artifact_root=tmp_path,
            ),
        )


def test_seed_reducers_include_direction_aware_worst_and_quantile() -> None:
    assert (
        _reduce_objective(
            [1.0, 7.0, 3.0],
            ObjectiveSpec(
                metric="x",
                direction=ObjectiveDirection.MAXIMIZE,
                seed_reducer=SeedReducer.WORST,
            ),
        )
        == 1.0
    )
    assert (
        _reduce_objective(
            [1.0, 7.0, 3.0],
            ObjectiveSpec(
                metric="x",
                direction=ObjectiveDirection.MINIMIZE,
                seed_reducer=SeedReducer.WORST,
            ),
        )
        == 7.0
    )
    assert (
        _reduce_objective(
            [0.0, 10.0],
            ObjectiveSpec(
                metric="x",
                seed_reducer=SeedReducer.QUANTILE,
                quantile=0.25,
            ),
        )
        == 2.5
    )


def test_live_llm_budgets_require_explicit_limits() -> None:
    try:
        LiveLLMBudget(enabled=True)
    except ValueError as exc:
        assert "max_calls and max_cost_usd" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unbounded live LLM study was accepted")


def test_llm_study_call_budget_is_enforced(tmp_path, deterministic_llm_client) -> None:
    client = deterministic_llm_client("refactor_core")
    summary = run_study(
        "software_organization",
        StudyConfig(
            mode=StudyMode.SINGLE,
            objectives=["trusted_score"],
            parameters=["audit_probability"],
            trials=2,
            seeds=[19],
            workers=1,
            artifact_root=tmp_path,
            live_llm=LiveLLMBudget(
                enabled=True,
                max_calls=2,
                max_cost_usd=1.0,
            ),
        ),
        llm_client=client,
    )
    assert len(client.requests) == 2
    assert summary.trial_count == 0


def test_malformed_llm_response_is_a_failed_decision(deterministic_llm_client) -> None:
    from icframe import compile_runtime, load_domain_pack
    from icframe.core.engine import RuntimeEngine
    from icframe.core.observer import NoopObserver

    engine = RuntimeEngine(
        compile_runtime(load_domain_pack("software_organization")),
        run_id="malformed-llm",
        seed=19,
        llm_client=deterministic_llm_client("", parsed={}),
        observer=NoopObserver(),
    )
    result = engine.step_internal()
    decision = next(item for item in result.decisions if item.agent_id == "llm_engineer_000")
    assert decision.failure == "malformed_llm_action"


def test_study_cancellation_produces_a_catalogued_summary(tmp_path) -> None:
    summary = run_study(
        "delayed_reward_learning",
        StudyConfig(
            mode=StudyMode.SINGLE,
            objectives=["trusted_score"],
            parameters=["epsilon"],
            trials=10,
            seeds=[11],
            workers=1,
            artifact_root=tmp_path,
        ),
        cancel_check=lambda: True,
    )
    assert summary.status.value == "cancelled"
    assert summary.trial_count == 0
    assert summary.retained_run_ids == []
