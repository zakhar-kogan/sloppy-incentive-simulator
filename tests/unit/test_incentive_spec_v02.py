from __future__ import annotations

import random

from pydantic import ValidationError

from icframe.analytics.networkx import project_incentive_trace
from icframe.constraints import explain_transition_availability, validate_constraints
from icframe.domain.incentive_spec import Availability, IncentiveSpec, load_incentive_spec
from icframe.pipelines import persist_incentive_run
from icframe.runtime.incentive import (
    _expand_population,
    _RuntimeWorld,
    compile_observation,
    run_incentive_simulation,
)

SPEC_PATH = "incentive_spec_example_tokenmaxxing.toml"


def test_tokenmaxxing_toml_loads_and_validates() -> None:
    spec = load_incentive_spec(SPEC_PATH)

    assert spec.spec.version == "0.2"
    assert spec.spec.name == "tokenmaxxing_case_study"
    assert "observed.kpi_score" in spec.outcome_space.channels
    assert len(spec.transitions) == 4


def test_invalid_spec_rejects_undeclared_effect_channel() -> None:
    spec = load_incentive_spec(SPEC_PATH)
    payload = spec.model_dump(mode="python", by_alias=True)
    payload["transitions"][0]["effects"]["latent.unlisted_channel"] = 1.0

    try:
        IncentiveSpec.model_validate(payload)
    except ValidationError as exc:
        assert "undeclared channels" in str(exc)
    else:
        raise AssertionError("expected undeclared outcome channel to fail validation")


def test_clingo_constraints_report_missing_enforcement_and_hard_blocked() -> None:
    spec = load_incentive_spec(SPEC_PATH)
    payload = spec.model_dump(mode="python", by_alias=True)
    payload["transitions"][2]["enforcement"] = None
    payload["transitions"][0]["availability"] = "hard_blocked"
    invalid = IncentiveSpec.model_validate(payload)

    report = validate_constraints(invalid)

    assert not report.ok
    assert {problem.code for problem in report.problems} == {
        "auditable_forbidden_missing_enforcement",
        "hard_blocked_transition",
    }


def test_constraint_explanation_reports_possible_violation_norm() -> None:
    spec = load_incentive_spec(SPEC_PATH)

    explanation = explain_transition_availability(
        spec,
        actor_id="proxy_maximizer_000",
        state="working",
        action="misreport_activity",
    )

    assert explanation.available
    assert explanation.transition_id == "misreport_activity"
    assert explanation.norm_status == "forbidden"
    assert "possible_violation" in explanation.reasons
    assert "norm_forbidden" in explanation.reasons


def test_seeded_runtime_is_deterministic_and_computes_metrics() -> None:
    spec = load_incentive_spec(SPEC_PATH)

    first = run_incentive_simulation(spec, seed=3)
    second = run_incentive_simulation(spec, seed=3)

    assert first.canonical_json() == second.canonical_json()
    assert first.events
    assert {"goodhart_gap", "exploit_rate", "governance_efficiency"} <= set(first.metric_results)
    assert first.metric_results["exploit_rate"] > 0.0
    assert first.events[0].constraint_explanation is not None


def test_runtime_applies_detected_conditional_sanction_for_regulated_actor() -> None:
    spec = load_incentive_spec(SPEC_PATH)
    payload = spec.model_dump(mode="python", by_alias=True)
    payload["population"] = [{"archetype": "regulated_actor", "count": 1}]
    for transition in payload["transitions"]:
        if transition["id"] == "misreport_activity":
            transition["enforcement"]["audit_probability"] = 1.0
            transition["enforcement"]["detection_probability"] = 1.0
            transition["enforcement"]["enforcement_probability"] = 1.0
    forced = IncentiveSpec.model_validate(payload)

    trace = run_incentive_simulation(forced, seed=1)
    event = trace.events[0]

    assert event.action == "misreport_activity"
    assert event.audit_sampled
    assert event.detected
    assert event.enforced
    assert event.final_outcome_vector["agent.personal_payoff"] == -21.0
    assert "misreport_activity:detected:10" in event.conditional_effects_applied


def test_visibility_compiler_hides_latent_outcomes_from_explorer() -> None:
    spec = load_incentive_spec(SPEC_PATH)
    agents = _expand_population(spec)
    agent = agents["proxy_maximizer_000"]
    world = _RuntimeWorld(
        spec=spec,
        rng=random.Random(0),
        seed=0,
        run_id="visibility-test",
        agents=agents,
    )

    observation = compile_observation(spec, world, agent)

    assert observation.visible_actions
    assert all(
        not channel.startswith("latent.")
        for outcome in observation.visible_outcomes.values()
        for channel in outcome
    )


def test_hard_blocked_actions_are_not_visible_or_chosen() -> None:
    spec = load_incentive_spec(SPEC_PATH)
    payload = spec.model_dump(mode="python", by_alias=True)
    payload["transitions"][2]["availability"] = Availability.HARD_BLOCKED.value
    blocked = IncentiveSpec.model_validate(payload)

    explanation = explain_transition_availability(
        blocked,
        actor_id="proxy_maximizer_000",
        state="working",
        action="misreport_activity",
    )
    trace = run_incentive_simulation(blocked, seed=4)

    assert explanation.hard_blocked
    assert not explanation.available
    assert all(event.action != "misreport_activity" for event in trace.events)


def test_v02_projection_and_persistence_adapters(tmp_path) -> None:
    spec = load_incentive_spec(SPEC_PATH)
    trace = run_incentive_simulation(spec, seed=5)

    graph = project_incentive_trace(trace)
    artifacts = persist_incentive_run(tmp_path, spec, trace)

    assert graph.nodes
    assert graph.edges
    assert set(artifacts) == {"constraints", "spec", "summary", "trace"}
    assert (tmp_path / "summary.json").exists()
