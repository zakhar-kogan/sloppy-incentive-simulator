from __future__ import annotations

from icframe.core import load_domain_pack
from icframe.domain.run import PlannerKind, StudyConfig, StudyMode
from icframe.planning import create_study_plan


def test_matrix_plan_is_byte_deterministic_and_cartesian(tmp_path) -> None:
    pack = load_domain_pack("software_organization")
    config = StudyConfig(
        study_id="deterministic-matrix",
        mode=StudyMode.SINGLE,
        objectives=["trusted_score"],
        parameters=["proxy_agents", "audit_probability"],
        seeds=[0, 7],
        artifact_root=tmp_path,
        planner=PlannerKind.MATRIX,
        planner_seed=17,
        parameter_matrix={
            "proxy_agents": [1, 4],
            "audit_probability": [0.0, 0.6],
        },
    )

    first = create_study_plan(pack, config)
    second = create_study_plan(pack, config)

    assert first.model_dump_json() == second.model_dump_json()
    assert first.canonical_hash == second.canonical_hash
    renamed = create_study_plan(
        pack,
        config.model_copy(update={"study_id": "same-plan-different-job"}),
    )
    assert first.canonical_hash == renamed.canonical_hash
    assert [trial.number for trial in first.trials] == [0, 1, 2, 3]
    assert [trial.parameters for trial in first.trials] == [
        {"proxy_agents": 1, "audit_probability": 0.0},
        {"proxy_agents": 1, "audit_probability": 0.6},
        {"proxy_agents": 4, "audit_probability": 0.0},
        {"proxy_agents": 4, "audit_probability": 0.6},
    ]


def test_seeded_random_plan_is_unique_and_repeatable(tmp_path) -> None:
    pack = load_domain_pack("software_organization")
    base = StudyConfig(
        study_id="deterministic-random",
        mode=StudyMode.SINGLE,
        objectives=["trusted_score"],
        parameters=["proxy_agents", "audit_probability"],
        seeds=[0],
        trials=12,
        artifact_root=tmp_path,
        planner=PlannerKind.RANDOM,
        planner_seed=91,
    )

    first = create_study_plan(pack, base)
    second = create_study_plan(pack, base)
    changed = create_study_plan(pack, base.model_copy(update={"planner_seed": 92}))

    assert first.model_dump_json() == second.model_dump_json()
    assert first.model_dump_json() != changed.model_dump_json()
    assignments = [tuple(sorted(trial.parameters.items())) for trial in first.trials]
    assert len(assignments) == len(set(assignments)) == 12
