from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from icframe.core import CompilationError, compile_runtime, load_domain_pack
from icframe.core.packs import _validate_pack_presentation
from icframe.domain.incentive_spec import (
    DomainPackManifest,
    IncentiveSpec,
    load_incentive_spec,
)
from icframe.symbolic import SymbolicCompilation


def test_legacy_versions_are_rejected_with_migration_message(tmp_path) -> None:
    path = tmp_path / "legacy.toml"
    path.write_text('[spec]\nversion = "0.3"\nname = "legacy"\n')
    with pytest.raises(ValueError, match=r"only v0\.4.*Migrate"):
        load_incentive_spec(path)
    legacy_json = tmp_path / "legacy.json"
    legacy_json.write_text("{}")
    with pytest.raises(ValueError, match="legacy JSON scenarios are unsupported"):
        load_incentive_spec(legacy_json)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("experiment", "schedule"), "staged"),
        (("visibility_profiles", "numeric", "graph"), "neighbors_only"),
    ],
)
def test_unsupported_capabilities_fail_validation(field, value) -> None:
    payload = load_domain_pack("public_goods").spec.model_dump(mode="python", by_alias=True)
    if field[0] == "visibility_profiles":
        field = (field[0], next(iter(payload["visibility_profiles"])), field[-1])
    cursor = payload
    for segment in field[:-1]:
        cursor = cursor[segment]
    cursor[field[-1]] = value
    with pytest.raises(ValidationError):
        IncentiveSpec.model_validate(payload)


def test_unknown_metric_selectors_are_not_silent_noops() -> None:
    payload = load_domain_pack("public_goods").spec.model_dump(mode="python", by_alias=True)
    payload["metrics"]["social_welfare"]["selector"] = "everyone"
    with pytest.raises(ValidationError, match="selector"):
        IncentiveSpec.model_validate(payload)


def test_duplicate_population_archetypes_are_rejected() -> None:
    payload = load_domain_pack("public_goods").spec.model_dump(mode="python", by_alias=True)
    payload["population"].append(dict(payload["population"][0]))

    with pytest.raises(ValidationError, match="duplicate population archetype"):
        IncentiveSpec.model_validate(payload)


def test_all_reference_packs_have_guidance_and_compile() -> None:
    for pack_id in (
        "public_goods",
        "software_organization",
        "delayed_reward_learning",
    ):
        pack = load_domain_pack(pack_id)
        assert pack.manifest.parameters
        assert pack.manifest.validation.golden_seeds
        assert pack.manifest.population_templates
        assert pack.manifest.report.mechanics_flow
        for template in pack.manifest.population_templates:
            archetype = pack.spec.archetypes[template.archetype]
            assert archetype.scalarizer
            if archetype.llm is not None:
                assert archetype.llm.model
                assert archetype.llm.system_prompt
        assert pack.spec.evaluation.constraints
        assert compile_runtime(pack).transitions


def test_pack_presentation_rejects_unknown_template_and_flow_evidence() -> None:
    pack = load_domain_pack("software_organization")
    manifest_payload = pack.manifest.model_dump(mode="python")
    manifest_payload["population_templates"][0]["archetype"] = "missing"
    manifest = DomainPackManifest.model_validate(manifest_payload)
    with pytest.raises(ValueError, match="unknown archetype"):
        _validate_pack_presentation(manifest, pack.spec)

    manifest_payload = pack.manifest.model_dump(mode="python")
    manifest_payload["report"]["mechanics_flow"]["nodes"][0]["evidence"] = [
        "transition:invented"
    ]
    manifest = DomainPackManifest.model_validate(manifest_payload)
    with pytest.raises(ValueError, match="unknown evidence"):
        _validate_pack_presentation(manifest, pack.spec)


def test_parallel_shared_non_add_updates_are_compile_errors() -> None:
    pack = load_domain_pack("public_goods")
    payload = pack.spec.model_dump(mode="python", by_alias=True)
    payload["transitions"][0]["state_updates"].append(
        {
            "scope": "global",
            "field": ["pool"],
            "operation": "set",
            "value": 1.0,
        }
    )
    with pytest.raises(CompilationError, match="non-commutative shared set"):
        compile_runtime(replace(pack, spec=IncentiveSpec.model_validate(payload)))


def test_agent_state_updates_are_namespaced() -> None:
    pack = load_domain_pack("public_goods")
    payload = pack.spec.model_dump(mode="python", by_alias=True)
    payload["transitions"][0]["state_updates"][0]["field"] = ["balance"]
    with pytest.raises(CompilationError, match="resources or attributes"):
        compile_runtime(replace(pack, spec=IncentiveSpec.model_validate(payload)))


def test_symbolic_adapter_is_invoked_once_at_compile_time(monkeypatch) -> None:
    pack = load_domain_pack("public_goods")
    payload = pack.spec.model_dump(mode="python", by_alias=True)
    payload["symbolic"] = {"enabled": True, "rules": ['blocked("tamper").']}
    calls = []

    def fake_compile(spec):
        calls.append(spec.spec.name)
        return SymbolicCompilation(blocked={"tamper"}, reasons={"tamper": ("symbolic:test",)})

    monkeypatch.setattr("icframe.symbolic.compile_symbolic", fake_compile)
    plan = compile_runtime(replace(pack, spec=IncentiveSpec.model_validate(payload)))
    assert calls == [pack.spec.spec.name]
    assert plan.transitions_by_state_action[("active", "tamper")].explanation_reasons == (
        "availability:possible_violation",
        "norm:forbidden",
        "symbolic:blocked",
        "symbolic:test",
    )
    assert calls == [pack.spec.spec.name]
