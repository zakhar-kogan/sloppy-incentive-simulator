from __future__ import annotations

import pytest
from pydantic import ValidationError

from icframe.core import load_domain_pack
from icframe.ui.request_models import PaginationQuery, RunRequest, StudyRequest


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"limit": 0}, "limit"),
        ({"limit": -1}, "limit"),
        ({"limit": 501}, "limit"),
        ({"offset": -1}, "offset"),
    ],
)
def test_pagination_is_bounded(payload, field) -> None:
    with pytest.raises(ValidationError, match=field):
        PaginationQuery.model_validate(payload)


def test_run_seed_batch_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 100"):
        RunRequest.model_validate({"pack": "public_goods", "seeds": list(range(101))})


def test_study_seed_batch_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 100"):
        StudyRequest.model_validate({"pack": "public_goods", "seeds": list(range(101))})


def test_population_overrides_are_unique_and_ui_runnable() -> None:
    population = {
        "archetype_id": "learner",
        "count": 1,
        "policy": "q_learning_simple",
        "role": "participant",
        "visibility_profile": "numeric",
    }
    with pytest.raises(ValidationError, match="must be unique"):
        RunRequest.model_validate(
            {"pack": "public_goods", "population_overrides": [population, population]}
        )
    with pytest.raises(ValidationError, match="external policies"):
        RunRequest.model_validate(
            {
                "pack": "public_goods",
                "population_overrides": [{**population, "policy": "external"}],
            }
        )


def test_llm_population_requires_per_archetype_configuration() -> None:
    with pytest.raises(ValidationError, match="llm configuration"):
        RunRequest.model_validate(
            {
                "pack": "software_organization",
                "population_overrides": [
                    {
                        "archetype_id": "writer",
                        "count": 1,
                        "policy": "llm_policy",
                        "role": "engineer",
                        "visibility_profile": "llm_public",
                    }
                ],
            }
        )


def test_canonical_llm_population_is_a_valid_ui_override() -> None:
    archetype = load_domain_pack("software_organization").spec.archetypes["llm_engineer"]
    request = RunRequest.model_validate(
        {
            "pack": "software_organization",
            "population_overrides": [
                {
                    "archetype_id": "llm_engineer",
                    "count": 1,
                    **archetype.model_dump(mode="json"),
                }
            ],
        }
    )
    assert request.population_overrides[0].llm.action_field == "action"
