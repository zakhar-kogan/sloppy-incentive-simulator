from __future__ import annotations

import pytest
from pydantic import ValidationError

from icframe.domain.norms import LawProgram
from icframe.domain.scenario import AgentConfig, AgentPolicy, Scenario


def test_scenario_models_are_strict_about_nested_inputs() -> None:
    with pytest.raises(ValidationError):
        Scenario(
            name="strictness",
            description="nested dictionaries should not be coerced",
            agents=[{"name": "alice", "policy": "cooperative", "endowment": 10.0}],
            laws=LawProgram(),
        )


def test_scenario_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(name="alice", policy=AgentPolicy.COOPERATIVE, endowment=10.0, extra_field=True)
