from __future__ import annotations

import json

import pytest

from icframe.llm import LLMRequest, LLMResponse


class DeterministicLLMClient:
    def __init__(
        self,
        action: str,
        *,
        parsed: dict[str, object] | None = None,
        estimated_cost: float | None = 0.0,
    ) -> None:
        self.action = action
        self.parsed = parsed
        self.estimated_cost = estimated_cost
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        parsed = (
            self.parsed
            if self.parsed is not None
            else {"action": self.action, "rationale": "test-double"}
        )
        return LLMResponse(
            content=json.dumps(parsed, sort_keys=True),
            parsed=parsed,
            provider=request.provider,
            model=request.model,
            estimated_cost=self.estimated_cost,
        )


@pytest.fixture
def deterministic_llm_client():
    return DeterministicLLMClient
