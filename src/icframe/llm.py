from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Protocol

from pydantic import Field

from icframe.domain.base import ICFrameModel


class LLMRequest(ICFrameModel):
    llm_call_id: str
    policy_decision_id: str
    provider: str = "mock"
    model: str
    system_prompt: str = ""
    prompt: str
    response_schema: dict[str, object] = Field(default_factory=dict)
    temperature: float = 0.0
    require_json: bool = True

    @property
    def request_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"llm_call_id"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class LLMResponse(ICFrameModel):
    content: str
    parsed: dict[str, object] = Field(default_factory=dict)
    provider: str = "mock"
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    retry_count: int = 0
    fallback_used: bool = False
    error_type: str | None = None


class LLMCallRecord(ICFrameModel):
    llm_call_id: str
    policy_decision_id: str
    provider: str
    model: str
    request_hash: str
    response_hash: str
    parsed_response: dict[str, object] = Field(default_factory=dict)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    retry_count: int = 0
    fallback_used: bool = False
    error_type: str | None = None
    redaction_mode: str = "balanced"


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a model completion for a policy decision request."""


class FakeLLMClient:
    """Deterministic LLM client for tests and replay-free local development."""

    def __init__(self, action: str, *, rationale: str = "fake-client") -> None:
        self.action = action
        self.rationale = rationale
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        parsed: dict[str, object] = {"action": self.action, "rationale": self.rationale}
        return LLMResponse(
            content=json.dumps(parsed, sort_keys=True),
            parsed=parsed,
            provider=request.provider,
            model=request.model,
        )


class RecordedLLMClient:
    """Replay LLM responses from a prior llm_calls.jsonl artifact."""

    def __init__(self, records_path: str | Path, *, fail_on_missing: bool = True) -> None:
        self.fail_on_missing = fail_on_missing
        self.records_by_id: dict[str, LLMCallRecord] = {}
        self.records_by_hash: dict[str, LLMCallRecord] = {}
        path = Path(records_path)
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = LLMCallRecord.model_validate_json(line)
                self.records_by_id[record.llm_call_id] = record
                self.records_by_hash[record.request_hash] = record

    def complete(self, request: LLMRequest) -> LLMResponse:
        record = self.records_by_id.get(request.llm_call_id) or self.records_by_hash.get(
            request.request_hash
        )
        if record is None:
            if self.fail_on_missing:
                raise ValueError(f"missing recorded LLM call for {request.llm_call_id}")
            return LLMResponse(
                content="{}",
                parsed={},
                provider=request.provider,
                model=request.model,
                error_type="missing_recorded_call",
            )
        return LLMResponse(
            content=json.dumps(record.parsed_response, sort_keys=True),
            parsed=record.parsed_response,
            provider=record.provider,
            model=record.model,
            latency_ms=record.latency_ms,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.total_tokens,
            estimated_cost=record.estimated_cost,
            retry_count=record.retry_count,
            fallback_used=record.fallback_used,
            error_type=record.error_type,
        )


class LiteLLMClient:
    """Optional live LiteLLM adapter.

    LiteLLM is imported lazily so the core simulator and tests do not require live
    provider dependencies or API keys.
    """

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError("install litellm to use LiteLLMClient") from exc

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        kwargs: dict[str, object] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.require_json or request.response_schema:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        response = litellm.completion(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = response.choices[0].message.content or ""
        parsed = _parse_json_object(content)
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return LLMResponse(
            content=content,
            parsed=parsed,
            provider=request.provider,
            model=request.model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class AgnoClient:
    """Optional wrapper for an Agno agent/team object supplied by the caller."""

    def __init__(self, agent: object) -> None:
        self.agent = agent

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not hasattr(self.agent, "run"):
            raise RuntimeError("AgnoClient expects an object with a run(prompt) method")
        started = time.perf_counter()
        result = self.agent.run(request.prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = getattr(result, "content", str(result))
        return LLMResponse(
            content=content,
            parsed=_parse_json_object(content),
            provider="agno",
            model=request.model,
            latency_ms=latency_ms,
        )


class AgnoPolicyAdapter(AgnoClient):
    """Semantic alias for using an Agno agent as an IncentiveSpec policy backend."""


def llm_call_record_from_response(
    request: LLMRequest,
    response: LLMResponse,
    *,
    redaction_mode: str = "balanced",
) -> LLMCallRecord:
    response_payload = response.model_dump(mode="json")
    return LLMCallRecord(
        llm_call_id=request.llm_call_id,
        policy_decision_id=request.policy_decision_id,
        provider=response.provider,
        model=response.model,
        request_hash=request.request_hash,
        response_hash=hashlib.sha256(
            json.dumps(response_payload, sort_keys=True).encode()
        ).hexdigest(),
        parsed_response=response.parsed,
        latency_ms=response.latency_ms,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        estimated_cost=response.estimated_cost,
        retry_count=response.retry_count,
        fallback_used=response.fallback_used,
        error_type=response.error_type,
        redaction_mode=redaction_mode,
    )


def _parse_json_object(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
