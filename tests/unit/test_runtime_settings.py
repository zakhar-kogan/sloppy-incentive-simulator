from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from icframe.llm import LiteLLMClient, LLMRequest
from icframe.runtime_settings import (
    DEFAULT_LLM_BASE_URL,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    fetch_openai_compatible_models,
    load_runtime_llm_settings,
    parse_openai_models_response,
)


def test_only_litellm_is_a_public_live_transport() -> None:
    import icframe

    assert hasattr(icframe, "LiteLLMClient")
    assert not hasattr(icframe, "FakeLLMClient")
    assert not hasattr(icframe, "AgnoClient")


def test_litellm_client_uses_live_transport_and_tracks_cost(monkeypatch) -> None:
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"ship"}'))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
        )

    module = SimpleNamespace(
        completion=completion,
        completion_cost=lambda **_: 0.0042,
    )
    monkeypatch.setitem(sys.modules, "litellm", module)
    client = LiteLLMClient(
        load_runtime_llm_settings(
            env={},
            dotenv_path="/missing/.env",
            base_url="http://localhost:11434",
            model="openai/test-model",
        )
    )
    response = client.complete(
        LLMRequest(
            llm_call_id="call-1",
            policy_decision_id="decision-1",
            model="fallback-model",
            system_prompt="Choose carefully.",
            prompt='{"valid_actions":["ship"]}',
        )
    )
    assert captured["model"] == "openai/test-model"
    assert captured["messages"][0]["role"] == "system"
    assert captured["response_format"] == {"type": "json_object"}
    assert response.parsed == {"action": "ship"}
    assert response.total_tokens == 15
    assert response.estimated_cost == pytest.approx(0.0042)


def test_runtime_settings_load_dotenv_and_redact_secret(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                f"{LLM_BASE_URL_ENV}=https://llm.example/v1",
                f"{LLM_API_KEY_ENV}=secret-from-dotenv",
                f"{LLM_MODEL_ENV}=openai/local-model",
            ]
        )
    )

    settings = load_runtime_llm_settings(env={}, dotenv_path=dotenv)

    assert settings.base_url == "https://llm.example/v1"
    assert settings.api_key == "secret-from-dotenv"
    assert settings.model == "openai/local-model"
    assert settings.api_key_source == ".env"
    assert "secret-from-dotenv" not in json.dumps(settings.redacted())


def test_runtime_settings_precedence_ui_then_env_then_dotenv(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                f"{LLM_BASE_URL_ENV}=https://dotenv.example/v1",
                f"{LLM_API_KEY_ENV}=dotenv-key",
                f"{LLM_MODEL_ENV}=dotenv-model",
            ]
        )
    )
    env = {
        LLM_BASE_URL_ENV: "https://env.example/v1",
        LLM_API_KEY_ENV: "env-key",
        LLM_MODEL_ENV: "env-model",
    }

    env_settings = load_runtime_llm_settings(env=env, dotenv_path=dotenv)
    ui_settings = load_runtime_llm_settings(
        env=env,
        dotenv_path=dotenv,
        base_url="https://ui.example/v1",
        api_key="ui-key",
        model="ui-model",
    )

    assert env_settings.base_url == "https://env.example/v1"
    assert env_settings.api_key == "env-key"
    assert env_settings.model == "env-model"
    assert env_settings.api_key_source == "shell env"
    assert ui_settings.base_url == "https://ui.example/v1"
    assert ui_settings.api_key == "ui-key"
    assert ui_settings.model == "ui-model"
    assert ui_settings.api_key_source == "ui session"


def test_runtime_settings_default_base_url_without_key() -> None:
    settings = load_runtime_llm_settings(env={}, dotenv_path="/missing/.env")

    assert settings.base_url == DEFAULT_LLM_BASE_URL
    assert settings.api_key is None
    assert settings.api_key_source == "missing"


def test_parse_openai_models_response() -> None:
    payload = {"data": [{"id": "b-model"}, {"id": "a-model"}, {"id": "a-model"}, {}]}

    assert parse_openai_models_response(payload) == ["a-model", "b-model"]


def test_failed_model_discovery_raises_runtime_error() -> None:
    settings = load_runtime_llm_settings(
        env={},
        dotenv_path="/missing/.env",
        base_url="http://127.0.0.1:9/v1",
        api_key="test-key",
        model="manual-model",
    )

    with pytest.raises(RuntimeError):
        fetch_openai_compatible_models(
            settings.base_url or "",
            settings.api_key or "",
            timeout=0.01,
        )
    assert settings.model == "manual-model"
