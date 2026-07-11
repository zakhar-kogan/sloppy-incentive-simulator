from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
LLM_BASE_URL_ENV = "ICFRAME_LLM_BASE_URL"
LLM_API_KEY_ENV = "ICFRAME_LLM_API_KEY"
LLM_MODEL_ENV = "ICFRAME_LLM_MODEL"


@dataclass(frozen=True)
class RuntimeLLMSettings:
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    api_key_source: str = "missing"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def redacted(self) -> dict[str, str | bool | None]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "has_api_key": self.has_api_key,
            "api_key_source": self.api_key_source,
        }


def load_runtime_llm_settings(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path = ".env",
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    api_key_source: str = "ui session",
) -> RuntimeLLMSettings:
    """Load current, non-spec LLM endpoint settings.

    Precedence is explicit/session overrides, then shell environment, then local
    .env. The API key is never written back to specs or artifacts by this layer.
    """

    env = env if env is not None else os.environ
    dotenv = _read_dotenv(dotenv_path)

    env_base_url = env.get(LLM_BASE_URL_ENV)
    env_api_key = env.get(LLM_API_KEY_ENV)
    env_model = env.get(LLM_MODEL_ENV)

    resolved_api_key = (
        _non_empty(api_key) or _non_empty(env_api_key) or _non_empty(dotenv.get(LLM_API_KEY_ENV))
    )
    source = "missing"
    if _non_empty(api_key):
        source = api_key_source
    elif _non_empty(env_api_key):
        source = "shell env"
    elif _non_empty(dotenv.get(LLM_API_KEY_ENV)):
        source = ".env"

    return RuntimeLLMSettings(
        base_url=(
            _non_empty(base_url)
            or _non_empty(env_base_url)
            or _non_empty(dotenv.get(LLM_BASE_URL_ENV))
            or DEFAULT_LLM_BASE_URL
        ),
        api_key=resolved_api_key,
        model=_non_empty(model) or _non_empty(env_model) or _non_empty(dotenv.get(LLM_MODEL_ENV)),
        api_key_source=source,
    )


def parse_openai_models_response(payload: str | bytes | dict[str, object]) -> list[str]:
    data = json.loads(payload) if isinstance(payload, str | bytes) else payload
    rows = data.get("data", []) if isinstance(data, dict) else []
    model_ids = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            model_ids.append(row["id"])
    return sorted(set(model_ids))


def fetch_openai_compatible_models(
    base_url: str,
    api_key: str,
    *,
    timeout: float = 15.0,
) -> list[str]:
    url = f"{base_url.rstrip('/')}/models"
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return parse_openai_models_response(response.read())
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"failed to fetch models from {url}: {exc}") from exc


def _read_dotenv(path: str | Path) -> dict[str, str]:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return {}
    values = {}
    for line in dotenv_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        values[key] = value
    return values


def _non_empty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
