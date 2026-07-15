from __future__ import annotations

from icframe.core import load_domain_pack
from icframe.profiles import TOKEN_FACTORY_BASE_URL, LLMProfile, apply_llm_profile, load_profiles


def test_profiles_have_safe_defaults_and_redact_secret_values(tmp_path) -> None:
    path = tmp_path / "icframe.toml"
    path.write_text(
        """
[execution.local]
type = "local"
workers = 2

[execution.remote]
type = "nebius_jobs"
parent_id = "project-test"
image = "registry.test/worker@sha256:digest"
bucket = "artifacts"
public_ip = true

[llm.custom]
type = "openai_compatible"
base_url = "https://example.test/v1"
api_key_env = "CUSTOM_API_KEY"
"""
    )

    profiles = load_profiles(path, env={"CUSTOM_API_KEY": "do-not-leak"})
    payload = profiles.public_payload({"CUSTOM_API_KEY": "do-not-leak"})

    assert profiles.execution_profile("local").workers == 2
    assert profiles.execution_profile("remote").public_ip is True
    assert profiles.llm_profile("nebius-token-factory").base_url == TOKEN_FACTORY_BASE_URL
    assert payload["llm"]["custom"]["has_api_key"] is True
    assert "do-not-leak" not in str(payload)


def test_llm_profile_applies_model_provider_and_pricing_without_backend_coupling() -> None:
    pack = load_domain_pack("software_organization")
    configured = apply_llm_profile(
        pack,
        LLMProfile(
            provider="nebius-token-factory",
            base_url=TOKEN_FACTORY_BASE_URL,
            api_key_env="NEBIUS_API_KEY",
            model="openai/test-model",
            input_cost_per_million_tokens_usd=1.25,
            output_cost_per_million_tokens_usd=2.5,
        ),
    )

    llm = configured.spec.archetypes["llm_engineer"].llm
    assert llm is not None
    assert llm.provider == "nebius-token-factory"
    assert llm.model == "openai/test-model"
    assert llm.input_cost_per_million_tokens_usd == 1.25
    assert pack.spec.archetypes["llm_engineer"].llm.model != llm.model
