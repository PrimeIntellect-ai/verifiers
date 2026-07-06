import pytest
from pydantic import ValidationError

from verifiers.types import ClientConfig, EndpointClientConfig
from verifiers.utils.client_utils import (
    _build_headers_and_api_key,
    resolve_client_config,
)


def test_client_config_allows_leaf_endpoint_configs():
    config = ClientConfig(
        api_base_url="http://localhost:8000/v1",
        endpoint_configs=[
            EndpointClientConfig(api_base_url="http://localhost:8001/v1"),
            {"api_base_url": "http://localhost:8002/v1"},
        ],
    )

    assert len(config.endpoint_configs) == 2
    assert config.endpoint_configs[0].api_base_url == "http://localhost:8001/v1"
    assert config.endpoint_configs[1].api_base_url == "http://localhost:8002/v1"


def test_client_config_rejects_recursive_endpoint_configs():
    with pytest.raises(ValidationError, match="cannot include endpoint_configs"):
        ClientConfig.model_validate(
            {
                "api_base_url": "http://localhost:8000/v1",
                "endpoint_configs": [
                    {
                        "api_base_url": "http://localhost:8001/v1",
                        "endpoint_configs": [
                            {"api_base_url": "http://localhost:8002/v1"}
                        ],
                    }
                ],
            }
        )


def test_client_config_accepts_empty_nested_endpoint_configs_key():
    config = ClientConfig.model_validate(
        {
            "api_base_url": "http://localhost:8000/v1",
            "endpoint_configs": [
                {
                    "api_base_url": "http://localhost:8001/v1",
                    "endpoint_configs": [],
                }
            ],
        }
    )

    assert len(config.endpoint_configs) == 1
    assert config.endpoint_configs[0].api_base_url == "http://localhost:8001/v1"


def test_prime_v0_client_context_comes_only_from_environment(monkeypatch, tmp_path):
    prime_dir = tmp_path / ".prime"
    prime_dir.mkdir()
    (prime_dir / "config.json").write_text(
        '{"api_key": "profile-key", "team_id": "profile-team"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    headers, api_key = _build_headers_and_api_key(ClientConfig())

    assert headers == {}
    assert api_key is None

    monkeypatch.setenv("PRIME_API_KEY", "env-key")
    monkeypatch.setenv("PRIME_TEAM_ID", "env-team")

    headers, api_key = _build_headers_and_api_key(ClientConfig())

    assert headers == {"X-Prime-Team-ID": "env-team"}
    assert api_key == "env-key"


def test_prime_v0_client_uses_inference_url_from_environment(monkeypatch):
    monkeypatch.setenv("PRIME_INFERENCE_URL", "https://runtime.example/v1/")

    default_config = resolve_client_config(ClientConfig())
    explicit_config = resolve_client_config(
        ClientConfig(api_base_url="https://explicit.example/v1")
    )

    assert default_config.api_base_url == "https://runtime.example/v1"
    assert explicit_config.api_base_url == "https://explicit.example/v1"
