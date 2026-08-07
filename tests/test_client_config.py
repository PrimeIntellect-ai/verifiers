import pytest
from pydantic import ValidationError

from verifiers.types import ClientConfig, EndpointClientConfig
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.types import SamplingConfig


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


def test_chat_dialect_omits_null_tools_for_openai_compatible_providers():
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": None,
        "provider_extension": None,
    }

    forwarded = ChatDialect().apply_overrides(
        body, "accounts/provider/models/model", SamplingConfig(temperature=0.0)
    )

    assert "tools" not in forwarded
    assert forwarded["provider_extension"] is None
    assert forwarded["model"] == "accounts/provider/models/model"
    assert forwarded["temperature"] == 0.0
