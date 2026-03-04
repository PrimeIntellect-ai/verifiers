import pytest
from pydantic import ValidationError

from verifiers.types import ClientConfig, EndpointClientConfig


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
