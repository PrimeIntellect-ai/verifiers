import pytest

from verifiers.v1.cli.eval.compat import build_v1_eval_config
from verifiers.v1.clients.config import EvalClientConfig
from verifiers.v1.configs.eval import EvalConfig


def test_legacy_anthropic_provider_maps_v0_client_type():
    config, warnings = build_v1_eval_config({"id": "echo-v0", "provider": "anthropic"})

    assert warnings == []
    assert config["id"] == "echo-v0"
    assert config["client"] == {
        "base_url": "https://api.anthropic.com",
        "api_key_var": "ANTHROPIC_API_KEY",
        "v0_client_type": "anthropic_messages",
    }
    parsed = EvalConfig.model_validate(config)
    assert isinstance(parsed.client, EvalClientConfig)
    assert parsed.client.v0_client_type == "anthropic_messages"


def test_legacy_anthropic_api_client_type_maps_v0_client_type():
    config, warnings = build_v1_eval_config(
        {"id": "echo-v0", "api_client_type": "anthropic_messages"}
    )

    assert warnings == []
    assert config["client"] == {"v0_client_type": "anthropic_messages"}


def test_v1_anthropic_provider_still_requires_legacy_env():
    with pytest.raises(ValueError, match="provider anthropic"):
        build_v1_eval_config({"taskset": {"id": "echo-v1"}, "provider": "anthropic"})


def test_save_results_warns_because_v1_always_writes_artifacts():
    config, warnings = build_v1_eval_config({"id": "echo-v0", "save_results": False})

    assert config["id"] == "echo-v0"
    assert warnings == [
        "ignoring v0-only `save_results`: v1 evals always write artifacts"
    ]
