from pathlib import Path

import pytest
from pydantic import ValidationError

from verifiers.types import ClientConfig
from verifiers.utils.eval_utils import load_endpoints


def test_load_endpoints_rejects_python_registry_path(tmp_path: Path):
    registry_path = tmp_path / "endpoints.py"
    registry_path.write_text(
        "ENDPOINTS = {\n"
        '    "gpt-4.1-mini": {"model": "gpt-4.1-mini", "url": "https://api.openai.com/v1", "key": "OPENAI_API_KEY"},\n'
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Python endpoint registries"):
        load_endpoints(str(registry_path))


def test_load_endpoints_rejects_deprecated_client_type_field(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "haiku"\n'
        'model = "claude-haiku-4-5"\n'
        'url = "https://api.anthropic.com"\n'
        'key = "ANTHROPIC_API_KEY"\n'
        'client_type = "anthropic_messages"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints == {}


def test_load_endpoints_toml_groups_variants_by_endpoint_id(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "gpt-5-mini"\n'
        'model = "openai/gpt-5-mini"\n'
        'url = "https://api.pinference.ai/api/v1"\n'
        'key = "PRIME_API_KEY"\n'
        "\n"
        "[[endpoint]]\n"
        'endpoint_id = "gpt-5-mini"\n'
        'model = "openai/gpt-5-mini"\n'
        'url = "https://api.openai.com/v1"\n'
        'key = "OPENAI_API_KEY"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert set(endpoints.keys()) == {"gpt-5-mini"}
    assert len(endpoints["gpt-5-mini"]) == 2
    assert endpoints["gpt-5-mini"][0]["url"] == "https://api.pinference.ai/api/v1"
    assert endpoints["gpt-5-mini"][1]["url"] == "https://api.openai.com/v1"


def test_load_endpoints_toml_accepts_long_field_names(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "gpt-5-mini"\n'
        'model = "openai/gpt-5-mini"\n'
        'api_base_url = "https://api.pinference.ai/api/v1"\n'
        'api_key_var = "PRIME_API_KEY"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints["gpt-5-mini"][0]["url"] == "https://api.pinference.ai/api/v1"
    assert endpoints["gpt-5-mini"][0]["key"] == "PRIME_API_KEY"


def test_load_endpoints_toml_accepts_matching_short_and_long_fields(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "gpt-5-mini"\n'
        'model = "openai/gpt-5-mini"\n'
        'url = "https://api.pinference.ai/api/v1"\n'
        'api_base_url = "https://api.pinference.ai/api/v1"\n'
        'key = "PRIME_API_KEY"\n'
        'api_key_var = "PRIME_API_KEY"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints["gpt-5-mini"][0]["url"] == "https://api.pinference.ai/api/v1"
    assert endpoints["gpt-5-mini"][0]["key"] == "PRIME_API_KEY"


def test_load_endpoints_toml_rejects_conflicting_url_fields(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "gpt-5-mini"\n'
        'model = "openai/gpt-5-mini"\n'
        'url = "https://a.example/v1"\n'
        'api_base_url = "https://b.example/v1"\n'
        'key = "PRIME_API_KEY"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints == {}


def test_load_endpoints_toml_rejects_conflicting_key_fields(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "gpt-5-mini"\n'
        'model = "openai/gpt-5-mini"\n'
        'url = "https://a.example/v1"\n'
        'key = "A_KEY"\n'
        'api_key_var = "B_KEY"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints == {}


def test_load_endpoints_directory_uses_toml_and_warns_on_ignored_python(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    python_registry = tmp_path / "endpoints.py"
    toml_registry = tmp_path / "endpoints.toml"

    python_registry.write_text(
        "ENDPOINTS = {\n"
        '    "from-py": {"model": "m", "url": "https://py.example/v1", "key": "PY_KEY"},\n'
        "}\n",
        encoding="utf-8",
    )
    toml_registry.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "from-toml"\n'
        'model = "m"\n'
        'url = "https://toml.example/v1"\n'
        'key = "TOML_KEY"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(tmp_path))
    assert set(endpoints.keys()) == {"from-toml"}
    assert "Ignoring deprecated Python endpoint registry" in capsys.readouterr().err

    toml_registry.unlink()
    with pytest.raises(ValueError, match="Python endpoint registries"):
        load_endpoints(str(tmp_path))


def test_qwen3_vl_endpoint_ids_map_to_vl_models():
    endpoints = load_endpoints("./configs/endpoints.toml")

    assert endpoints["qwen3-vl-30b-i"][0]["model"] == "qwen/qwen3-vl-30b-a3b-instruct"
    assert endpoints["qwen3-vl-30b-t"][0]["model"] == "qwen/qwen3-vl-30b-a3b-thinking"
    assert (
        endpoints["qwen3-vl-235b-i"][0]["model"] == "qwen/qwen3-vl-235b-a22b-instruct"
    )
    assert (
        endpoints["qwen3-vl-235b-t"][0]["model"] == "qwen/qwen3-vl-235b-a22b-thinking"
    )


def test_load_endpoints_toml_accepts_type_shorthand(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "haiku"\n'
        'model = "claude-haiku-4-5"\n'
        'url = "https://api.anthropic.com"\n'
        'key = "ANTHROPIC_API_KEY"\n'
        'type = "anthropic_messages"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints["haiku"][0]["api_client_type"] == "anthropic_messages"


def test_load_endpoints_toml_accepts_openai_responses_type(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "gpt-responses"\n'
        'model = "gpt-5.2"\n'
        'url = "https://api.openai.com/v1"\n'
        'key = "OPENAI_API_KEY"\n'
        'type = "openai_responses"\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints["gpt-responses"][0]["api_client_type"] == "openai_responses"


def test_load_endpoints_toml_accepts_headers_table(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "proxy"\n'
        'model = "m"\n'
        'url = "https://api.example/v1"\n'
        'key = "K"\n'
        'headers = { "X-Custom" = "v1" }\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints["proxy"][0]["extra_headers"] == {"X-Custom": "v1"}


def test_load_endpoints_toml_accepts_extra_headers_alias(tmp_path: Path):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "proxy"\n'
        'model = "m"\n'
        'url = "https://api.example/v1"\n'
        'key = "K"\n'
        'extra_headers = { "X-A" = "a" }\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints["proxy"][0]["extra_headers"] == {"X-A": "a"}


def test_load_endpoints_toml_rejects_headers_and_extra_headers_together(
    tmp_path: Path,
):
    registry_path = tmp_path / "endpoints.toml"
    registry_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "proxy"\n'
        'model = "m"\n'
        'url = "https://api.example/v1"\n'
        'key = "K"\n'
        'headers = { "X-A" = "a" }\n'
        'extra_headers = { "X-B" = "b" }\n',
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert endpoints == {}


def test_load_endpoints_malformed_headers_string_falls_back_to_empty_registry(
    tmp_path: Path,
):
    toml_path = tmp_path / "endpoints.toml"
    toml_path.write_text(
        "[[endpoint]]\n"
        'endpoint_id = "x"\n'
        'model = "m"\n'
        'url = "https://api.example/v1"\n'
        'key = "K"\n'
        'headers = "invalid"\n',
        encoding="utf-8",
    )

    assert load_endpoints(str(toml_path)) == {}


def test_client_config_validates_extra_header_keys():
    with pytest.raises(ValidationError):
        ClientConfig(extra_headers={"": "x"})
    with pytest.raises(ValidationError):
        ClientConfig(extra_headers={"X": 1})  # type: ignore[arg-type]
