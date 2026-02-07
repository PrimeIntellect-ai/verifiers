from pathlib import Path

from verifiers.utils.eval_utils import load_endpoints


def test_load_endpoints_python_registry_normalizes_to_lists(tmp_path: Path):
    registry_path = tmp_path / "endpoints.py"
    registry_path.write_text(
        "ENDPOINTS = {\n"
        '    "gpt-4.1-mini": {"model": "gpt-4.1-mini", "url": "https://api.openai.com/v1", "key": "OPENAI_API_KEY"},\n'
        "}\n",
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert set(endpoints.keys()) == {"gpt-4.1-mini"}
    assert len(endpoints["gpt-4.1-mini"]) == 1
    endpoint = endpoints["gpt-4.1-mini"][0]
    assert endpoint["model"] == "gpt-4.1-mini"
    assert endpoint["url"] == "https://api.openai.com/v1"
    assert endpoint["key"] == "OPENAI_API_KEY"


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


def test_load_endpoints_python_registry_supports_list_variants(tmp_path: Path):
    registry_path = tmp_path / "endpoints.py"
    registry_path.write_text(
        "ENDPOINTS = {\n"
        '    "gpt-5-mini": [\n'
        '        {"model": "gpt-5-mini", "url": "https://a.example/v1", "key": "A_KEY"},\n'
        '        {"model": "gpt-5-mini", "url": "https://b.example/v1", "key": "A_KEY"},\n'
        "    ]\n"
        "}\n",
        encoding="utf-8",
    )

    endpoints = load_endpoints(str(registry_path))

    assert set(endpoints.keys()) == {"gpt-5-mini"}
    assert len(endpoints["gpt-5-mini"]) == 2
    assert endpoints["gpt-5-mini"][0]["url"] == "https://a.example/v1"
    assert endpoints["gpt-5-mini"][1]["url"] == "https://b.example/v1"


def test_load_endpoints_directory_prefers_toml_then_python(tmp_path: Path):
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

    toml_registry.unlink()
    endpoints = load_endpoints(str(tmp_path))
    assert set(endpoints.keys()) == {"from-py"}


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
