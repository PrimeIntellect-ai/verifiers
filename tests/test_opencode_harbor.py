import importlib.util
import json
from pathlib import Path


def _load_opencode_module():
    module_path = (
        Path(__file__).resolve().parent.parent
        / "environments"
        / "opencode_harbor"
        / "opencode_harbor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_opencode_harbor_module", module_path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_BASE_URL = "https://tunnel.example.invalid/rollout/abc123/v1"


def test_opencode_config_renders_valid_json():
    module = _load_opencode_module()
    config_str = module._build_opencode_config(
        base_url=FAKE_BASE_URL,
        disabled_tools=["webfetch", "question"],
        system_prompt_path="/opencode/prompt.txt",
    )

    config = json.loads(config_str)
    assert config["$schema"] == "https://opencode.ai/config.json"
    assert config["provider"]["intercepted"]["options"]["baseURL"] == FAKE_BASE_URL
    assert "agent" in config
    assert config["agent"]["build"]["prompt"] == "{file:/opencode/prompt.txt}"
    assert config["agent"]["build"]["tools"]["webfetch"] is False
    assert config["agent"]["build"]["tools"]["question"] is False


def test_opencode_config_with_mcp_servers():
    module = _load_opencode_module()

    mcp_servers = [
        {
            "name": "mcp-server",
            "transport": "streamable-http",
            "url": "http://mcp-server:8000/mcp",
        },
        {
            "name": "local-tools",
            "transport": "stdio",
            "command": "/usr/bin/mcp-server",
            "args": ["--verbose"],
        },
    ]
    config_str = module._build_opencode_config(
        base_url=FAKE_BASE_URL, mcp_servers=mcp_servers
    )

    config = json.loads(config_str)
    assert "mcp" in config
    assert config["mcp"]["mcp-server"] == {
        "type": "remote",
        "url": "http://mcp-server:8000/mcp",
    }
    assert config["mcp"]["local-tools"] == {
        "type": "local",
        "command": ["/usr/bin/mcp-server", "--verbose"],
    }


def test_opencode_config_no_mcp_when_empty():
    module = _load_opencode_module()
    config_str = module._build_opencode_config(base_url=FAKE_BASE_URL)
    config = json.loads(config_str)
    assert "mcp" not in config


def test_opencode_config_mcp_sse_transport():
    module = _load_opencode_module()

    mcp_servers = [
        {
            "name": "sse-server",
            "transport": "sse",
            "url": "http://mcp-server:8000/sse",
        },
    ]
    config_str = module._build_opencode_config(
        base_url=FAKE_BASE_URL, mcp_servers=mcp_servers
    )

    config = json.loads(config_str)
    assert config["mcp"]["sse-server"] == {
        "type": "remote",
        "url": "http://mcp-server:8000/sse",
    }


def test_opencode_config_mcp_stdio_no_args():
    module = _load_opencode_module()

    mcp_servers = [
        {
            "name": "simple-server",
            "transport": "stdio",
            "command": "/usr/bin/mcp-server",
        },
    ]
    config_str = module._build_opencode_config(
        base_url=FAKE_BASE_URL, mcp_servers=mcp_servers
    )

    config = json.loads(config_str)
    assert config["mcp"]["simple-server"] == {
        "type": "local",
        "command": ["/usr/bin/mcp-server"],
    }


def test_run_command_does_not_contain_config():
    module = _load_opencode_module()
    run_command = module._build_run_command("/app")
    assert "opencode.json" not in run_command
    assert "opencode run" in run_command
