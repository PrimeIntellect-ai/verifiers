import importlib.util
import json
import subprocess
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


def test_opencode_config_renders_valid_json_after_shell_expansion():
    module = _load_opencode_module()
    run_command = module._build_run_command(
        "/app",
        disabled_tools=["webfetch", "question"],
        has_system_prompt=True,
    )

    prefix = "cat > ~/.config/opencode/opencode.json << EOFCONFIG\n"
    suffix = "\nEOFCONFIG"
    config_block = run_command.split(prefix, 1)[1].split(suffix, 1)[0]

    script = f"""OPENAI_BASE_URL=https://example.invalid SCHEMA_DOLLAR='$' bash -lc 'cat <<EOFCONFIG
{config_block}
EOFCONFIG'"""
    rendered = subprocess.run(
        script,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    config = json.loads(rendered)
    assert config["$schema"] == "https://opencode.ai/config.json"
    assert config["provider"]["intercepted"]["options"]["baseURL"] == (
        "https://example.invalid"
    )
    assert "agent" in config
    assert config["agent"]["build"]["prompt"] == "{file:/opencode/prompt.txt}"
    assert config["agent"]["build"]["tools"]["webfetch"] is False
    assert config["agent"]["build"]["tools"]["question"] is False
