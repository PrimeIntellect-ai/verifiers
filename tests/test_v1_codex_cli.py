from typing import Any, cast

import pytest
import verifiers as vf
from harnesses import CodexCLI, CodexCLIConfig, CodexCLIProgramConfig
from harnesses.codex_cli import codex_chatgpt_auth_json


def test_codex_cli_builds_openai_api_sandbox_program() -> None:
    harness = CodexCLI(
        config=CodexCLIConfig(
            system_prompt="Use tests.",
            program=CodexCLIProgramConfig(agent_workdir="/workspace"),
        )
    )
    program = cast(dict[str, Any], harness.program_config.data())
    command = cast(list[str], program["command"])
    setup = cast(str, program["setup"])
    files = cast(dict[str, object], program["files"])
    artifacts = cast(dict[str, object], program["artifacts"])
    env = cast(dict[str, object], program["env"])
    run_script = command[-1]

    assert isinstance(harness, vf.Harness)
    assert program["sandbox"] is not False
    assert "CODEX_RELEASE=latest" in setup
    assert "https://chatgpt.com/codex/install.sh" in setup
    assert "apt-get -o Acquire::Retries=3 update" in setup
    assert "apt-get -o Acquire::Retries=3 install" in setup
    assert "python3" in setup
    assert "/usr/local/bin/python" in setup
    assert "/codex-cli/instruction.txt" in files
    assert "/codex-cli/system.txt" in files
    assert "codex_cli_log" in artifacts
    assert "codex_cli_last_message" in artifacts
    assert env["OPENAI_MODEL"] == "runtime.model"
    assert "printf '%s' \"$OPENAI_API_KEY\" | codex login --with-api-key" in run_script
    assert 'model_provider="openai"' in run_script
    assert "openai_base_url=" in run_script
    assert "developer_instructions=" in run_script
    assert "--dangerously-bypass-approvals-and-sandbox" in run_script
    assert "/workspace" in run_script


def test_codex_cli_version_spec_sets_installer_release() -> None:
    harness = CodexCLI(config=CodexCLIConfig(version="codex@0.137.0"))
    program = cast(dict[str, Any], harness.program_config.data())
    setup = cast(str, program["setup"])

    assert "CODEX_RELEASE=0.137.0" in setup


def test_codex_cli_chatgpt_auth_uses_forwarded_auth_json_secret() -> None:
    harness = CodexCLI(
        config=CodexCLIConfig(
            program=CodexCLIProgramConfig(
                auth_mode="chatgpt",
            )
        )
    )
    program = cast(dict[str, Any], harness.program_config.data())
    env = cast(dict[str, object], program["env"])
    run_script = cast(list[str], program["command"])[-1]

    assert "dirs" not in program
    assert cast(dict[str, object], env["CODEX_AUTH_JSON"]) == {
        "fn": "harnesses.codex_cli:codex_chatgpt_auth_json",
        "auth_json_var": "CODEX_AUTH_JSON",
    }
    assert 'printf \'%s\' "$CODEX_AUTH_JSON" > "$CODEX_HOME/auth.json"' in run_script
    assert "codex login --with-api-key" not in run_script
    assert "openai_base_url=" not in run_script


def test_codex_chatgpt_auth_json_reads_configured_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_CODEX_AUTH_JSON", '{"auth_mode":"chatgpt"}')

    assert codex_chatgpt_auth_json("CUSTOM_CODEX_AUTH_JSON") == (
        '{"auth_mode":"chatgpt"}'
    )


def test_codex_chatgpt_auth_json_requires_configured_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEX_AUTH_JSON", raising=False)

    with pytest.raises(RuntimeError, match="CODEX_AUTH_JSON must contain"):
        codex_chatgpt_auth_json("CODEX_AUTH_JSON")


def test_codex_cli_imports_from_package() -> None:
    assert CodexCLI
