from verifiers.envs.experimental.composable import Harness
from verifiers.envs.experimental.composable.harnesses import (
    DEFAULT_CODEX_VERSION,
    build_codex_install_script,
    build_codex_run_command,
    codex_harness,
)


def test_codex_install_script_downloads_pinned_release_with_retries():
    script = build_codex_install_script()

    assert DEFAULT_CODEX_VERSION in script
    assert "github.com/openai/codex/releases/download" in script
    assert "CODEX_TARGET=x86_64-unknown-linux-musl" in script
    assert "CODEX_TARGET=aarch64-unknown-linux-musl" in script
    assert "for attempt in range(1, 6)" in script
    assert "time.sleep(delay)" in script
    assert "ln -sf /opt/codex/codex /usr/local/bin/codex" in script


def test_codex_run_command_uses_verifiers_proxy_and_prompt_stdin():
    command = build_codex_run_command(
        agent_workdir="/workspace/src",
        timeout_seconds=600,
        model_reasoning_effort="xhigh",
    )

    assert "timeout --kill-after=30s 600s /opt/codex/codex" in command
    assert 'model_provider="vf_proxy"' in command
    assert "model_providers.vf_proxy.base_url" in command
    assert "OPENAI_BASE_URL" in command
    assert 'model_reasoning_effort="xhigh"' in command
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "exec --ignore-user-config --ignore-rules --skip-git-repo-check" in command
    assert "CODEX_AGENT_WORKDIR=/workspace/src" in command
    assert "CODEX_PROMPT=-" in command
    assert "< /codex/prompt.md" in command
    assert "--enable goals" not in command


def test_codex_goal_run_command_uses_goal_prompt():
    command = build_codex_run_command(
        goal_mode=True,
        agent_workdir="/workspace/src",
        goal_prompt="/goal Read /codex/goal.md and finish.",
    )

    assert "--enable goals" in command
    assert "/goal Read /codex/goal.md and finish." in command
    assert "< /codex/prompt.md" not in command


def test_codex_harness_wires_files_and_prompt():
    harness = codex_harness(
        system_prompt="Use shell commands.",
        goal_mode=True,
        agent_workdir="/workspace/src",
        timeout_seconds=600,
    )

    assert isinstance(harness, Harness)
    assert harness.system_prompt == "Use shell commands."
    assert harness.instruction_path == "/codex/instruction.md"
    assert harness.system_prompt_path == "/codex/system.md"
    assert harness.log_path == "/logs/agent/codex.log"
    assert "/goal Read /codex/goal.md" in harness.run_command
    assert "rust-v0.132.0" in harness.install_script
