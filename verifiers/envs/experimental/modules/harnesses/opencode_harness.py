from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any

from verifiers.envs.experimental.channels import SandboxSpec
from verifiers.envs.experimental.task import Task
from verifiers.envs.experimental.modules.harnesses.cli_harness import (
    CliHarness,
)

if TYPE_CHECKING:
    from verifiers.envs.experimental.resources import Resources

DEFAULT_RELEASE_REPO = "PrimeIntellect-ai/opencode"
DEFAULT_RELEASE_VERSION = "1.1.63-rl2"
DEFAULT_RELEASE_SHA256 = (
    "47f4102796da50769e27d2c9ea6a9cf7941f76898390cb497278cab39c4b6ed4"
)

DEFAULT_OPENCODE_SYSTEM_PROMPT = """You are OpenCode, a coding agent running in a sandbox.
Work directly in the task workspace. Make the minimal changes needed to complete the task.
"""


def build_opencode_install_command(
    release_repo: str = DEFAULT_RELEASE_REPO,
    release_version: str = DEFAULT_RELEASE_VERSION,
    release_sha256: str = DEFAULT_RELEASE_SHA256,
    install_ripgrep: bool = True,
) -> str:
    rg_install = (
        "apt-get install -y -qq ripgrep > /dev/null 2>&1 || true"
        if install_ripgrep
        else ""
    )
    return f"""\
set -e
apt-get update -qq && apt-get install -y -qq curl tar > /dev/null 2>&1
{rg_install}
OPENCODE_RELEASE_REPO={shlex.quote(release_repo)}
OPENCODE_RELEASE_VERSION={shlex.quote(release_version)}
case "$(uname -m)" in
  x86_64) OPENCODE_ARCH=x64 ;;
  aarch64|arm64) OPENCODE_ARCH=arm64 ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac
OPENCODE_ASSET="opencode-linux-$OPENCODE_ARCH.tar.gz"
OPENCODE_RELEASE_TAG="${{OPENCODE_RELEASE_VERSION#v}}"
OPENCODE_RELEASE_URL="https://github.com/$OPENCODE_RELEASE_REPO/releases/download/v$OPENCODE_RELEASE_TAG/$OPENCODE_ASSET"
mkdir -p "$HOME/.opencode/bin"
curl -fsSL "$OPENCODE_RELEASE_URL" -o /tmp/opencode.tar.gz
echo {shlex.quote(release_sha256 + "  /tmp/opencode.tar.gz")} | sha256sum -c -
tar -xzf /tmp/opencode.tar.gz -C /tmp
install -m 755 /tmp/opencode "$HOME/.opencode/bin/opencode"
"""


def build_opencode_config(
    system_prompt_path: str | None,
    disabled_tools: list[str] | None,
    disable_compaction: bool = True,
    provider_key: str = "${OPENAI_MODEL%%/*}",
    provider_display_name: str | None = None,
    model_id: str = "$OPENAI_MODEL",
    model_key: str = "${OPENAI_MODEL##*/}",
    model_display_name: str | None = None,
    provider_timeout_ms: int = 3_600_000,
) -> str:
    config: dict[str, Any] = {
        "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
        "provider": {
            provider_key: {
                "npm": "@ai-sdk/openai-compatible",
                "name": provider_display_name or provider_key,
                "options": {
                    "baseURL": "$OPENAI_BASE_URL",
                    "apiKey": "${OPENAI_API_KEY:-intercepted}",
                    "timeout": provider_timeout_ms,
                },
                "models": {
                    model_key: {
                        "name": model_display_name or model_key,
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        "interleaved": {"field": "reasoning_content"},
                    }
                },
            }
        },
        "model": model_id,
    }
    if disable_compaction:
        config["compaction"] = {"auto": False, "prune": False}
    agent: dict[str, Any] = {}
    if system_prompt_path:
        agent["prompt"] = "{file:" + system_prompt_path + "}"
    if disabled_tools:
        agent["tools"] = {tool: False for tool in disabled_tools}
    if agent:
        config["agent"] = {"build": agent}
    return json.dumps(config, indent=2)


class OpenCode(CliHarness):
    """Reference OpenCode sandbox harness."""

    DEFAULT_DISABLED_TOOLS = [
        "apply_patch",
        "write",
        "multiedit",
        "glob",
        "todowrite",
        "todoread",
        "websearch",
        "task",
        "batch",
        "list",
        "read",
        "question",
        "webfetch",
        "grep",
        "plan_enter",
        "plan_exit",
        "lsp",
        "codesearch",
        "skill",
    ]

    def __init__(
        self,
        agent_workdir: str = "/app",
        instruction_path: str = "/opencode/prompt.txt",
        system_prompt_path: str = "/opencode/system.txt",
        log_path: str = "/opencode/logs.txt",
        system_prompt: str | None = DEFAULT_OPENCODE_SYSTEM_PROMPT,
        disabled_tools: list[str] | None = None,
        sandbox: SandboxSpec | None = None,
        install_command: str | None = None,
        install_timeout: int = 300,
        release_repo: str = DEFAULT_RELEASE_REPO,
        release_version: str = DEFAULT_RELEASE_VERSION,
        release_sha256: str = DEFAULT_RELEASE_SHA256,
        install_ripgrep: bool = True,
        allow_git: bool = False,
        disable_compaction: bool = True,
        provider_key: str = "${OPENAI_MODEL%%/*}",
        provider_display_name: str | None = None,
        model_id: str = "$OPENAI_MODEL",
        model_key: str = "${OPENAI_MODEL##*/}",
        model_display_name: str | None = None,
        provider_timeout_ms: int = 3_600_000,
        **kwargs: Any,
    ):
        disabled_tools = (
            self.DEFAULT_DISABLED_TOOLS if disabled_tools is None else disabled_tools
        )
        config = build_opencode_config(
            system_prompt_path=system_prompt_path if system_prompt else None,
            disabled_tools=disabled_tools,
            disable_compaction=disable_compaction,
            provider_key=provider_key,
            provider_display_name=provider_display_name,
            model_id=model_id,
            model_key=model_key,
            model_display_name=model_display_name,
            provider_timeout_ms=provider_timeout_ms,
        )
        script = f"""\
set -eo pipefail
export PATH="$HOME/.opencode/bin:$PATH"
export OPENCODE_DISABLE_FILETIME_CHECK=true
export ALLOW_GIT={"1" if allow_git else "0"}
SCHEMA_DOLLAR='$'
mkdir -p ~/.config/opencode /opencode /logs/agent "$AGENT_WORKDIR"
cat > ~/.config/opencode/opencode.json <<EOFCONFIG
{config}
EOFCONFIG
cd "$AGENT_WORKDIR"
cat {shlex.quote(instruction_path)} | opencode run 2>&1 | tee {shlex.quote(log_path)}
"""
        command = f"bash -lc {shlex.quote(script)}"
        super().__init__(
            command=command,
            instruction_path=instruction_path,
            system_prompt_path=system_prompt_path,
            agent_workdir=agent_workdir,
            log_path=log_path,
            system_prompt=system_prompt,
            sandbox=sandbox,
            install_command=install_command
            or build_opencode_install_command(
                release_repo=release_repo,
                release_version=release_version,
                release_sha256=release_sha256,
                install_ripgrep=install_ripgrep,
            ),
            install_timeout=install_timeout,
            **kwargs,
        )

    async def setup_sandbox_contents(
        self, task: Task, state, resources: Resources
    ) -> None:
        await super().setup_sandbox_contents(task, state, resources)
        await self.with_retry(self.sandbox_client.execute_command)(
            state["sandbox_id"], "mkdir -p /opencode"
        )

    async def finalize_state(self, task, state, resources):
        if state.get("sandbox_id") and self.log_path:
            try:
                result = await self.with_retry(self.sandbox_client.read_file)(
                    state["sandbox_id"], self.log_path, timeout=10
                )
                state["agent_logs"] = getattr(result, "content", "")
            except Exception:
                pass
        return await super().finalize_state(task, state, resources)
