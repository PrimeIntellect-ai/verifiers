from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .cli import CLIHarness
from .configs import (
    OPENCODE_DEFAULT_AGENT_WORKDIR,
    OPENCODE_DEFAULT_DISABLED_TOOLS,
    OPENCODE_DEFAULT_INSTRUCTION_PATH,
    OPENCODE_DEFAULT_LOG_PATH,
    OPENCODE_DEFAULT_RELEASE_REPO,
    OPENCODE_DEFAULT_RELEASE_SHA256,
    OPENCODE_DEFAULT_RELEASE_VERSION,
    OPENCODE_DEFAULT_SYSTEM_PROMPT,
    OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH,
    OpenCodeConfig,
)
from ...config import SandboxConfig
from ...utils.mcp_proxy_utils import proxy_command
from ...utils.prompt_utils import (
    state_system_prompt_text,
    task_text as task_instruction_text,
)

DEFAULT_RELEASE_REPO = OPENCODE_DEFAULT_RELEASE_REPO
DEFAULT_RELEASE_VERSION = OPENCODE_DEFAULT_RELEASE_VERSION
DEFAULT_RELEASE_SHA256 = OPENCODE_DEFAULT_RELEASE_SHA256
DEFAULT_AGENT_WORKDIR = OPENCODE_DEFAULT_AGENT_WORKDIR
DEFAULT_INSTRUCTION_PATH = OPENCODE_DEFAULT_INSTRUCTION_PATH
DEFAULT_SYSTEM_PROMPT_PATH = OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH
DEFAULT_LOG_PATH = OPENCODE_DEFAULT_LOG_PATH
DEFAULT_SYSTEM_PROMPT = OPENCODE_DEFAULT_SYSTEM_PROMPT
DEFAULT_DISABLED_TOOLS = list(OPENCODE_DEFAULT_DISABLED_TOOLS)

_UNSET: object = object()


class OpenCode(CLIHarness):
    config_type = OpenCodeConfig

    def __init__(
        self,
        *,
        agent_workdir: str | None = None,
        instruction_path: str | None = None,
        system_prompt_path: str | None = None,
        log_path: str | None = None,
        system_prompt: object | None = _UNSET,
        disabled_tools: list[str] | None = None,
        allow_git: bool | None = None,
        disable_compaction: bool | None = None,
        release_repo: str | None = None,
        release_version: str | None = None,
        release_sha256: str | None = None,
        install_ripgrep: bool | None = None,
        provider_timeout_ms: int | None = None,
        sandbox: bool | Mapping[str, object] | SandboxConfig | None = None,
        program: Mapping[str, object] | None = None,
        max_turns: int | None = None,
        config: OpenCodeConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        config_data: dict[str, object] = {
            "agent_workdir": agent_workdir,
            "instruction_path": instruction_path,
            "system_prompt_path": system_prompt_path,
            "log_path": log_path,
            "disabled_tools": disabled_tools,
            "allow_git": allow_git,
            "disable_compaction": disable_compaction,
            "release_repo": release_repo,
            "release_version": release_version,
            "release_sha256": release_sha256,
            "install_ripgrep": install_ripgrep,
            "provider_timeout_ms": provider_timeout_ms,
            "max_turns": max_turns,
        }
        if system_prompt is not _UNSET:
            config_data["system_prompt"] = system_prompt
        config = OpenCodeConfig.from_config(config, **config_data)
        if system_prompt is None:
            config.system_prompt = None
        sandbox_config: bool | Mapping[str, object] | SandboxConfig
        sandbox_config = (
            config.sandbox if sandbox is None and config.sandbox is not None else True
        )
        if sandbox is not None:
            sandbox_config = sandbox
        files: dict[str, object] = {
            config.instruction_path: task_instruction_text,
        }
        if config.system_prompt is not None:
            files[config.system_prompt_path] = state_system_prompt_text
        artifacts = {
            "opencode_log": {
                "path": config.log_path,
                "format": "text",
                "optional": True,
            }
        }
        system_prompt_disabled = config.system_prompt is None
        super().__init__(
            command=[
                "bash",
                "-lc",
                build_opencode_run_script(
                    agent_workdir=config.agent_workdir,
                    instruction_path=config.instruction_path,
                    log_path=config.log_path,
                    allow_git=config.allow_git,
                ),
            ],
            sandbox=sandbox_config,
            files=files,
            setup=build_install_script(
                release_repo=config.release_repo,
                release_version=config.release_version,
                release_sha256=config.release_sha256,
                install_ripgrep=config.install_ripgrep,
            ),
            tools={
                "mcp": build_opencode_mcp_setup_script(
                    agent_workdir=config.agent_workdir,
                    system_prompt_path=config.system_prompt_path
                    if config.system_prompt is not None
                    else None,
                    log_path=config.log_path,
                    disabled_tools=config.disabled_tools,
                    disable_compaction=config.disable_compaction,
                    provider_timeout_ms=config.provider_timeout_ms,
                )
            },
            artifacts=artifacts,
            program=program,
            system_prompt=config.system_prompt,
            max_turns=config.max_turns,
            config=config.model_dump(exclude_none=False),
            **kwargs,
        )
        if system_prompt_disabled:
            self.config.system_prompt = None
            self.system_prompt = None


def build_install_script(
    release_repo: str = DEFAULT_RELEASE_REPO,
    release_version: str = DEFAULT_RELEASE_VERSION,
    release_sha256: str = DEFAULT_RELEASE_SHA256,
    install_ripgrep: bool = True,
) -> str:
    rg_install = (
        "apt-get -o Acquire::Retries=3 install -y -qq ripgrep > /dev/null 2>&1 || true"
        if install_ripgrep
        else ""
    )
    sha256_check = f'echo "{release_sha256}  /tmp/opencode.tar.gz" | sha256sum -c -'
    # Acquire::Retries=3 mitigates transient archive.ubuntu.com CDN sync
    # mismatches that fail fresh-sandbox apt-get calls mid-rollout.
    return f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl tar ca-certificates > /dev/null 2>&1
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
if [ -x "$HOME/.opencode/bin/opencode" ]; then
  echo "OpenCode already installed, skipping download"
else
  curl -fsSL "$OPENCODE_RELEASE_URL" -o /tmp/opencode.tar.gz
  {sha256_check}
  tar -xzf /tmp/opencode.tar.gz -C /tmp
  install -m 755 /tmp/opencode "$HOME/.opencode/bin/opencode"
  rm -f /tmp/opencode.tar.gz /tmp/opencode
fi
"""


def build_opencode_config(
    *,
    disabled_tools: list[str],
    system_prompt_path: str | None,
    disable_compaction: bool,
    provider_timeout_ms: int,
) -> str:
    agent_config: dict[str, object] = {
        "title": {"disable": True},
    }
    config: dict[str, object] = {
        "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
        "provider": {
            "intercepted": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Intercepted",
                "options": {
                    "baseURL": "$OPENAI_BASE_URL",
                    "apiKey": "${OPENAI_API_KEY:-intercepted}",
                    "timeout": provider_timeout_ms,
                },
                "models": {
                    "model": {
                        "name": "Intercepted Model",
                        "modalities": {"input": ["text"], "output": ["text"]},
                    }
                },
            }
        },
        "model": "intercepted/model",
        # Keep the small-model pin to avoid falling back to the default small
        # model and hitting rate limits; disable title calls below.
        "small_model": "intercepted/model",
        "agent": agent_config,
        "mcp": {
            "verifiers-tools": {
                "type": "local",
                "command": proxy_command(),
                "enabled": True,
            }
        },
    }
    if disable_compaction:
        config["compaction"] = {"auto": False, "prune": False}
    build_config: dict[str, object] = {}
    if system_prompt_path is not None:
        build_config["prompt"] = "{file:" + system_prompt_path + "}"
    if disabled_tools:
        build_config["tools"] = {tool: False for tool in disabled_tools}
    if build_config:
        agent_config["build"] = build_config
    return json.dumps(config, indent=2)


def build_opencode_run_script(
    *,
    agent_workdir: str,
    instruction_path: str,
    log_path: str,
    allow_git: bool,
) -> str:
    script = f"""\
set -eo pipefail
export PATH="$HOME/.opencode/bin:$PATH"
export OPENCODE_DISABLE_FILETIME_CHECK=true
export ALLOW_GIT={"1" if allow_git else "0"}

OPENCODE_WORKDIR="${{AGENT_WORKDIR:-}}"
if [[ -z "$OPENCODE_WORKDIR" ]]; then
    OPENCODE_WORKDIR={shlex.quote(agent_workdir)}
fi

cd "$OPENCODE_WORKDIR"
cat {shlex.quote(instruction_path)} | opencode run 2>&1 | tee {shlex.quote(log_path)}
"""
    return script


def build_opencode_mcp_setup_script(
    *,
    agent_workdir: str,
    system_prompt_path: str | None,
    log_path: str,
    disabled_tools: list[str],
    disable_compaction: bool,
    provider_timeout_ms: int,
) -> str:
    config_json = build_opencode_config(
        disabled_tools=disabled_tools,
        system_prompt_path=system_prompt_path,
        disable_compaction=disable_compaction,
        provider_timeout_ms=provider_timeout_ms,
    )
    log_dir = str(PurePosixPath(log_path).parent)
    return f"""\
set -e
export PATH="$HOME/.opencode/bin:$PATH"

OPENCODE_WORKDIR="${{AGENT_WORKDIR:-}}"
if [[ -z "$OPENCODE_WORKDIR" ]]; then
    OPENCODE_WORKDIR={shlex.quote(agent_workdir)}
fi

mkdir -p ~/.config/opencode {shlex.quote(log_dir)} "$OPENCODE_WORKDIR"
SCHEMA_DOLLAR='$'
cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG
"""
