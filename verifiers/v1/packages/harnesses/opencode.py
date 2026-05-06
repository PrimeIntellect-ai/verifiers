from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, cast

from verifiers.types import MessageContent

from .cli import CLIHarness
from ...state import State
from ...task import Task
from ...utils.mcp_proxy_utils import proxy_command

DEFAULT_RELEASE_REPO = "PrimeIntellect-ai/opencode"
DEFAULT_RELEASE_VERSION = "1.1.63-rl2"
DEFAULT_RELEASE_SHA256 = (
    "47f4102796da50769e27d2c9ea6a9cf7941f76898390cb497278cab39c4b6ed4"
)
DEFAULT_AGENT_WORKDIR = "/app"
DEFAULT_INSTRUCTION_PATH = "/opencode/instruction.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/opencode/system.txt"
DEFAULT_LOG_PATH = "/logs/agent/opencode.txt"
DEFAULT_SYSTEM_PROMPT = """\
You are OpenCode, an interactive CLI tool that helps users with tasks.

Your output is displayed in a command line interface. Be concise and direct.
Use tools to complete tasks. Do not use shell commands or code comments as a
way to communicate with the user.
"""
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
    "plan_exit",
    "plan_enter",
    "lsp",
    "codesearch",
    "skill",
]


class OpenCode(CLIHarness):
    def __init__(
        self,
        *,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        instruction_path: str = DEFAULT_INSTRUCTION_PATH,
        system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
        log_path: str = DEFAULT_LOG_PATH,
        system_prompt: object | None = DEFAULT_SYSTEM_PROMPT,
        disabled_tools: list[str] | None = None,
        allow_git: bool = False,
        disable_compaction: bool = True,
        release_repo: str = DEFAULT_RELEASE_REPO,
        release_version: str = DEFAULT_RELEASE_VERSION,
        release_sha256: str = DEFAULT_RELEASE_SHA256,
        install_ripgrep: bool = True,
        provider_timeout_ms: int = 3_600_000,
        sandbox: bool | Mapping[str, object] = True,
        program: Mapping[str, object] | None = None,
        max_turns: int | None = 4,
        **kwargs: Any,
    ):
        disabled = (
            list(DEFAULT_DISABLED_TOOLS)
            if disabled_tools is None
            else list(disabled_tools)
        )
        files: dict[str, object] = {
            instruction_path: task_instruction_text,
        }
        if system_prompt is not None:
            files[system_prompt_path] = state_system_prompt_text
        artifacts = {
            "opencode_log": {
                "path": log_path,
                "format": "text",
                "optional": True,
            }
        }
        super().__init__(
            command=[
                "bash",
                "-lc",
                build_opencode_run_script(
                    agent_workdir=agent_workdir,
                    instruction_path=instruction_path,
                    log_path=log_path,
                    allow_git=allow_git,
                ),
            ],
            sandbox=sandbox,
            files=files,
            setup=build_install_script(
                release_repo=release_repo,
                release_version=release_version,
                release_sha256=release_sha256,
                install_ripgrep=install_ripgrep,
            ),
            tools={
                "mcp": build_opencode_mcp_setup_script(
                    agent_workdir=agent_workdir,
                    system_prompt_path=system_prompt_path
                    if system_prompt is not None
                    else None,
                    log_path=log_path,
                    disabled_tools=disabled,
                    disable_compaction=disable_compaction,
                    provider_timeout_ms=provider_timeout_ms,
                )
            },
            artifacts=artifacts,
            program=program,
            system_prompt=system_prompt,
            max_turns=max_turns,
            **kwargs,
        )


def build_install_script(
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
    sha256_check = f'echo "{release_sha256}  /tmp/opencode.tar.gz" | sha256sum -c -'
    return f"""\
set -e
apt-get update -qq && apt-get install -y -qq curl tar ca-certificates > /dev/null 2>&1
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
{sha256_check}
tar -xzf /tmp/opencode.tar.gz -C /tmp
install -m 755 /tmp/opencode "$HOME/.opencode/bin/opencode"
"""


def build_opencode_config(
    *,
    disabled_tools: list[str],
    system_prompt_path: str | None,
    disable_compaction: bool,
    provider_timeout_ms: int,
) -> str:
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
        config["agent"] = {"build": build_config}
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


def task_instruction_text(task: Task, state: State) -> str:
    _ = state
    instruction = task.get("instruction")
    if isinstance(instruction, str):
        return instruction
    return messages_text(task.get("prompt", []))


def state_system_prompt_text(task: Task, state: State) -> str:
    _ = task
    return messages_text(state.get("system_prompt", []))


def messages_text(messages: object) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return str(messages or "")
    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if content is not None:
            parts.append(content_text(content))
        elif isinstance(message, Mapping):
            item = cast(Mapping[str, object], message)
            parts.append(content_text(item.get("content")))
        else:
            parts.append(str(message))
    return "\n\n".join(part for part in parts if part)


def content_text(content: MessageContent | object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                item = cast(Mapping[str, object], part)
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content)
