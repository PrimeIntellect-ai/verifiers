"""OpenCode harness configuration.

Provides install script, config generation, and run command templates shared
across OpenCode-based environments.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

DEFAULT_RELEASE_REPO = "rasdani/opencode"
DEFAULT_RELEASE_VERSION = "1.1.63-swe8"
DEFAULT_RELEASE_SHA256 = (
    "b34504f10b0aeab22537259a9ceda8dc7973527dfb37a94ddf2bcf4b5ba15dac"
)
OPENCODE_CLI_PACKAGE = DEFAULT_RELEASE_REPO
OPENCODE_CLI_VERSION = DEFAULT_RELEASE_VERSION
OPENCODE_CLI_SHA256 = DEFAULT_RELEASE_SHA256
DEFAULT_SYSTEM_PROMPT = (Path(__file__).parent / "prompt.txt").read_text()
DEFAULT_SKILLS_PATH = "/task/skills"

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


def build_install_script(
    release_repo: str = DEFAULT_RELEASE_REPO,
    release_version: str = DEFAULT_RELEASE_VERSION,
    release_sha256: str = DEFAULT_RELEASE_SHA256,
    install_ripgrep: bool = True,
) -> str:
    """Build the shell script that installs OpenCode in a sandbox."""
    # OpenCode searches its own bin dir first, so the installer keeps the pinned
    # release isolated under the user's home instead of relying on system paths.
    rg_install = ""
    if install_ripgrep:
        rg_install = "apt-get install -y -qq ripgrep > /dev/null 2>&1 || true"

    return f"""\
set -e
apt-get update -qq && apt-get install -y -qq curl tar > /dev/null 2>&1
{rg_install}

OPENCODE_RELEASE_REPO="{release_repo}"
OPENCODE_RELEASE_VERSION="{release_version}"

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
echo "{release_sha256}  /tmp/opencode.tar.gz" | sha256sum -c -
tar -xzf /tmp/opencode.tar.gz -C /tmp
install -m 755 /tmp/opencode "$HOME/.opencode/bin/opencode"
echo "OpenCode installed successfully"
"""


def build_opencode_config(
    disabled_tools: list[str] | None = None,
    system_prompt_path: str | None = None,
    disable_compaction: bool = True,
    provider_key: str = "${OPENAI_MODEL%%/*}",
    provider_display_name: str | None = None,
    model_id: str = "$OPENAI_MODEL",
    model_key: str = "${OPENAI_MODEL##*/}",
    model_display_name: str | None = None,
    provider_timeout_ms: int = 3_600_000,
    mcp_servers: list[dict[str, Any] | str] | None = None,
) -> str:
    """Generate opencode.json config content."""
    # The config targets the evaluator's OpenAI-compatible proxy, while leaving
    # provider/model keys shell-expandable so each rollout can set them at runtime.
    config: dict = {
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

    agent_build: dict = {}
    if system_prompt_path:
        agent_build["prompt"] = "{file:" + system_prompt_path + "}"
    if disabled_tools:
        agent_build["tools"] = {tool: False for tool in disabled_tools}
    if agent_build:
        config["agent"] = {"build": agent_build}
    if mcp_servers:
        config["mcp"] = json.loads(build_opencode_mcp_config(mcp_servers))

    return json.dumps(config, indent=2)


def build_opencode_mcp_config(
    mcp_servers: list[dict[str, Any] | str] | None = None,
) -> str:
    """Render shared MCP server dictionaries as OpenCode ``mcp`` JSON."""
    servers: dict[str, dict[str, Any]] = {}
    for server in mcp_servers or []:
        if isinstance(server, str):
            server = {"name": server, "transport": "stdio", "command": server}

        name = str(server["name"])
        transport = str(server.get("transport") or "stdio")
        if server.get("url") and transport != "stdio":
            entry: dict[str, Any] = {"type": "remote", "url": server["url"]}
            headers = server.get("headers") or server.get("http_headers")
            if isinstance(headers, dict) and headers:
                entry["headers"] = headers
            if server.get("oauth") is not None:
                entry["oauth"] = server["oauth"]
        else:
            command = server["command"]
            if isinstance(command, str):
                command = shlex.split(command)
            entry = {
                "type": "local",
                "command": [*command, *(server.get("args") or [])],
            }
            env = server.get("environment") or server.get("env")
            if isinstance(env, dict) and env:
                entry["environment"] = env

        if server.get("enabled") is not None:
            entry["enabled"] = server["enabled"]
        if server.get("timeout") is not None:
            entry["timeout"] = server["timeout"]
        elif server.get("connectionTimeoutMs") is not None:
            entry["timeout"] = server["connectionTimeoutMs"]
        servers[name] = entry

    return json.dumps(servers, indent=2)


def build_opencode_run_command(
    agent_workdir: str = "/app",
    prompt_path: str = "/opencode/prompt.txt",
    log_path: str = "/opencode/logs.txt",
    disabled_tools: list[str] | None = None,
    system_prompt_path: str | None = None,
    disable_compaction: bool = True,
    allow_git: bool = False,
    provider_key: str = "${OPENAI_MODEL%%/*}",
    provider_display_name: str | None = None,
    model_id: str = "$OPENAI_MODEL",
    model_key: str = "${OPENAI_MODEL##*/}",
    model_display_name: str | None = None,
    provider_timeout_ms: int = 3_600_000,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    skills_dir: str | None = None,
) -> str:
    """Build the shell command that configures and runs OpenCode."""
    # The generated command writes opencode.json inside the sandbox immediately
    # before execution so env vars like OPENAI_MODEL expand in the target shell.
    effective_disabled_tools = disabled_tools
    if skills_dir and disabled_tools:
        effective_disabled_tools = [tool for tool in disabled_tools if tool != "skill"]

    config_json = build_opencode_config(
        disabled_tools=effective_disabled_tools,
        system_prompt_path=system_prompt_path,
        disable_compaction=disable_compaction,
        provider_key=provider_key,
        provider_display_name=provider_display_name,
        model_id=model_id,
        model_key=model_key,
        model_display_name=model_display_name,
        provider_timeout_ms=provider_timeout_ms,
        mcp_servers=mcp_servers,
    )

    skills_block = ""
    if skills_dir:
        quoted_skills_dir = shlex.quote(skills_dir)
        skills_block = f"""\
mkdir -p ~/.config/opencode/skills "$HOME/.agents/skills"
cp -r {quoted_skills_dir}/* ~/.config/opencode/skills/ 2>/dev/null || true
cp -r {quoted_skills_dir}/* "$HOME/.agents/skills/" 2>/dev/null || true
"""

    script = f"""\
set -eo pipefail

export PATH="$HOME/.opencode/bin:$PATH"
export OPENCODE_DISABLE_FILETIME_CHECK=true
export ALLOW_GIT={"1" if allow_git else "0"}

mkdir -p ~/.config/opencode /logs/agent {agent_workdir}

SCHEMA_DOLLAR='$'

cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG
{skills_block}

cd {agent_workdir}
cat {prompt_path} | opencode run 2>&1 | tee {log_path}
"""
    return f"bash -lc {shlex.quote(script)}"


OPENCODE_INSTALL_SCRIPT = build_install_script()
OPENCODE_CONFIG = {
    "install_script": OPENCODE_INSTALL_SCRIPT,
    "cli_package": OPENCODE_CLI_PACKAGE,
    "cli_version": OPENCODE_CLI_VERSION,
    "cli_sha256": OPENCODE_CLI_SHA256,
}


def opencode_harness(
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    task_system_prompt: str | None = None,
    disabled_tools: list[str] | None = None,
    agent_workdir: str = "/app",
    allow_git: bool = False,
    disable_compaction: bool = True,
    release_repo: str = DEFAULT_RELEASE_REPO,
    release_version: str = DEFAULT_RELEASE_VERSION,
    release_sha256: str = DEFAULT_RELEASE_SHA256,
    instruction_path: str = "/opencode/prompt.txt",
    system_prompt_path: str = "/opencode/system.txt",
    log_path: str = "/opencode/logs.txt",
    provider_key: str = "${OPENAI_MODEL%%/*}",
    provider_display_name: str | None = None,
    model_id: str = "$OPENAI_MODEL",
    model_key: str = "${OPENAI_MODEL##*/}",
    model_display_name: str | None = None,
    provider_timeout_ms: int = 3_600_000,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    skills_dir: str | None = None,
):
    """Create a Harness configured for OpenCode."""
    from harnesses.base import make_native_harness

    # Task-specific system text is appended without changing the base prompt file
    # path that ComposableEnv writes into the sandbox.
    if task_system_prompt:
        prompts = [prompt for prompt in (system_prompt, task_system_prompt) if prompt]
        system_prompt = "\n".join(prompts)

    install_script = build_install_script(
        release_repo=release_repo,
        release_version=release_version,
        release_sha256=release_sha256,
    )

    return make_native_harness(
        build_run_command=build_opencode_run_command,
        run_kwargs={
            "agent_workdir": agent_workdir,
            "prompt_path": instruction_path,
            "log_path": log_path,
            "disabled_tools": disabled_tools,
            "system_prompt_path": system_prompt_path if system_prompt else None,
            "disable_compaction": disable_compaction,
            "allow_git": allow_git,
            "provider_key": provider_key,
            "provider_display_name": provider_display_name,
            "model_id": model_id,
            "model_key": model_key,
            "model_display_name": model_display_name,
            "provider_timeout_ms": provider_timeout_ms,
        },
        install_script=install_script,
        system_prompt=system_prompt,
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_path=log_path,
        default_skills_path=DEFAULT_SKILLS_PATH,
        mcp_servers=mcp_servers,
        skills_dir=skills_dir,
    )
