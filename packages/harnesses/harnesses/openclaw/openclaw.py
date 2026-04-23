"""OpenClaw harness configuration.

The generated Harness installs the upstream ``openclaw`` CLI, writes a
rollout-local ``openclaw.json`` for the evaluator's OpenAI-compatible endpoint,
and runs ``openclaw agent --local`` against the sandbox task prompt.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from harnesses._integrity import (
    NVM_INSTALL_SHA256,
    NVM_INSTALL_VERSION,
    build_verified_npm_install_command,
    build_verified_nvm_install_command,
    indent_shell_block,
)

OPENCLAW_CLI_PACKAGE = "openclaw"
OPENCLAW_CLI_VERSION = "2026.4.11"
OPENCLAW_CLI_SHA256 = "95fdabf7a4cfdd2cff5490eff3a237efc63f95ba9b971c9eba40e067115bc0e6"
DEFAULT_PACKAGE_VERSION = OPENCLAW_CLI_VERSION
DEFAULT_PACKAGE_SHA256 = OPENCLAW_CLI_SHA256
DEFAULT_INSTRUCTION_PATH = "/openclaw/prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/openclaw/system.txt"
DEFAULT_LOG_DIR = "/logs/agent"
DEFAULT_LOG_PATH = f"{DEFAULT_LOG_DIR}/openclaw.txt"
DEFAULT_STATE_DIR = f"{DEFAULT_LOG_DIR}/openclaw-state"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_AGENT_ID = "main"
DEFAULT_SKILLS_PATH = "/task/skills"
DEFAULT_PROVIDER_NAME = "intercepted"
DEFAULT_MODEL_ID = "$OPENAI_MODEL"
DEFAULT_API = "openai-completions"
DEFAULT_THINKING = "off"
DEFAULT_COMPAT = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
}


def build_openclaw_install_script(
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
) -> str:
    """Build the shell script that installs OpenClaw in a sandbox."""
    # OpenClaw is distributed as the `openclaw` npm package. Non-Alpine images
    # get nvm so the CLI has a recent enough Node 22 runtime.
    npm_install = indent_shell_block(
        build_verified_npm_install_command(
            OPENCLAW_CLI_PACKAGE,
            package_version,
            package_sha256,
            "OPENCLAW_NPM",
        )
    )
    nvm_install = indent_shell_block(build_verified_nvm_install_command())

    return f"""\
set -e

if command -v apk >/dev/null 2>&1; then
  apk add --no-cache curl bash nodejs npm ca-certificates
{npm_install}
else
  if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl bash ca-certificates > /dev/null 2>&1
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y curl bash ca-certificates
  elif command -v yum >/dev/null 2>&1; then
    yum install -y curl bash ca-certificates
  elif command -v pacman >/dev/null 2>&1; then
    pacman -Sy --noconfirm curl bash ca-certificates
  fi

{nvm_install}
  export NVM_DIR="$HOME/.nvm"
  . "$NVM_DIR/nvm.sh"
  nvm install 22
  nvm alias default 22
{npm_install}
fi

if [ -s "$HOME/.nvm/nvm.sh" ]; then
  . "$HOME/.nvm/nvm.sh"
fi

for bin in node npm openclaw; do
  BIN_PATH="$(command -v "$bin" 2>/dev/null || true)"
  if [ -n "$BIN_PATH" ]; then
    ln -sf "$BIN_PATH" "/usr/local/bin/$bin" 2>/dev/null || true
  fi
done

openclaw --version
"""


def build_openclaw_mcp_config(
    mcp_servers: list[dict[str, Any] | str] | None = None,
) -> str:
    """Render shared MCP server dictionaries as OpenClaw ``mcp.servers`` JSON.

    Stdio entries keep command/args/env fields; URL entries become
    streamable-http or sse server definitions.
    """
    # The public config shape is mcp.servers.<name>. This helper accepts the
    # same compact server dictionaries used by the other harnesses.
    servers: dict[str, dict[str, Any]] = {}
    for server in mcp_servers or []:
        if isinstance(server, str):
            server = {"name": server, "command": server}

        name = str(server["name"])
        transport = str(server.get("transport") or "stdio")
        if server.get("url") and transport != "stdio":
            entry: dict[str, Any] = {
                "url": server["url"],
                "transport": "streamable-http" if transport == "http" else transport,
            }
            headers = server.get("headers") or server.get("http_headers")
            if isinstance(headers, dict) and headers:
                entry["headers"] = headers
        else:
            entry = {"command": server["command"]}
            if server.get("args"):
                entry["args"] = server["args"]
            env = server.get("env")
            if isinstance(env, dict) and env:
                entry["env"] = env
            cwd = server.get("cwd") or server.get("workingDirectory")
            if cwd:
                entry["cwd"] = cwd

        if server.get("connectionTimeoutMs") is not None:
            entry["connectionTimeoutMs"] = server["connectionTimeoutMs"]
        servers[name] = entry

    return json.dumps({"servers": servers}, indent=2)


def build_openclaw_config(
    provider_name: str = DEFAULT_PROVIDER_NAME,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "$OPENAI_BASE_URL",
    api_key: str = "OPENAI_API_KEY",
    api: str = DEFAULT_API,
    reasoning: bool = False,
    input_modalities: list[str] | None = None,
    context_window: int = 128_000,
    max_tokens: int = 16_384,
    compat: dict[str, Any] | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
) -> str:
    """Render ``openclaw.json`` for the intercepted OpenAI-compatible model.

    The default values intentionally contain shell placeholders; the run command
    writes this JSON with an expanding heredoc so each rollout can supply its own
    ``OPENAI_BASE_URL`` and ``OPENAI_MODEL``.
    """
    model: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "api": api,
        "reasoning": reasoning,
        "input": input_modalities or ["text", "image"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": context_window,
        "maxTokens": max_tokens,
        "compat": compat if compat is not None else DEFAULT_COMPAT,
    }
    config: dict[str, Any] = {
        "models": {
            "providers": {
                provider_name: {
                    "baseUrl": base_url,
                    "apiKey": api_key,
                    "api": api,
                    "models": [model],
                }
            }
        },
        "agents": {
            "defaults": {
                "model": {"primary": f"{provider_name}/{model_id}"},
                "workspace": "$OPENCLAW_AGENT_WORKDIR",
                "skipBootstrap": True,
            }
        },
    }
    if mcp_servers:
        config["mcp"] = json.loads(build_openclaw_mcp_config(mcp_servers))

    return json.dumps(config, indent=2)


def build_openclaw_run_command(
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    state_dir: str = DEFAULT_STATE_DIR,
    agent_id: str = DEFAULT_AGENT_ID,
    provider_name: str = DEFAULT_PROVIDER_NAME,
    model_id: str = DEFAULT_MODEL_ID,
    api: str = DEFAULT_API,
    reasoning: bool = False,
    thinking: str | None = DEFAULT_THINKING,
    timeout_seconds: int | None = None,
    json_output: bool = True,
    skills_dir: str | None = None,
    append_system_prompt: str | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Build the shell command that configures and runs OpenClaw locally.

    The command writes ``openclaw.json``, copies optional skills into the
    workspace, assembles CLI flags as a bash array, and runs one local agent turn
    with JSON output by default.
    """
    if agent_workdir == DEFAULT_AGENT_WORKDIR:
        workdir_assignment = f"OPENCLAW_AGENT_WORKDIR={DEFAULT_AGENT_WORKDIR}"
    else:
        workdir_assignment = f"OPENCLAW_AGENT_WORKDIR={shlex.quote(agent_workdir)}"

    if model_id == DEFAULT_MODEL_ID:
        model_assignment = 'OPENCLAW_MODEL="$OPENAI_MODEL"'
    else:
        model_assignment = f"OPENCLAW_MODEL={shlex.quote(model_id)}"

    config = build_openclaw_config(
        provider_name=provider_name,
        model_id="$OPENCLAW_MODEL",
        api=api,
        reasoning=reasoning,
        mcp_servers=mcp_servers,
    )

    arg_lines = [
        "OPENCLAW_ARGS+=(agent)",
        "OPENCLAW_ARGS+=(--local)",
        f"OPENCLAW_ARGS+=(--agent {shlex.quote(agent_id)})",
        'OPENCLAW_ARGS+=(--message "$OPENCLAW_PROMPT")',
    ]
    if thinking:
        arg_lines.append(f"OPENCLAW_ARGS+=(--thinking {shlex.quote(thinking)})")
    if timeout_seconds is not None:
        arg_lines.append(f"OPENCLAW_ARGS+=(--timeout {timeout_seconds})")
    if json_output:
        arg_lines.append("OPENCLAW_ARGS+=(--json)")
    for arg in extra_args or []:
        arg_lines.append(f"OPENCLAW_ARGS+=({shlex.quote(arg)})")

    skills_block = ""
    if skills_dir:
        # OpenClaw discovers workspace skills from <workspace>/skills. Mirroring
        # to ~/.agents/skills also supports shared skill loaders used by agents.
        quoted_skills_dir = shlex.quote(skills_dir)
        skills_block = f"""\
mkdir -p "$OPENCLAW_AGENT_WORKDIR/skills" "$HOME/.agents/skills"
cp -r {quoted_skills_dir}/* "$OPENCLAW_AGENT_WORKDIR/skills/" 2>/dev/null || true
cp -r {quoted_skills_dir}/* "$HOME/.agents/skills/" 2>/dev/null || true
"""

    append_system_prompt_block = ""
    if append_system_prompt:
        append_system_prompt_block = (
            f"OPENCLAW_SYSTEM_PROMPT+=$'\\n\\n'{shlex.quote(append_system_prompt)}\n"
        )

    joined_arg_lines = "\n".join(arg_lines)
    script = f"""\
set -eo pipefail

if [ -s "$HOME/.nvm/nvm.sh" ]; then
  . "$HOME/.nvm/nvm.sh"
fi
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export OPENCLAW_STATE_DIR={shlex.quote(state_dir)}
export OPENCLAW_CONFIG_PATH="$OPENCLAW_STATE_DIR/openclaw.json"
export OPENCLAW_AGENT_DIR="$OPENCLAW_STATE_DIR/agents/{shlex.quote(agent_id)}/agent"
export PI_CODING_AGENT_DIR="$OPENCLAW_AGENT_DIR"
export OPENCLAW_SKIP_CHANNELS=1
{workdir_assignment}
{model_assignment}
mkdir -p \\
  {shlex.quote(log_path.rsplit("/", 1)[0])} \\
  "$OPENCLAW_STATE_DIR" \\
  "$OPENCLAW_AGENT_DIR" \\
  "$OPENCLAW_AGENT_WORKDIR"

cat > "$OPENCLAW_CONFIG_PATH" <<OPENCLAW_CONFIG_JSON
{config}
OPENCLAW_CONFIG_JSON
{skills_block}
OPENCLAW_PROMPT="$(cat {shlex.quote(instruction_path)})"
OPENCLAW_SYSTEM_PROMPT=""
if [ -s {shlex.quote(system_prompt_path)} ]; then
  OPENCLAW_SYSTEM_PROMPT="$(cat {shlex.quote(system_prompt_path)})"
fi
{append_system_prompt_block}
if [ -n "$OPENCLAW_SYSTEM_PROMPT" ]; then
  OPENCLAW_PROMPT="$OPENCLAW_SYSTEM_PROMPT

$OPENCLAW_PROMPT"
fi

OPENCLAW_ARGS=()
{joined_arg_lines}
cd "$OPENCLAW_AGENT_WORKDIR"
openclaw "${{OPENCLAW_ARGS[@]}}" 2>&1 </dev/null | tee -a {shlex.quote(log_path)}
"""
    return f"bash -lc {shlex.quote(script)}"


OPENCLAW_INSTALL_SCRIPT = build_openclaw_install_script()
OPENCLAW_CONFIG = {
    "install_script": OPENCLAW_INSTALL_SCRIPT,
    "cli_package": OPENCLAW_CLI_PACKAGE,
    "cli_version": OPENCLAW_CLI_VERSION,
    "cli_sha256": OPENCLAW_CLI_SHA256,
    "node_installer_version": NVM_INSTALL_VERSION,
    "node_installer_sha256": NVM_INSTALL_SHA256,
}


def openclaw_harness(
    system_prompt: str | None = None,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    state_dir: str = DEFAULT_STATE_DIR,
    agent_id: str = DEFAULT_AGENT_ID,
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    provider_name: str = DEFAULT_PROVIDER_NAME,
    model_id: str = DEFAULT_MODEL_ID,
    api: str = DEFAULT_API,
    reasoning: bool = False,
    thinking: str | None = DEFAULT_THINKING,
    timeout_seconds: int | None = None,
    json_output: bool = True,
    skills_dir: str | None = None,
    append_system_prompt: str | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    extra_args: list[str] | None = None,
):
    """Create a Harness configured for OpenClaw's local agent CLI."""
    from harnesses.base import make_native_harness

    install_script = build_openclaw_install_script(
        package_version=package_version,
        package_sha256=package_sha256,
    )

    # ComposableEnv writes prompt files; this factory wires OpenClaw's package
    # install, isolated state dir, model config, and local command around them.
    return make_native_harness(
        build_run_command=build_openclaw_run_command,
        run_kwargs={
            "agent_workdir": agent_workdir,
            "instruction_path": instruction_path,
            "system_prompt_path": system_prompt_path,
            "log_path": log_path,
            "state_dir": state_dir,
            "agent_id": agent_id,
            "provider_name": provider_name,
            "model_id": model_id,
            "api": api,
            "reasoning": reasoning,
            "thinking": thinking,
            "timeout_seconds": timeout_seconds,
            "json_output": json_output,
            "append_system_prompt": append_system_prompt,
            "extra_args": extra_args,
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
