"""Claude Code harness configuration.

The generated Harness uses a rollout-local CLAUDE_CONFIG_DIR, maps
Anthropic/OpenAI model env vars onto Claude Code's expectations, and runs the
CLI non-interactively against the sandbox task prompt.
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

CLAUDE_CODE_CLI_PACKAGE = "@anthropic-ai/claude-code"
CLAUDE_CODE_CLI_VERSION = "2.1.104"
CLAUDE_CODE_CLI_SHA256 = (
    "c94154dadeb8e95fecabf255c1f08f0be2085b2731dc1faafa08c271c48fd2f7"
)
DEFAULT_PACKAGE_VERSION = CLAUDE_CODE_CLI_VERSION
DEFAULT_PACKAGE_SHA256 = CLAUDE_CODE_CLI_SHA256
DEFAULT_INSTRUCTION_PATH = "/claude-code/prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/claude-code/system.txt"
DEFAULT_LOG_DIR = "/logs/agent"
DEFAULT_LOG_PATH = f"{DEFAULT_LOG_DIR}/claude-code.txt"
DEFAULT_CONFIG_DIR = f"{DEFAULT_LOG_DIR}/claude-code-sessions"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_OUTPUT_FORMAT = "stream-json"
DEFAULT_PERMISSION_MODE = "bypassPermissions"
DEFAULT_SKILLS_PATH = "/task/skills"


def build_claude_code_install_script(
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
) -> str:
    """Build the shell script that installs Claude Code in a sandbox."""
    # Use the official npm package in both Alpine and glibc images; nvm gives
    # non-Alpine images a predictable modern Node version.
    npm_install = indent_shell_block(
        build_verified_npm_install_command(
            CLAUDE_CODE_CLI_PACKAGE,
            package_version,
            package_sha256,
            "CLAUDE_CODE_NPM",
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

for bin in node npm claude; do
  BIN_PATH="$(command -v "$bin" 2>/dev/null || true)"
  if [ -n "$BIN_PATH" ]; then
    ln -sf "$BIN_PATH" "/usr/local/bin/$bin" 2>/dev/null || true
  fi
done

claude --version
"""


def build_claude_code_mcp_config(
    mcp_servers: list[dict[str, Any] | str] | None = None,
) -> str:
    """Render Claude Code MCP server configuration as .claude.json content.

    Stdio entries become command configs, URL entries become remote configs, and
    streamable-http is mapped to Claude Code's http transport name.
    """
    # Claude Code stores MCP servers in .claude.json. The harness accepts the
    # shared server shape used across harnesses and adapts transport names here.
    servers: dict[str, dict[str, Any]] = {}
    for server in mcp_servers or []:
        if isinstance(server, str):
            server = {"name": server, "transport": "stdio", "command": server}

        name = str(server["name"])
        transport = str(server.get("transport") or "stdio")
        if transport == "stdio":
            entry: dict[str, Any] = {
                "type": "stdio",
                "command": server["command"],
            }
            if server.get("args"):
                entry["args"] = server["args"]
            env = server.get("env")
            if isinstance(env, dict) and env:
                entry["env"] = env
        else:
            entry = {
                "type": "http" if transport == "streamable-http" else transport,
                "url": server["url"],
            }
            headers = server.get("headers") or server.get("http_headers")
            if isinstance(headers, dict) and headers:
                entry["headers"] = headers
        servers[name] = entry

    return json.dumps({"mcpServers": servers}, indent=2)


def build_claude_code_run_command(
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    config_dir: str = DEFAULT_CONFIG_DIR,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    permission_mode: str | None = DEFAULT_PERMISSION_MODE,
    max_turns: int | None = None,
    reasoning_effort: str | None = None,
    max_budget_usd: str | None = None,
    fallback_model: str | None = None,
    append_system_prompt: str | None = None,
    allowed_tools: str | list[str] | None = None,
    disallowed_tools: str | list[str] | None = None,
    max_thinking_tokens: int | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    skills_dir: str | None = None,
    memory_dir: str | None = None,
) -> str:
    """Build the shell command that configures Claude Code for one rollout.

    The command resolves model/env vars, prepares CLAUDE_CONFIG_DIR, copies
    skills and memory, writes MCP config, appends system prompt text, assembles
    CLI flags, and runs `claude --print`.
    """
    # Preserve AGENT_WORKDIR as a runtime shell expansion for the common default,
    # but quote custom paths when the caller pins a concrete directory.
    if agent_workdir == DEFAULT_AGENT_WORKDIR:
        workdir_assignment = f"CLAUDE_CODE_AGENT_WORKDIR={DEFAULT_AGENT_WORKDIR}"
    else:
        workdir_assignment = f"CLAUDE_CODE_AGENT_WORKDIR={shlex.quote(agent_workdir)}"

    arg_lines = [
        f"CLAUDE_ARGS+=(--output-format={shlex.quote(output_format)})",
        "CLAUDE_ARGS+=(--print)",
        "CLAUDE_ARGS+=(--verbose)",
    ]
    # Build a bash array instead of a single string so prompts, tools, and model
    # names can contain spaces without changing argument boundaries.
    if permission_mode:
        arg_lines.append(
            f"CLAUDE_ARGS+=(--permission-mode {shlex.quote(permission_mode)})"
        )
    if max_turns is not None:
        arg_lines.append(f"CLAUDE_ARGS+=(--max-turns {max_turns})")
    if reasoning_effort:
        arg_lines.append(f"CLAUDE_ARGS+=(--effort {shlex.quote(reasoning_effort)})")
    if max_budget_usd:
        arg_lines.append(
            f"CLAUDE_ARGS+=(--max-budget-usd {shlex.quote(max_budget_usd)})"
        )
    if fallback_model:
        arg_lines.append(
            f"CLAUDE_ARGS+=(--fallback-model {shlex.quote(fallback_model)})"
        )

    for value in (
        [allowed_tools] if isinstance(allowed_tools, str) else allowed_tools or []
    ):
        arg_lines.append(f"CLAUDE_ARGS+=(--allowedTools {shlex.quote(value)})")
    for value in (
        [disallowed_tools]
        if isinstance(disallowed_tools, str)
        else disallowed_tools or []
    ):
        arg_lines.append(f"CLAUDE_ARGS+=(--disallowedTools {shlex.quote(value)})")

    mcp_config = build_claude_code_mcp_config(mcp_servers)
    mcp_block = ""
    if mcp_servers:
        # Claude Code discovers project-scoped MCP servers from the config dir
        # used by this rollout.
        mcp_block = f"""\
cat > "$CLAUDE_CONFIG_DIR/.claude.json" <<'CLAUDE_CODE_MCP_CONFIG'
{mcp_config}
CLAUDE_CODE_MCP_CONFIG
"""

    skills_block = ""
    if skills_dir:
        # Skills are copied into Claude's config dir so its built-in discovery can
        # surface them during non-interactive execution.
        quoted_skills_dir = shlex.quote(skills_dir)
        skills_block = (
            f"cp -r {quoted_skills_dir}/* "
            '"$CLAUDE_CONFIG_DIR/skills/" 2>/dev/null || true\n'
        )

    memory_block = ""
    if memory_dir:
        # Pre-seeded memory lets callers provide project context without baking it
        # into the user prompt.
        quoted_memory_dir = shlex.quote(memory_dir)
        memory_block = f"""\
mkdir -p "$CLAUDE_CONFIG_DIR/projects/-app/memory"
cp -r {quoted_memory_dir}/* "$CLAUDE_CONFIG_DIR/projects/-app/memory/" 2>/dev/null || true
"""

    thinking_tokens_line = ""
    if max_thinking_tokens is not None:
        thinking_tokens_line = (
            f"export MAX_THINKING_TOKENS={shlex.quote(str(max_thinking_tokens))}\n"
        )

    append_system_prompt_block = ""
    if append_system_prompt:
        append_system_prompt_block = (
            f"CLAUDE_CODE_SYSTEM_PROMPT+=$'\\n\\n'{shlex.quote(append_system_prompt)}\n"
        )

    joined_arg_lines = "\n".join(arg_lines)
    script = f"""\
set -eo pipefail

if [ -s "$HOME/.nvm/nvm.sh" ]; then
  . "$HOME/.nvm/nvm.sh"
fi
export PATH="$HOME/.local/bin:$PATH"

export ANTHROPIC_API_KEY="${{ANTHROPIC_API_KEY:-${{ANTHROPIC_AUTH_TOKEN:-intercepted}}}}"
if [ -n "${{OPENAI_BASE_URL:-}}" ] && [ -z "${{ANTHROPIC_BASE_URL:-}}" ]; then
  export ANTHROPIC_BASE_URL="$OPENAI_BASE_URL"
fi
CLAUDE_CODE_MODEL="${{ANTHROPIC_MODEL:-${{OPENAI_MODEL:-}}}}"
if [ -z "${{ANTHROPIC_BASE_URL:-}}" ]; then
  CLAUDE_CODE_MODEL="${{CLAUDE_CODE_MODEL##*/}}"
fi
if [ -n "$CLAUDE_CODE_MODEL" ]; then
  export ANTHROPIC_MODEL="$CLAUDE_CODE_MODEL"
fi
if [ -n "${{ANTHROPIC_BASE_URL:-}}" ] && [ -n "${{ANTHROPIC_MODEL:-}}" ]; then
  export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
  export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
  export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"
  export CLAUDE_CODE_SUBAGENT_MODEL="$ANTHROPIC_MODEL"
fi
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export FORCE_AUTO_BACKGROUND_TASKS=1
export ENABLE_BACKGROUND_TASKS=1
export IS_SANDBOX=1
{thinking_tokens_line}
export CLAUDE_CONFIG_DIR={shlex.quote(config_dir)}
{workdir_assignment}
mkdir -p \\
  {shlex.quote(log_path.rsplit("/", 1)[0])} \\
  "$CLAUDE_CONFIG_DIR/debug" \\
  "$CLAUDE_CONFIG_DIR/projects/-app" \\
  "$CLAUDE_CONFIG_DIR/shell-snapshots" \\
  "$CLAUDE_CONFIG_DIR/statsig" \\
  "$CLAUDE_CONFIG_DIR/todos" \\
  "$CLAUDE_CONFIG_DIR/skills" \\
  "$CLAUDE_CODE_AGENT_WORKDIR"

if [ -d "$HOME/.claude/skills" ]; then
  cp -r "$HOME/.claude/skills/." "$CLAUDE_CONFIG_DIR/skills/" 2>/dev/null || true
fi
{skills_block}{memory_block}{mcp_block}
CLAUDE_CODE_PROMPT="$(cat {shlex.quote(instruction_path)})"
CLAUDE_CODE_SYSTEM_PROMPT=""
if [ -s {shlex.quote(system_prompt_path)} ]; then
  CLAUDE_CODE_SYSTEM_PROMPT="$(cat {shlex.quote(system_prompt_path)})"
fi
{append_system_prompt_block}
CLAUDE_ARGS=()
{joined_arg_lines}
if [ -n "$CLAUDE_CODE_SYSTEM_PROMPT" ]; then
  CLAUDE_ARGS+=(--append-system-prompt "$CLAUDE_CODE_SYSTEM_PROMPT")
fi
if [ -n "${{ANTHROPIC_MODEL:-}}" ]; then
  CLAUDE_ARGS+=(--model "$ANTHROPIC_MODEL")
fi

cd "$CLAUDE_CODE_AGENT_WORKDIR"
claude "${{CLAUDE_ARGS[@]}}" -- "$CLAUDE_CODE_PROMPT" 2>&1 </dev/null | tee -a {shlex.quote(log_path)}
"""
    return f"bash -lc {shlex.quote(script)}"


CLAUDE_CODE_INSTALL_SCRIPT = build_claude_code_install_script()
CLAUDE_CODE_CONFIG = {
    "install_script": CLAUDE_CODE_INSTALL_SCRIPT,
    "cli_package": CLAUDE_CODE_CLI_PACKAGE,
    "cli_version": CLAUDE_CODE_CLI_VERSION,
    "cli_sha256": CLAUDE_CODE_CLI_SHA256,
    "node_installer_version": NVM_INSTALL_VERSION,
    "node_installer_sha256": NVM_INSTALL_SHA256,
}


def claude_code_harness(
    system_prompt: str | None = None,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    config_dir: str = DEFAULT_CONFIG_DIR,
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    permission_mode: str | None = DEFAULT_PERMISSION_MODE,
    max_turns: int | None = None,
    reasoning_effort: str | None = None,
    max_budget_usd: str | None = None,
    fallback_model: str | None = None,
    append_system_prompt: str | None = None,
    allowed_tools: str | list[str] | None = None,
    disallowed_tools: str | list[str] | None = None,
    max_thinking_tokens: int | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    skills_dir: str | None = None,
    memory_dir: str | None = None,
):
    """Create a Harness configured for Claude Code."""
    from harnesses.base import make_native_harness

    install_script = build_claude_code_install_script(
        package_version=package_version,
        package_sha256=package_sha256,
    )

    return make_native_harness(
        build_run_command=build_claude_code_run_command,
        run_kwargs={
            "agent_workdir": agent_workdir,
            "instruction_path": instruction_path,
            "system_prompt_path": system_prompt_path,
            "log_path": log_path,
            "config_dir": config_dir,
            "output_format": output_format,
            "permission_mode": permission_mode,
            "max_turns": max_turns,
            "reasoning_effort": reasoning_effort,
            "max_budget_usd": max_budget_usd,
            "fallback_model": fallback_model,
            "append_system_prompt": append_system_prompt,
            "allowed_tools": allowed_tools,
            "disallowed_tools": disallowed_tools,
            "max_thinking_tokens": max_thinking_tokens,
            "memory_dir": memory_dir,
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
