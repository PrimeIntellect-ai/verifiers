"""OpenAI Codex CLI harness configuration.

The generated Harness installs Codex in the sandbox, receives task/system prompt
files from ComposableEnv, and talks to the evaluator through OpenAI-style env
vars such as OPENAI_BASE_URL and OPENAI_MODEL.
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

CODEX_CLI_PACKAGE = "@openai/codex"
CODEX_CLI_VERSION = "0.120.0"
CODEX_CLI_SHA256 = "1dd5a298cf40c96590d3841c94f07d6135416682f3a8c89a04713bbfba8b975e"
DEFAULT_PACKAGE_VERSION = CODEX_CLI_VERSION
DEFAULT_PACKAGE_SHA256 = CODEX_CLI_SHA256
DEFAULT_INSTRUCTION_PATH = "/codex/prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/codex/system.txt"
DEFAULT_LOG_DIR = "/logs/agent"
DEFAULT_LOG_PATH = f"{DEFAULT_LOG_DIR}/codex.txt"
DEFAULT_LAST_MESSAGE_PATH = f"{DEFAULT_LOG_DIR}/codex-last-message.txt"
DEFAULT_CODEX_HOME = f"{DEFAULT_LOG_DIR}/codex-home"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_SKILLS_PATH = "/task/skills"


def _toml_key(value: str) -> str:
    """Return a TOML key, quoting names that are not bare-key safe."""
    if (
        value
        and not value[0].isdigit()
        and all(c.isalnum() or c in "_-" for c in value)
    ):
        return value
    return json.dumps(value)


def _toml_value(value: Any) -> str:
    """Render simple Python values as TOML literals for generated config."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(json.dumps(str(item)) for item in value) + "]"
    return json.dumps(str(value))


def build_codex_install_script(
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    install_ripgrep: bool = True,
) -> str:
    """Build the shell script that installs OpenAI Codex CLI in a sandbox."""
    # Alpine can use distro node/npm directly; glibc images get nvm so the CLI
    # has a recent enough Node without depending on the base image.
    rg_package = " ripgrep" if install_ripgrep else ""
    npm_install = indent_shell_block(
        build_verified_npm_install_command(
            CODEX_CLI_PACKAGE,
            package_version,
            package_sha256,
            "CODEX_NPM",
        )
    )
    nvm_install = indent_shell_block(build_verified_nvm_install_command())
    return f"""\
set -e

if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then
  apk add --no-cache curl bash nodejs npm ca-certificates{rg_package}
{npm_install}
else
  if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl bash ca-certificates{rg_package} > /dev/null 2>&1
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y curl bash ca-certificates{rg_package}
  elif command -v yum >/dev/null 2>&1; then
    yum install -y curl bash ca-certificates{rg_package}
  elif command -v pacman >/dev/null 2>&1; then
    pacman -Sy --noconfirm curl bash ca-certificates{rg_package}
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

for bin in node npm codex; do
  BIN_PATH="$(command -v "$bin" 2>/dev/null || true)"
  if [ -n "$BIN_PATH" ]; then
    ln -sf "$BIN_PATH" "/usr/local/bin/$bin" 2>/dev/null || true
  fi
done

codex --version
"""


def build_codex_mcp_config(
    mcp_servers: list[dict[str, Any] | str] | None = None,
) -> str:
    """Render Codex MCP server dictionaries as config.toml entries.

    Accepts string shorthands, stdio command servers, URL servers, env/header
    sections, and per-tool config dictionaries.
    """
    # Codex reads MCP servers from config.toml, so this accepts the same compact
    # server dictionaries used by the other harnesses and emits Codex's table form.
    lines: list[str] = []
    for server in mcp_servers or []:
        if isinstance(server, str):
            server = {"name": server, "command": server}

        name = _toml_key(str(server["name"]))
        lines.append(f"[mcp_servers.{name}]")
        if server.get("url") and server.get("transport") != "stdio":
            lines.append(f"url = {_toml_value(server['url'])}")
        else:
            lines.append(f"command = {_toml_value(server['command'])}")
            if server.get("args"):
                lines.append(f"args = {_toml_value(server['args'])}")

        for key in (
            "enabled",
            "startup_timeout_sec",
            "disabled_tools",
            "enabled_tools",
            "env_vars",
            "bearer_token_env_var",
            "oauth_resource",
        ):
            if key in server and server[key] is not None:
                lines.append(f"{key} = {_toml_value(server[key])}")

        env = server.get("env")
        if isinstance(env, dict) and env:
            lines.append("")
            lines.append(f"[mcp_servers.{name}.env]")
            lines.extend(
                f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in env.items()
            )

        headers = server.get("http_headers") or server.get("headers")
        if isinstance(headers, dict) and headers:
            lines.append("")
            lines.append(f"[mcp_servers.{name}.http_headers]")
            lines.extend(
                f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in headers.items()
            )

        tools = server.get("tools")
        if isinstance(tools, dict):
            for tool_name, tool_config in tools.items():
                lines.append("")
                lines.append(f"[mcp_servers.{name}.tools.{_toml_key(str(tool_name))}]")
                if isinstance(tool_config, str):
                    lines.append(f"approval_mode = {_toml_value(tool_config)}")
                elif isinstance(tool_config, dict):
                    lines.extend(
                        f"{_toml_key(str(k))} = {_toml_value(v)}"
                        for k, v in tool_config.items()
                    )
        lines.append("")

    return "\n".join(lines).rstrip()


def build_codex_run_command(
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    last_message_path: str = DEFAULT_LAST_MESSAGE_PATH,
    codex_home: str = DEFAULT_CODEX_HOME,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    reasoning_summary: str | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    skills_dir: str | None = None,
    sandbox_bypass: bool = True,
    skip_git_repo_check: bool = True,
    json_output: bool = True,
    enable_unified_exec: bool = True,
    extra_config_overrides: list[str] | None = None,
) -> str:
    """Build the shell command that prepares CODEX_HOME and runs codex exec.

    The command writes auth/config files, copies skills, combines system and
    task prompts, and then invokes Codex with eval-oriented sandbox flags.
    """
    # Keep the default workdir shell-expanded so env authors can override it with
    # AGENT_WORKDIR without rebuilding the Harness object.
    if agent_workdir == DEFAULT_AGENT_WORKDIR:
        workdir_assignment = f"CODEX_AGENT_WORKDIR={DEFAULT_AGENT_WORKDIR}"
    else:
        workdir_assignment = f"CODEX_AGENT_WORKDIR={shlex.quote(agent_workdir)}"

    config_args: list[str] = []
    # Reasoning settings are passed as Codex config overrides rather than prompt
    # text, matching the CLI's native control surface.
    if reasoning_effort:
        config_args.extend(["--config", f"model_reasoning_effort={reasoning_effort}"])
    if reasoning_summary:
        config_args.extend(["--config", f"model_reasoning_summary={reasoning_summary}"])
    for override in extra_config_overrides or []:
        config_args.extend(["--config", shlex.quote(override)])

    flags: list[str] = []
    if sandbox_bypass:
        flags.append("--dangerously-bypass-approvals-and-sandbox")
    if skip_git_repo_check:
        flags.append("--skip-git-repo-check")
    if json_output:
        flags.append("--json")
    if enable_unified_exec:
        flags.extend(["--enable", "unified_exec"])
    flags.extend(["--model", '"$CODEX_MODEL"'])
    flags.extend(["--output-last-message", shlex.quote(last_message_path)])
    flags.extend(config_args)
    joined_flags = " \\\n  ".join(flags)

    mcp_config = build_codex_mcp_config(mcp_servers)
    config_block = ""
    if mcp_config:
        # The config file starts empty for each rollout; MCP entries are appended
        # only when the caller provided servers.
        config_block = f"""\
cat >> "$CODEX_HOME/config.toml" <<'CODEX_MCP_CONFIG'
{mcp_config}
CODEX_MCP_CONFIG
"""

    skills_block = ""
    if skills_dir:
        # Codex checks CODEX_HOME/skills, while some shared skill tooling checks
        # ~/.agents/skills, so mirror the same files into both locations.
        quoted_skills_dir = shlex.quote(skills_dir)
        skills_block = f"""\
mkdir -p "$CODEX_HOME/skills" "$HOME/.agents/skills"
cp -r {quoted_skills_dir}/* "$CODEX_HOME/skills/" 2>/dev/null || true
cp -r {quoted_skills_dir}/* "$HOME/.agents/skills/" 2>/dev/null || true
"""

    script = f"""\
set -eo pipefail

if [ -s "$HOME/.nvm/nvm.sh" ]; then
  . "$HOME/.nvm/nvm.sh"
fi

export CODEX_HOME={shlex.quote(codex_home)}
export OPENAI_API_KEY="${{OPENAI_API_KEY:-${{CODEX_API_KEY:-intercepted}}}}"
export CODEX_API_KEY="${{CODEX_API_KEY:-$OPENAI_API_KEY}}"
CODEX_MODEL="${{OPENAI_MODEL##*/}}"

{workdir_assignment}
mkdir -p \\
  {shlex.quote(log_path.rsplit("/", 1)[0])} \\
  {shlex.quote(last_message_path.rsplit("/", 1)[0])} \\
  "$CODEX_HOME" \\
  "$CODEX_AGENT_WORKDIR"

: > "$CODEX_HOME/config.toml"
{config_block}{skills_block}
mkdir -p /tmp/codex-secrets
cat >/tmp/codex-secrets/auth.json <<EOF
{{
  "OPENAI_API_KEY": "${{OPENAI_API_KEY}}"
}}
EOF
ln -sf /tmp/codex-secrets/auth.json "$CODEX_HOME/auth.json"
trap 'rm -rf /tmp/codex-secrets "$CODEX_HOME/auth.json" "$CODEX_HOME/tmp"' EXIT

if [ -s {shlex.quote(system_prompt_path)} ]; then
  CODEX_PROMPT="$(cat {shlex.quote(system_prompt_path)})

$(cat {shlex.quote(instruction_path)})"
else
  CODEX_PROMPT="$(cat {shlex.quote(instruction_path)})"
fi

cd "$CODEX_AGENT_WORKDIR"
codex exec \\
  {joined_flags} \\
  -- \\
  "$CODEX_PROMPT" 2>&1 </dev/null | tee -a {shlex.quote(log_path)}
"""
    return f"bash -lc {shlex.quote(script)}"


CODEX_INSTALL_SCRIPT = build_codex_install_script()
CODEX_CONFIG = {
    "install_script": CODEX_INSTALL_SCRIPT,
    "cli_package": CODEX_CLI_PACKAGE,
    "cli_version": CODEX_CLI_VERSION,
    "cli_sha256": CODEX_CLI_SHA256,
    "node_installer_version": NVM_INSTALL_VERSION,
    "node_installer_sha256": NVM_INSTALL_SHA256,
}


def codex_harness(
    system_prompt: str | None = None,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    last_message_path: str = DEFAULT_LAST_MESSAGE_PATH,
    codex_home: str = DEFAULT_CODEX_HOME,
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    reasoning_summary: str | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    skills_dir: str | None = None,
    sandbox_bypass: bool = True,
    skip_git_repo_check: bool = True,
    json_output: bool = True,
    enable_unified_exec: bool = True,
    extra_config_overrides: list[str] | None = None,
):
    """Create a Harness configured for OpenAI Codex CLI."""
    from harnesses.base import make_native_harness

    install_script = build_codex_install_script(
        package_version=package_version,
        package_sha256=package_sha256,
    )

    # ComposableEnv owns prompt file materialization; this factory only wires the
    # install and run commands around those paths.
    return make_native_harness(
        build_run_command=build_codex_run_command,
        run_kwargs={
            "agent_workdir": agent_workdir,
            "instruction_path": instruction_path,
            "system_prompt_path": system_prompt_path,
            "log_path": log_path,
            "last_message_path": last_message_path,
            "codex_home": codex_home,
            "reasoning_effort": reasoning_effort,
            "reasoning_summary": reasoning_summary,
            "sandbox_bypass": sandbox_bypass,
            "skip_git_repo_check": skip_git_repo_check,
            "json_output": json_output,
            "enable_unified_exec": enable_unified_exec,
            "extra_config_overrides": extra_config_overrides,
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
