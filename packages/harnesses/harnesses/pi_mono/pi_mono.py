"""Pi coding agent harness configuration.

The generated Harness installs the pi CLI from ``pi-mono``, writes a rollout
local models.json for the evaluator's OpenAI-compatible endpoint, and runs pi in
non-interactive JSON mode against the sandbox task prompt.
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

PI_MONO_CLI_PACKAGE = "@mariozechner/pi-coding-agent"
PI_MONO_CLI_VERSION = "0.66.1"
PI_MONO_CLI_SHA256 = "34ddba0371107ede41872bb9dc998d10277592090d3cfb1ee81b039f019bcf21"
DEFAULT_PACKAGE_VERSION = PI_MONO_CLI_VERSION
DEFAULT_PACKAGE_SHA256 = PI_MONO_CLI_SHA256
DEFAULT_INSTRUCTION_PATH = "/pi-mono/prompt.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/pi-mono/system.txt"
DEFAULT_LOG_DIR = "/logs/agent"
DEFAULT_LOG_PATH = f"{DEFAULT_LOG_DIR}/pi-mono.txt"
DEFAULT_AGENT_DIR = f"{DEFAULT_LOG_DIR}/pi-mono-agent"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_SKILLS_PATH = "/task/skills"
DEFAULT_PROVIDER_NAME = "intercepted"
DEFAULT_MODEL_ID = "$OPENAI_MODEL"
DEFAULT_API = "openai-completions"
DEFAULT_OUTPUT_MODE = "json"
DEFAULT_TOOLS = ("read", "bash", "edit", "write", "grep", "find", "ls")
DEFAULT_COMPAT = {
    "supportsStore": False,
    "supportsDeveloperRole": False,
    "supportsReasoningEffort": False,
    "maxTokensField": "max_tokens",
}


def build_pi_mono_install_script(
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
) -> str:
    """Build the shell script that installs the pi coding agent in a sandbox.

    Alpine can use distro node/npm directly; glibc images get nvm so the CLI has
    a predictable modern Node version.
    """
    npm_install = indent_shell_block(
        build_verified_npm_install_command(
            PI_MONO_CLI_PACKAGE,
            package_version,
            package_sha256,
            "PI_MONO_NPM",
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

for bin in node npm pi; do
  BIN_PATH="$(command -v "$bin" 2>/dev/null || true)"
  if [ -n "$BIN_PATH" ]; then
    ln -sf "$BIN_PATH" "/usr/local/bin/$bin" 2>/dev/null || true
  fi
done

pi --version
"""


def build_pi_mono_models_config(
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
) -> str:
    """Render pi's models.json for a custom OpenAI-compatible provider.

    The default values intentionally contain shell placeholders; the run command
    writes this JSON with an expanding heredoc so each rollout can supply its own
    OPENAI_BASE_URL and OPENAI_MODEL.
    """
    model: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "reasoning": reasoning,
        "input": input_modalities or ["text", "image"],
        "contextWindow": context_window,
        "maxTokens": max_tokens,
        "compat": compat if compat is not None else DEFAULT_COMPAT,
    }
    config = {
        "providers": {
            provider_name: {
                "baseUrl": base_url,
                "api": api,
                "apiKey": api_key,
                "models": [model],
            }
        }
    }
    return json.dumps(config, indent=2)


def build_pi_mono_run_command(
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    agent_dir: str = DEFAULT_AGENT_DIR,
    provider_name: str = DEFAULT_PROVIDER_NAME,
    model_id: str = DEFAULT_MODEL_ID,
    api: str = DEFAULT_API,
    reasoning: bool = False,
    thinking: str | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    tools: tuple[str, ...] | list[str] | str | None = DEFAULT_TOOLS,
    skills_dir: str | None = None,
    append_system_prompt: str | None = None,
    no_extensions: bool = True,
    no_prompt_templates: bool = True,
    no_themes: bool = True,
    no_session: bool = True,
    offline: bool = True,
    extra_args: list[str] | None = None,
) -> str:
    """Build the shell command that configures and runs pi for one rollout.

    The command writes models.json, copies optional skills, assembles CLI flags
    as a bash array, and pipes the task prompt through stdin to avoid argument
    parsing surprises when the prompt starts with dashes.
    """
    if agent_workdir == DEFAULT_AGENT_WORKDIR:
        workdir_assignment = f"PI_MONO_AGENT_WORKDIR={DEFAULT_AGENT_WORKDIR}"
    else:
        workdir_assignment = f"PI_MONO_AGENT_WORKDIR={shlex.quote(agent_workdir)}"

    if model_id == DEFAULT_MODEL_ID:
        model_assignment = 'PI_MONO_MODEL="$OPENAI_MODEL"'
    else:
        model_assignment = f"PI_MONO_MODEL={shlex.quote(model_id)}"

    models_config = build_pi_mono_models_config(
        provider_name=provider_name,
        model_id="$PI_MONO_MODEL",
        api=api,
        reasoning=reasoning,
    )

    arg_lines = [
        f"PI_MONO_ARGS+=(--provider {shlex.quote(provider_name)})",
        'PI_MONO_ARGS+=(--model "$PI_MONO_MODEL")',
    ]
    if output_mode == "json":
        arg_lines.append("PI_MONO_ARGS+=(--mode json)")
    else:
        arg_lines.append("PI_MONO_ARGS+=(--print)")
    if thinking:
        arg_lines.append(f"PI_MONO_ARGS+=(--thinking {shlex.quote(thinking)})")
    if tools:
        tool_list = ",".join([tools] if isinstance(tools, str) else tools)
        arg_lines.append(f"PI_MONO_ARGS+=(--tools {shlex.quote(tool_list)})")
    if no_extensions:
        arg_lines.append("PI_MONO_ARGS+=(--no-extensions)")
    if no_prompt_templates:
        arg_lines.append("PI_MONO_ARGS+=(--no-prompt-templates)")
    if no_themes:
        arg_lines.append("PI_MONO_ARGS+=(--no-themes)")
    if no_session:
        arg_lines.append("PI_MONO_ARGS+=(--no-session)")
    if offline:
        arg_lines.append("PI_MONO_ARGS+=(--offline)")
    for arg in extra_args or []:
        arg_lines.append(f"PI_MONO_ARGS+=({shlex.quote(arg)})")

    skills_block = ""
    if skills_dir:
        quoted_skills_dir = shlex.quote(skills_dir)
        skills_block = f"""\
cp -r {quoted_skills_dir}/* "$PI_CODING_AGENT_DIR/skills/" 2>/dev/null || true
PI_MONO_ARGS+=(--skill "$PI_CODING_AGENT_DIR/skills")
"""

    append_system_prompt_block = ""
    if append_system_prompt:
        append_system_prompt_block = (
            f"PI_MONO_SYSTEM_PROMPT+=$'\\n\\n'{shlex.quote(append_system_prompt)}\n"
        )

    offline_env = "export PI_OFFLINE=1\n" if offline else ""
    joined_arg_lines = "\n".join(arg_lines)
    script = f"""\
set -eo pipefail

if [ -s "$HOME/.nvm/nvm.sh" ]; then
  . "$HOME/.nvm/nvm.sh"
fi
export PATH="$HOME/.local/bin:$PATH"
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export PI_SKIP_VERSION_CHECK=1
{offline_env}export PI_CODING_AGENT_DIR={shlex.quote(agent_dir)}
{workdir_assignment}
{model_assignment}
mkdir -p \\
  {shlex.quote(log_path.rsplit("/", 1)[0])} \\
  "$PI_CODING_AGENT_DIR/skills" \\
  "$PI_CODING_AGENT_DIR/sessions" \\
  "$PI_MONO_AGENT_WORKDIR"

cat > "$PI_CODING_AGENT_DIR/models.json" <<PI_MONO_MODELS_JSON
{models_config}
PI_MONO_MODELS_JSON

PI_MONO_PROMPT="$(cat {shlex.quote(instruction_path)})"
PI_MONO_SYSTEM_PROMPT=""
if [ -s {shlex.quote(system_prompt_path)} ]; then
  PI_MONO_SYSTEM_PROMPT="$(cat {shlex.quote(system_prompt_path)})"
fi
{append_system_prompt_block}
PI_MONO_ARGS=()
{joined_arg_lines}
if [ -n "$PI_MONO_SYSTEM_PROMPT" ]; then
  PI_MONO_ARGS+=(--system-prompt "$PI_MONO_SYSTEM_PROMPT")
fi
{skills_block}
cd "$PI_MONO_AGENT_WORKDIR"
printf '%s' "$PI_MONO_PROMPT" | pi "${{PI_MONO_ARGS[@]}}" 2>&1 | tee -a {shlex.quote(log_path)}
"""
    return f"bash -lc {shlex.quote(script)}"


PI_MONO_INSTALL_SCRIPT = build_pi_mono_install_script()
PI_MONO_CONFIG = {
    "install_script": PI_MONO_INSTALL_SCRIPT,
    "cli_package": PI_MONO_CLI_PACKAGE,
    "cli_version": PI_MONO_CLI_VERSION,
    "cli_sha256": PI_MONO_CLI_SHA256,
    "node_installer_version": NVM_INSTALL_VERSION,
    "node_installer_sha256": NVM_INSTALL_SHA256,
}


def pi_mono_harness(
    system_prompt: str | None = None,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    agent_dir: str = DEFAULT_AGENT_DIR,
    package_version: str = DEFAULT_PACKAGE_VERSION,
    package_sha256: str = DEFAULT_PACKAGE_SHA256,
    provider_name: str = DEFAULT_PROVIDER_NAME,
    model_id: str = DEFAULT_MODEL_ID,
    api: str = DEFAULT_API,
    reasoning: bool = False,
    thinking: str | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    tools: tuple[str, ...] | list[str] | str | None = DEFAULT_TOOLS,
    skills_dir: str | None = None,
    append_system_prompt: str | None = None,
    no_extensions: bool = True,
    no_prompt_templates: bool = True,
    no_themes: bool = True,
    no_session: bool = True,
    offline: bool = True,
    extra_args: list[str] | None = None,
):
    """Create a Harness configured for the pi coding agent."""
    from harnesses.base import make_native_harness

    # ComposableEnv writes the prompt files; this factory wires pi's package
    # install, isolated config dir, and non-interactive run command around them.
    install_script = build_pi_mono_install_script(
        package_version=package_version,
        package_sha256=package_sha256,
    )

    return make_native_harness(
        build_run_command=build_pi_mono_run_command,
        run_kwargs={
            "agent_workdir": agent_workdir,
            "instruction_path": instruction_path,
            "system_prompt_path": system_prompt_path,
            "log_path": log_path,
            "agent_dir": agent_dir,
            "provider_name": provider_name,
            "model_id": model_id,
            "api": api,
            "reasoning": reasoning,
            "thinking": thinking,
            "output_mode": output_mode,
            "tools": tools,
            "append_system_prompt": append_system_prompt,
            "no_extensions": no_extensions,
            "no_prompt_templates": no_prompt_templates,
            "no_themes": no_themes,
            "no_session": no_session,
            "offline": offline,
            "extra_args": extra_args,
        },
        install_script=install_script,
        system_prompt=system_prompt,
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_path=log_path,
        default_skills_path=DEFAULT_SKILLS_PATH,
        skills_dir=skills_dir,
        supports_tools=False,
        pass_mcp_servers=False,
    )
