"""OpenCode harness configuration.

Provides install script, config generation, and run command templates
that are shared across all OpenCode-based environments (SWE, Lean, Math, etc.).

Usage::

    from verifiers.envs.experimental.composable.harnesses.opencode import opencode_harness
    harness = opencode_harness(system_prompt="You are a coding agent...")
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path, PurePosixPath

# ── Defaults ─────────────────────────────────────────────────────────────

DEFAULT_RELEASE_REPO = "PrimeIntellect-ai/opencode"
DEFAULT_RELEASE_VERSION = "1.1.63-rl2"
DEFAULT_RELEASE_SHA256 = (
    "47f4102796da50769e27d2c9ea6a9cf7941f76898390cb497278cab39c4b6ed4"
)
DEFAULT_SYSTEM_PROMPT = (Path(__file__).parent / "prompt.txt").read_text()

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


# ── Install script ───────────────────────────────────────────────────────


def build_install_script(
    release_repo: str = DEFAULT_RELEASE_REPO,
    release_version: str = DEFAULT_RELEASE_VERSION,
    release_sha256: str = DEFAULT_RELEASE_SHA256,
    install_ripgrep: bool = True,
) -> str:
    """Build the shell script that installs OpenCode in a sandbox."""
    rg_install = (
        "apt-get -o Acquire::Retries=3 install -y -qq ripgrep > /dev/null 2>&1 || true"
        if install_ripgrep
        else ""
    )
    sha256_check = f'echo "{release_sha256}  /tmp/opencode.tar.gz" | sha256sum -c -'
    # Acquire::Retries=3 mitigates transient archive.ubuntu.com CDN sync mismatches
    # (e.g. "File has unexpected size ... Mirror sync in progress?"). See launchpad
    # bug #1876035. apt's default retries is 0, so one bad fetch fails the rollout.
    return f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl tar > /dev/null 2>&1
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
{sha256_check}
tar -xzf /tmp/opencode.tar.gz -C /tmp
install -m 755 /tmp/opencode "$HOME/.opencode/bin/opencode"
echo "OpenCode installed successfully"
"""


# ── Config generation ────────────────────────────────────────────────────


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
) -> str:
    """Generate opencode.json config content."""
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

    return json.dumps(config, indent=2)


# ── Run command ──────────────────────────────────────────────────────────


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
) -> str:
    """Build the shell command that configures and runs OpenCode."""
    config_json = build_opencode_config(
        disabled_tools=disabled_tools,
        system_prompt_path=system_prompt_path,
        disable_compaction=disable_compaction,
        provider_key=provider_key,
        provider_display_name=provider_display_name,
        model_id=model_id,
        model_key=model_key,
        model_display_name=model_display_name,
        provider_timeout_ms=provider_timeout_ms,
    )

    log_dir = str(PurePosixPath(log_path).parent)

    script = f"""\
set -eo pipefail

export PATH="$HOME/.opencode/bin:$PATH"
export OPENCODE_DISABLE_FILETIME_CHECK=true
export ALLOW_GIT={"1" if allow_git else "0"}

# ComposableEnv exports AGENT_WORKDIR from taskset.get_workdir(info) for each
# rollout. Prefer that runtime value; agent_workdir is only the static fallback
# for direct callers that do not run through ComposableEnv.
OPENCODE_WORKDIR="${{AGENT_WORKDIR:-}}"
if [[ -z "$OPENCODE_WORKDIR" ]]; then
    OPENCODE_WORKDIR={shlex.quote(agent_workdir)}
fi

mkdir -p ~/.config/opencode {shlex.quote(log_dir)} "$OPENCODE_WORKDIR"

# Ensure OPENAI_MODEL has provider/model format for opencode AI SDK config.
# LoRA adapter names (e.g. "rft-abc123") lack a slash, causing empty modelID.
if [[ "$OPENAI_MODEL" != *"/"* ]]; then
    export OPENAI_MODEL="vllm/$OPENAI_MODEL"
fi

SCHEMA_DOLLAR='$'

cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG

cd "$OPENCODE_WORKDIR"
cat {shlex.quote(prompt_path)} | opencode run 2>&1 | tee {shlex.quote(log_path)}
"""
    return f"bash -lc {shlex.quote(script)}"


# ── Convenience: pre-built install script ────────────────────────────────

OPENCODE_INSTALL_SCRIPT = build_install_script()


# ── Harness factory ──────────────────────────────────────────────────────


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
):
    """Create a Harness configured for OpenCode.

    Usage::

        from verifiers.envs.experimental.composable.harnesses.opencode import opencode_harness
        harness = opencode_harness(system_prompt="You are a coding agent...")
    """
    from verifiers.envs.experimental.composable import Harness

    if task_system_prompt:
        if system_prompt:
            system_prompt = system_prompt + "\n" + task_system_prompt
        else:
            system_prompt = task_system_prompt

    return Harness(
        install_script=build_install_script(
            release_repo=release_repo,
            release_version=release_version,
            release_sha256=release_sha256,
        ),
        run_command=build_opencode_run_command(
            agent_workdir=agent_workdir,
            prompt_path=instruction_path,
            log_path=log_path,
            disabled_tools=disabled_tools,
            system_prompt_path=system_prompt_path if system_prompt else None,
            disable_compaction=disable_compaction,
            allow_git=allow_git,
            provider_key=provider_key,
            provider_display_name=provider_display_name,
            model_id=model_id,
            model_key=model_key,
            model_display_name=model_display_name,
            provider_timeout_ms=provider_timeout_ms,
        ),
        system_prompt=system_prompt,
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_path=log_path,
    )
