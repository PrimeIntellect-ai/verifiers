import json
import shlex
from pathlib import PurePosixPath
from typing import cast

from .command import configure_command_harness
from .configs import OpenCodeConfig
from ...harness import Harness
from ...utils.mcp_proxy_utils import proxy_command
from ...types import (
    ConfigData,
    ProgramCommand,
    ProgramChannels,
    ProgramOptionMap,
    ProgramSetup,
)


class OpenCode(Harness[OpenCodeConfig]):
    def __init__(self, config: OpenCodeConfig | None = None):
        config = cast(OpenCodeConfig, self._coerce_config(config))
        super().__init__(config=config.model_copy(update={"program": None}))
        self.config = config
        configure_command_harness(
            self,
            config,
            command=self.command(config),
            setup=self.setup(config),
            artifacts=self.artifacts(config),
            channels=self.channels(config),
        )

    def command(self, config: OpenCodeConfig) -> ProgramCommand:
        return [
            "bash",
            "-lc",
            build_opencode_run_script(
                agent_workdir=config.agent_workdir,
                instruction_path=config.instruction_path,
                log_path=config.log_path,
                allow_git=config.allow_git,
            ),
        ]

    def setup(self, config: OpenCodeConfig) -> ProgramSetup:
        return build_install_script(
            release_repo=config.release_repo,
            release_version=config.release_version,
            release_sha256=config.release_sha256,
            install_ripgrep=config.install_ripgrep,
        )

    def artifacts(self, config: OpenCodeConfig) -> ProgramOptionMap:
        return {
            "opencode_log": {
                "path": config.log_path,
                "format": "text",
                "optional": True,
            }
        }

    def channels(self, config: OpenCodeConfig) -> ProgramChannels:
        return {
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
        }


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)


def build_install_script(
    release_repo: str,
    release_version: str,
    release_sha256: str,
    install_ripgrep: bool,
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
    agent_config: ConfigData = {
        "title": {"disable": True},
    }
    config: ConfigData = {
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
    build_config: ConfigData = {}
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
