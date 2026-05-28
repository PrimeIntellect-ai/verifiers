import json
import shlex
from pathlib import PurePosixPath

from verifiers.v1.config import ConfigSource
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.program import (
    Program,
    ProgramChannels,
    ProgramCommand,
    ProgramOptionMap,
    ProgramSetup,
    ProgramValue,
)
from verifiers.v1.types import (
    ConfigData,
    ConfigMap,
    PromptInput,
)
from verifiers.v1.utils.mcp_proxy_utils import proxy_command
from verifiers.v1.utils.config_utils import coerce_config

OPENCODE_DEFAULT_RELEASE_REPO = "PrimeIntellect-ai/opencode"
OPENCODE_DEFAULT_RELEASE_VERSION = "1.1.63-rl2"
OPENCODE_DEFAULT_RELEASE_SHA256 = (
    "47f4102796da50769e27d2c9ea6a9cf7941f76898390cb497278cab39c4b6ed4"
)
OPENCODE_DEFAULT_AGENT_WORKDIR = "/app"
OPENCODE_DEFAULT_INSTRUCTION_PATH = "/opencode/instruction.txt"
OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH = "/opencode/system.txt"
OPENCODE_DEFAULT_LOG_PATH = "/logs/agent/opencode.txt"
OPENCODE_DEFAULT_SYSTEM_PROMPT = """\
You are OpenCode, an interactive CLI tool that helps users with tasks.

Your output is displayed in a command line interface. Be concise and direct.
Use tools to complete tasks. Do not use shell commands or code comments as a
way to communicate with the user.
"""
OPENCODE_DEFAULT_DISABLED_TOOLS = [
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


class OpenCodeConfig(HarnessConfig):
    agent_workdir: str = OPENCODE_DEFAULT_AGENT_WORKDIR
    instruction_path: str = OPENCODE_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = OPENCODE_DEFAULT_LOG_PATH
    system_prompt: PromptInput | None = OPENCODE_DEFAULT_SYSTEM_PROMPT
    disabled_tools: list[str] = OPENCODE_DEFAULT_DISABLED_TOOLS
    allow_git: bool = False
    disable_compaction: bool = True
    release_repo: str = OPENCODE_DEFAULT_RELEASE_REPO
    release_version: str = OPENCODE_DEFAULT_RELEASE_VERSION
    release_sha256: str = OPENCODE_DEFAULT_RELEASE_SHA256
    install_ripgrep: bool = True
    provider_timeout_ms: int = 3_600_000
    max_turns: int = 4


class OpenCode(Harness[OpenCodeConfig]):
    config: OpenCodeConfig

    def __init__(self, config: ConfigSource = None):
        config_value = coerce_config(OpenCodeConfig, config)
        self.command_program_parts = self._program_config(config_value)
        super().__init__(config=config_value)

    def load_program(self) -> Program:
        program, _ = self.command_program_parts
        return program

    def load_sandbox(self) -> ConfigMap | None:
        _, sandbox = self.command_program_parts
        return sandbox

    @classmethod
    def _program_config(
        cls, config: OpenCodeConfig
    ) -> tuple[Program, ConfigData | None]:
        return Harness.command_program_config(
            config,
            command=cls._command(config),
            files=cls._files(config),
            setup=cls._setup(config),
            artifacts=cls._artifacts(config),
            channels=cls._channels(config),
        )

    @classmethod
    def _command(cls, config: OpenCodeConfig) -> ProgramCommand:
        return [
            "bash",
            "-lc",
            cls._run_script(
                agent_workdir=config.agent_workdir,
                instruction_path=config.instruction_path,
                log_path=config.log_path,
                allow_git=config.allow_git,
            ),
        ]

    @classmethod
    def _setup(cls, config: OpenCodeConfig) -> ProgramSetup:
        return cls._install_script(
            release_repo=config.release_repo,
            release_version=config.release_version,
            release_sha256=config.release_sha256,
            install_ripgrep=config.install_ripgrep,
        )

    @classmethod
    def _files(cls, config: OpenCodeConfig) -> ProgramOptionMap:
        files: dict[str, ProgramValue] = {
            config.instruction_path: {
                "fn": "verifiers.v1.utils.prompt_utils:task_text"
            },
        }
        if config.system_prompt is not None:
            files[config.system_prompt_path] = {
                "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
            }
        return files

    @classmethod
    def _artifacts(cls, config: OpenCodeConfig) -> ProgramOptionMap:
        return {
            "opencode_log": {
                "path": config.log_path,
                "format": "text",
                "optional": True,
            }
        }

    @classmethod
    def _channels(cls, config: OpenCodeConfig) -> ProgramChannels:
        return {
            "mcp": cls._mcp_setup_script(
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

    @classmethod
    def _install_script(
        cls,
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

    @classmethod
    def _opencode_config(
        cls,
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

    @classmethod
    def _run_script(
        cls,
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
if [ -z "$OPENCODE_WORKDIR" ]; then
    OPENCODE_WORKDIR={shlex.quote(agent_workdir)}
fi

cd "$OPENCODE_WORKDIR"
cat {shlex.quote(instruction_path)} | opencode run 2>&1 | tee {shlex.quote(log_path)}
"""
        return script

    @classmethod
    def _mcp_setup_script(
        cls,
        *,
        agent_workdir: str,
        system_prompt_path: str | None,
        log_path: str,
        disabled_tools: list[str],
        disable_compaction: bool,
        provider_timeout_ms: int,
    ) -> str:
        config_json = cls._opencode_config(
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
if [ -z "$OPENCODE_WORKDIR" ]; then
    OPENCODE_WORKDIR={shlex.quote(agent_workdir)}
fi

mkdir -p ~/.config/opencode {shlex.quote(log_dir)} "$OPENCODE_WORKDIR"
SCHEMA_DOLLAR='$'
cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG
"""


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)
