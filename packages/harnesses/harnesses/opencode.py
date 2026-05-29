import json
import shlex
from pathlib import PurePosixPath

import verifiers as vf
from pydantic import model_validator
from verifiers.v1.utils.mcp_proxy_utils import proxy_command

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


class OpenCodeConfig(vf.HarnessConfig):
    agent_workdir: str = OPENCODE_DEFAULT_AGENT_WORKDIR
    instruction_path: str = OPENCODE_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = OPENCODE_DEFAULT_LOG_PATH
    system_prompt: vf.PromptInput | None = OPENCODE_DEFAULT_SYSTEM_PROMPT
    disabled_tools: list[str] = OPENCODE_DEFAULT_DISABLED_TOOLS
    allow_git: bool = False
    disable_compaction: bool = True
    release_repo: str = OPENCODE_DEFAULT_RELEASE_REPO
    release_version: str = OPENCODE_DEFAULT_RELEASE_VERSION
    release_sha256: str = OPENCODE_DEFAULT_RELEASE_SHA256
    install_ripgrep: bool = True
    provider_timeout_ms: int = 3_600_000
    max_turns: int = 4

    @model_validator(mode="after")
    def configure_program(self) -> "OpenCodeConfig":
        if self.program.command is not None and "program" not in self.model_fields_set:
            return self
        harness_config = self
        files: dict[str, vf.ProgramValue] = {
            harness_config.instruction_path: {
                "fn": "verifiers.v1.utils.prompt_utils:task_text"
            },
        }
        if harness_config.system_prompt is not None:
            files[harness_config.system_prompt_path] = {
                "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
            }
        artifacts: dict[str, vf.ProgramValue] = {
            "opencode_log": {
                "path": harness_config.log_path,
                "format": "text",
                "optional": True,
            }
        }
        rg_install = (
            "apt-get -o Acquire::Retries=3 install -y -qq ripgrep > /dev/null 2>&1 || true"
            if harness_config.install_ripgrep
            else ""
        )
        sha256_check = f'echo "{harness_config.release_sha256}  /tmp/opencode.tar.gz" | sha256sum -c -'
        # Acquire::Retries=3 mitigates transient archive.ubuntu.com CDN sync
        # mismatches that fail fresh-sandbox apt-get calls mid-rollout.
        setup = f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl tar ca-certificates > /dev/null 2>&1
{rg_install}

OPENCODE_RELEASE_REPO={shlex.quote(harness_config.release_repo)}
OPENCODE_RELEASE_VERSION={shlex.quote(harness_config.release_version)}

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
        agent_config: vf.ConfigData = {
            "title": {"disable": True},
        }
        opencode_config: vf.ConfigData = {
            "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
            "provider": {
                "intercepted": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Intercepted",
                    "options": {
                        "baseURL": "$OPENAI_BASE_URL",
                        "apiKey": "${OPENAI_API_KEY:-intercepted}",
                        "timeout": harness_config.provider_timeout_ms,
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
        if harness_config.disable_compaction:
            opencode_config["compaction"] = {"auto": False, "prune": False}
        build_config: vf.ConfigData = {}
        system_prompt_path = (
            harness_config.system_prompt_path
            if harness_config.system_prompt is not None
            else None
        )
        if system_prompt_path is not None:
            build_config["prompt"] = "{file:" + system_prompt_path + "}"
        if harness_config.disabled_tools:
            build_config["tools"] = {
                tool: False for tool in harness_config.disabled_tools
            }
        if build_config:
            agent_config["build"] = build_config
        config_json = json.dumps(opencode_config, indent=2)
        log_dir = str(PurePosixPath(harness_config.log_path).parent)
        mcp_setup = f"""\
set -e
export PATH="$HOME/.opencode/bin:$PATH"

OPENCODE_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$OPENCODE_WORKDIR" ]; then
    OPENCODE_WORKDIR={shlex.quote(harness_config.agent_workdir)}
fi

mkdir -p ~/.config/opencode {shlex.quote(log_dir)} "$OPENCODE_WORKDIR"
SCHEMA_DOLLAR='$'
cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG
"""
        run_script = f"""\
set -eo pipefail
export PATH="$HOME/.opencode/bin:$PATH"
export OPENCODE_DISABLE_FILETIME_CHECK=true
export ALLOW_GIT={"1" if harness_config.allow_git else "0"}

OPENCODE_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$OPENCODE_WORKDIR" ]; then
    OPENCODE_WORKDIR={shlex.quote(harness_config.agent_workdir)}
fi

cd "$OPENCODE_WORKDIR"
cat {shlex.quote(harness_config.instruction_path)} | opencode run 2>&1 | tee {shlex.quote(harness_config.log_path)}
"""
        self.program = vf.ProgramConfig.from_command(
            command=["bash", "-lc", run_script],
            program=harness_config.program,
            default_sandbox=harness_config.sandbox,
            files=files,
            setup=setup,
            artifacts=artifacts,
            channels={"mcp": mcp_setup},
        )
        self.model_fields_set.discard("program")
        return self


class OpenCode(vf.Harness[OpenCodeConfig]):
    config: OpenCodeConfig


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)
