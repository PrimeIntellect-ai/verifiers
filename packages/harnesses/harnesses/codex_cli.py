import os
import shlex
from pathlib import PurePosixPath
from typing import Literal

import verifiers as vf

from .utils import split_versioned_agent_spec

CODEX_CLI_DEFAULT_VERSION = "codex@latest"
CODEX_CLI_DEFAULT_INSTALL_DIR = "/opt/codex-cli"
CODEX_CLI_DEFAULT_AGENT_WORKDIR = "/app"
CODEX_CLI_DEFAULT_CODEX_HOME_PATH = "/codex-cli/home"
CODEX_CLI_DEFAULT_INSTRUCTION_PATH = "/codex-cli/instruction.txt"
CODEX_CLI_DEFAULT_SYSTEM_PROMPT_PATH = "/codex-cli/system.txt"
CODEX_CLI_DEFAULT_LOG_PATH = "/logs/agent/codex-cli.jsonl"
CODEX_CLI_DEFAULT_LAST_MESSAGE_PATH = "/logs/agent/codex-cli-last-message.txt"
CODEX_CLI_DEFAULT_SYSTEM_PROMPT = "Complete the user's task using the available tools."

CodexCLIAuthMode = Literal["api_key", "chatgpt"]


def codex_chatgpt_auth_json(auth_json_var: str) -> str:
    value = os.environ.get(auth_json_var)
    if not value:
        raise RuntimeError(
            f"{auth_json_var} must contain Codex ChatGPT auth.json content."
        )
    return value


class CodexCLIProgramConfig(vf.ProgramConfig):
    agent_workdir: str = CODEX_CLI_DEFAULT_AGENT_WORKDIR
    codex_home_path: str = CODEX_CLI_DEFAULT_CODEX_HOME_PATH
    auth_json_var: str = "CODEX_AUTH_JSON"
    instruction_path: str = CODEX_CLI_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = CODEX_CLI_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = CODEX_CLI_DEFAULT_LOG_PATH
    last_message_path: str = CODEX_CLI_DEFAULT_LAST_MESSAGE_PATH
    auth_mode: CodexCLIAuthMode = "api_key"
    sandbox: vf.SandboxConfig | None = vf.SandboxConfig()

    def resolve(self, version: str = CODEX_CLI_DEFAULT_VERSION) -> vf.ProgramConfig:
        files: dict[str, vf.ProgramValue] = {
            self.instruction_path: {"fn": "verifiers.v1.utils.prompt_utils:task_text"},
            self.system_prompt_path: {
                "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
            },
        }
        artifacts = vf.ArtifactsConfig.model_validate(
            {
                "codex_cli_log": {
                    "path": self.log_path,
                    "format": "text",
                    "optional": True,
                },
                "codex_cli_last_message": {
                    "path": self.last_message_path,
                    "format": "text",
                    "optional": True,
                },
            }
        )
        name, parsed_version = split_versioned_agent_spec(version)
        release = parsed_version or name
        if release in {"", "codex", "openai/codex"}:
            release = "latest"

        install_home = f"{CODEX_CLI_DEFAULT_INSTALL_DIR}/home"
        bin_dir = f"{CODEX_CLI_DEFAULT_INSTALL_DIR}/bin"
        setup = f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq ca-certificates curl git python3 tar > /dev/null 2>&1
ln -sf "$(command -v python3)" /usr/local/bin/python
CODEX_RELEASE={shlex.quote(release)} \\
CODEX_NON_INTERACTIVE=1 \\
CODEX_INSTALL_DIR={shlex.quote(bin_dir)} \\
CODEX_HOME={shlex.quote(install_home)} \\
sh -c "$(curl -fsSL https://chatgpt.com/codex/install.sh)"
"""
        if self.auth_mode == "api_key":
            auth_setup = """\
if [ -z "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY=intercepted
fi
printf '%s' "$OPENAI_API_KEY" | codex login --with-api-key >/dev/null
CODEX_CONFIG_ARGS+=(-c "openai_base_url=\\"${OPENAI_BASE_URL:-https://api.openai.com/v1}\\"")
"""
        else:
            auth_setup = """\
CODEX_AUTH_JSON_VAR={auth_json_var}
CODEX_AUTH_JSON="$(printenv "$CODEX_AUTH_JSON_VAR" || true)"
if [ -z "$CODEX_AUTH_JSON" ]; then
  echo "Codex ChatGPT auth requires $CODEX_AUTH_JSON_VAR to contain auth.json." >&2
  exit 1
fi
printf '%s' "$CODEX_AUTH_JSON" > "$CODEX_HOME/auth.json"
chmod 600 "$CODEX_HOME/auth.json"
""".format(auth_json_var=shlex.quote(self.auth_json_var))
        log_dir = str(PurePosixPath(self.log_path).parent)
        last_message_dir = str(PurePosixPath(self.last_message_path).parent)
        run_script = f"""\
set -eo pipefail
export PATH={shlex.quote(bin_dir)}:"$PATH"
export CODEX_HOME={shlex.quote(self.codex_home_path)}

CODEX_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$CODEX_WORKDIR" ]; then
    CODEX_WORKDIR={shlex.quote(self.agent_workdir)}
fi

CODEX_MODEL="${{OPENAI_MODEL:-gpt-5}}"
case "$CODEX_MODEL" in
  openai/*) CODEX_MODEL="${{CODEX_MODEL#openai/}}" ;;
esac

mkdir -p "$CODEX_HOME" "$CODEX_WORKDIR" {shlex.quote(log_dir)} {shlex.quote(last_message_dir)}
CODEX_CONFIG_ARGS=(-c 'model_provider="openai"')
{auth_setup}
if [ -s {shlex.quote(self.system_prompt_path)} ]; then
  CODEX_DEVELOPER_INSTRUCTIONS="$(sed 's/\\\\/\\\\\\\\/g; s/"/\\\\"/g' {shlex.quote(self.system_prompt_path)} | awk '{{printf "%s\\\\n", $0}}')"
  CODEX_CONFIG_ARGS+=(-c "developer_instructions=\\"$CODEX_DEVELOPER_INSTRUCTIONS\\"")
fi

cd "$CODEX_WORKDIR"
timeout --kill-after=30s "${{AGENT_TIMEOUT_SECONDS:-3600}}" codex exec \\
  --ignore-user-config \\
  --ephemeral \\
  --skip-git-repo-check \\
  --dangerously-bypass-approvals-and-sandbox \\
  --json \\
  --model "$CODEX_MODEL" \\
  --cd "$CODEX_WORKDIR" \\
  --output-last-message {shlex.quote(self.last_message_path)} \\
  "${{CODEX_CONFIG_ARGS[@]}}" \\
  - < {shlex.quote(self.instruction_path)} 2>&1 | tee {shlex.quote(self.log_path)}
"""
        env: dict[str, vf.ProgramValue] = {"OPENAI_MODEL": "runtime.model"}
        if self.auth_mode == "chatgpt":
            env[self.auth_json_var] = {
                "fn": "harnesses.codex_cli:codex_chatgpt_auth_json",
                "auth_json_var": self.auth_json_var,
            }

        return self.resolve_command(
            command=["bash", "-lc", run_script],
            default_sandbox=self.sandbox,
            files=files,
            setup=setup,
            env=env,
            artifacts=artifacts,
        )


class CodexCLIConfig(vf.HarnessConfig):
    system_prompt: vf.PromptInput | vf.SystemPromptConfig | None = (
        CODEX_CLI_DEFAULT_SYSTEM_PROMPT
    )
    version: str = CODEX_CLI_DEFAULT_VERSION
    program: CodexCLIProgramConfig = CodexCLIProgramConfig()
    max_turns: int = 4


class CodexCLI(vf.Harness[CodexCLIConfig]):
    config: CodexCLIConfig

    def load_program_config(self, config: CodexCLIConfig) -> vf.ProgramConfig:
        return config.program.resolve(version=config.version)


def load_harness(config: CodexCLIConfig) -> CodexCLI:
    return CodexCLI(config=config)
