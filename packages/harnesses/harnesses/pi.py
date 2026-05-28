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
    ProgramValue,
)
from verifiers.v1.sandbox import SandboxConfig
from verifiers.v1.state import State
from verifiers.v1.types import (
    ConfigData,
    ConfigMap,
    PromptInput,
)
from verifiers.v1.utils.mcp_proxy_utils import proxy_command
from verifiers.v1.utils.config_utils import coerce_config

PI_DEFAULT_PACKAGE = "@earendil-works/pi-coding-agent"
PI_DEFAULT_WORKDIR = "/app"
PI_DEFAULT_INSTRUCTION_PATH = "/pi/instruction.txt"
PI_DEFAULT_SYSTEM_PROMPT_PATH = "/pi/system.txt"
PI_DEFAULT_LOG_PATH = "/logs/agent/pi.txt"
PI_DEFAULT_SYSTEM_PROMPT = "Complete the user's task using the available tools."


class PiConfig(HarnessConfig):
    agent_workdir: str = PI_DEFAULT_WORKDIR
    instruction_path: str = PI_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = PI_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = PI_DEFAULT_LOG_PATH
    system_prompt: PromptInput | None = PI_DEFAULT_SYSTEM_PROMPT
    package: str = PI_DEFAULT_PACKAGE
    install_mcp_adapter: bool = True
    sandbox: SandboxConfig | None = SandboxConfig()
    max_turns: int = 4


class Pi(Harness[PiConfig]):
    config: PiConfig

    def __init__(self, config: ConfigSource = None):
        config_value = coerce_config(PiConfig, config)
        self.command_program_parts = pi_program_config(config_value)
        super().__init__(config=config_value)

    def load_program(self) -> Program:
        program, _ = self.command_program_parts
        return program

    def load_sandbox(self) -> ConfigMap | None:
        _, sandbox = self.command_program_parts
        return sandbox


def load_harness(config: PiConfig) -> Pi:
    return Pi(config=config)


def pi_program_config(config: PiConfig) -> tuple[Program, ConfigData | None]:
    return Harness.command_program_config(
        config,
        command=pi_command(config),
        files=pi_files(config),
        setup=pi_setup(config),
        artifacts=pi_artifacts(config),
        channels=pi_channels(config),
    )


def pi_command(config: PiConfig) -> ProgramCommand:
    return [
        "bash",
        "-lc",
        build_pi_run_script(
            agent_workdir=config.agent_workdir,
            instruction_path=config.instruction_path,
            system_prompt_path=config.system_prompt_path
            if config.system_prompt is not None
            else None,
            log_path=config.log_path,
        ),
    ]


def pi_setup(config: PiConfig) -> str:
    return f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates nodejs npm xz-utils > /dev/null 2>&1
npm install -g --ignore-scripts n
n 22.19.0
hash -r
npm install -g --ignore-scripts {shlex.quote(config.package)}
"""


def pi_files(config: PiConfig) -> ProgramOptionMap:
    files: dict[str, ProgramValue] = {
        config.instruction_path: {"fn": "verifiers.v1.utils.prompt_utils:task_text"},
    }
    if config.system_prompt is not None:
        files[config.system_prompt_path] = {
            "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
        }
    return files


def pi_artifacts(config: PiConfig) -> ProgramOptionMap:
    return {
        "pi_log": {
            "path": config.log_path,
            "format": "text",
            "optional": True,
        }
    }


def pi_channels(config: PiConfig) -> ProgramChannels | None:
    if not config.install_mcp_adapter:
        return None
    return {
        "mcp": {
            "fn": "harnesses.pi:pi_mcp_setup_script",
            "agent_workdir": config.agent_workdir,
            "install_mcp_adapter": config.install_mcp_adapter,
        }
    }


def pi_mcp_setup_script(
    state: State,
    agent_workdir: str,
    install_mcp_adapter: bool,
) -> str:
    return build_pi_mcp_setup_script(
        agent_workdir=agent_workdir,
        endpoint_config=pi_endpoint_config(state),
        install_mcp_adapter=install_mcp_adapter,
    )


def build_pi_mcp_setup_script(
    *,
    agent_workdir: str,
    endpoint_config: ConfigMap,
    install_mcp_adapter: bool,
) -> str:
    models_json = pi_models_json(endpoint_config)
    mcp_json = pi_mcp_json() if install_mcp_adapter else None
    install_adapter = "pi install npm:pi-mcp-adapter -l" if install_mcp_adapter else ""
    mcp_write = ""
    if mcp_json is not None:
        mcp_write = f"""\
cat > "$PI_WORKDIR/.mcp.json" <<'EOFMCP'
{mcp_json}
EOFMCP
"""
    return f"""\
set -e

PI_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$PI_WORKDIR" ]; then
    PI_WORKDIR={shlex.quote(agent_workdir)}
fi

mkdir -p "$HOME/.pi/agent" "$PI_WORKDIR"
cat > "$HOME/.pi/agent/models.json" <<'EOFMODELS'
{models_json}
EOFMODELS
{mcp_write}
cd "$PI_WORKDIR"
{install_adapter}
"""


def build_pi_run_script(
    *,
    agent_workdir: str,
    instruction_path: str,
    system_prompt_path: str | None,
    log_path: str,
) -> str:
    log_dir = str(PurePosixPath(log_path).parent)
    system_prompt_arg = (
        f'--system-prompt "$(cat {shlex.quote(system_prompt_path)})"'
        if system_prompt_path is not None
        else ""
    )
    return f"""\
set -eo pipefail

PI_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$PI_WORKDIR" ]; then
    PI_WORKDIR={shlex.quote(agent_workdir)}
fi

mkdir -p {shlex.quote(log_dir)} "$PI_WORKDIR"
cd "$PI_WORKDIR"
pi --no-session --no-context-files --provider verifiers --model model \
  {system_prompt_arg} -p @{shlex.quote(instruction_path)} 2>&1 | tee {shlex.quote(log_path)}
"""


def pi_endpoint_config(state: State) -> dict[str, str]:
    return state.get_endpoint_config(api="chat")


def pi_models_json(endpoint_config: ConfigMap) -> str:
    api = str(endpoint_config.get("api_client_type") or "openai_chat_completions")
    api_name = {
        "openai_chat_completions": "openai-completions",
        "openai_responses": "openai-responses",
        "anthropic_messages": "anthropic-messages",
    }.get(api, "openai-completions")
    config = {
        "providers": {
            "verifiers": {
                "baseUrl": str(endpoint_config["base_url"]),
                "api": api_name,
                "apiKey": str(endpoint_config["api_key"]),
                "models": [
                    {
                        "id": "model",
                        "name": str(endpoint_config["model"]),
                    }
                ],
            }
        }
    }
    return json.dumps(config, indent=2)


def pi_mcp_json() -> str:
    command, *args = proxy_command()
    config = {
        "mcpServers": {
            "verifiers-tools": {
                "command": command,
                "args": args,
                "lifecycle": "lazy",
            }
        }
    }
    return json.dumps(config, indent=2)
