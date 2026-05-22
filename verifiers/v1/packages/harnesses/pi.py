import json
import shlex
from pathlib import PurePosixPath
from typing import cast

from .command import configure_command_harness
from .configs import PiConfig
from ...harness import Harness
from ...state import State
from ...utils.mcp_proxy_utils import proxy_command
from ...utils.binding_utils import Bindings
from ...types import ConfigMap, ProgramChannels, ProgramCommand, ProgramOptionMap


class Pi(Harness[PiConfig]):
    def __init__(self, config: PiConfig | None = None):
        config = cast(PiConfig, self._coerce_config(config))
        super().__init__(config=config.model_copy(update={"program": None}))
        self.config = config
        configure_command_harness(
            self,
            config,
            command=self.command(config),
            setup=self.setup(config),
            bindings=self.bindings_value(config),
            artifacts=self.artifacts(config),
            channels=self.channels(config),
        )

    def command(self, config: PiConfig) -> ProgramCommand:
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

    def setup(self, config: PiConfig) -> str:
        return f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates nodejs npm > /dev/null 2>&1
npm install -g {shlex.quote(config.package)}
"""

    def artifacts(self, config: PiConfig) -> ProgramOptionMap:
        return {
            "pi_log": {
                "path": config.log_path,
                "format": "text",
                "optional": True,
            }
        }

    def channels(self, config: PiConfig) -> ProgramChannels | None:
        if not config.install_mcp_adapter:
            return None
        return {
            "mcp": build_pi_mcp_setup(
                agent_workdir=config.agent_workdir,
                install_mcp_adapter=config.install_mcp_adapter,
            )
        }

    def bindings_value(self, config: PiConfig) -> Bindings:
        return {"setup_pi.endpoint_config": pi_endpoint_config}


def build_pi_mcp_setup(
    *,
    agent_workdir: str,
    install_mcp_adapter: bool,
):
    def setup_pi(endpoint_config) -> str:
        return build_pi_mcp_setup_script(
            agent_workdir=agent_workdir,
            endpoint_config=endpoint_config,
            install_mcp_adapter=install_mcp_adapter,
        )

    return setup_pi


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
if [[ -z "$PI_WORKDIR" ]]; then
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
if [[ -z "$PI_WORKDIR" ]]; then
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
