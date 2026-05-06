from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .cli import CLIHarness
from ...config import SandboxConfig
from ...state import State
from ...utils.mcp_proxy_utils import proxy_command
from ...utils.prompt_utils import (
    state_system_prompt_text,
    task_text as task_instruction_text,
)

DEFAULT_PI_PACKAGE = "@mariozechner/pi-coding-agent"
DEFAULT_PI_WORKDIR = "/app"
DEFAULT_INSTRUCTION_PATH = "/pi/instruction.txt"
DEFAULT_SYSTEM_PROMPT_PATH = "/pi/system.txt"
DEFAULT_LOG_PATH = "/logs/agent/pi.txt"
DEFAULT_SYSTEM_PROMPT = "Complete the user's task using the available tools."


class Pi(CLIHarness):
    def __init__(
        self,
        *,
        agent_workdir: str = DEFAULT_PI_WORKDIR,
        instruction_path: str = DEFAULT_INSTRUCTION_PATH,
        system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
        log_path: str = DEFAULT_LOG_PATH,
        system_prompt: object | None = DEFAULT_SYSTEM_PROMPT,
        package: str = DEFAULT_PI_PACKAGE,
        install_mcp_adapter: bool = True,
        sandbox: bool | Mapping[str, object] | SandboxConfig = True,
        program: Mapping[str, object] | None = None,
        max_turns: int | None = 4,
        **kwargs: Any,
    ):
        files: dict[str, object] = {
            instruction_path: task_instruction_text,
        }
        if system_prompt is not None:
            files[system_prompt_path] = state_system_prompt_text
        artifacts = {
            "pi_log": {
                "path": log_path,
                "format": "text",
                "optional": True,
            }
        }
        super().__init__(
            command=[
                "bash",
                "-lc",
                build_pi_run_script(
                    agent_workdir=agent_workdir,
                    instruction_path=instruction_path,
                    system_prompt_path=system_prompt_path
                    if system_prompt is not None
                    else None,
                    log_path=log_path,
                ),
            ],
            sandbox=sandbox,
            files=files,
            setup=build_pi_install_script(package=package),
            tools={
                "mcp": build_pi_mcp_setup(
                    agent_workdir=agent_workdir,
                    install_mcp_adapter=install_mcp_adapter,
                )
            }
            if install_mcp_adapter
            else None,
            bindings={"setup_pi.endpoint_config": pi_endpoint_config},
            artifacts=artifacts,
            program=program,
            system_prompt=system_prompt,
            max_turns=max_turns,
            **kwargs,
        )


def build_pi_install_script(package: str = DEFAULT_PI_PACKAGE) -> str:
    return f"""\
set -e
apt-get update -qq && apt-get install -y -qq curl ca-certificates nodejs npm > /dev/null 2>&1
npm install -g {shlex.quote(package)}
"""


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
    endpoint_config: Mapping[str, object],
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


def pi_models_json(endpoint_config: Mapping[str, object]) -> str:
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
