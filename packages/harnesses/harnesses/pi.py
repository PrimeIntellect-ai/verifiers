import json
import shlex
from pathlib import PurePosixPath

import verifiers as vf
from pydantic import model_validator
from verifiers.v1.utils.mcp_proxy_utils import proxy_command

PI_DEFAULT_PACKAGE = "@earendil-works/pi-coding-agent"
PI_DEFAULT_WORKDIR = "/app"
PI_DEFAULT_INSTRUCTION_PATH = "/pi/instruction.txt"
PI_DEFAULT_SYSTEM_PROMPT_PATH = "/pi/system.txt"
PI_DEFAULT_LOG_PATH = "/logs/agent/pi.txt"
PI_DEFAULT_SYSTEM_PROMPT = "Complete the user's task using the available tools."


class PiConfig(vf.HarnessConfig):
    agent_workdir: str = PI_DEFAULT_WORKDIR
    instruction_path: str = PI_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = PI_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = PI_DEFAULT_LOG_PATH
    system_prompt: vf.PromptInput | None = PI_DEFAULT_SYSTEM_PROMPT
    package: str = PI_DEFAULT_PACKAGE
    install_mcp_adapter: bool = True
    sandbox: vf.SandboxConfig | None = vf.SandboxConfig()
    max_turns: int = 4

    @model_validator(mode="after")
    def configure_program(self) -> "PiConfig":
        if self.program.command is not None and "program" not in self.model_fields_set:
            return self
        config = self
        files: dict[str, vf.ProgramValue] = {
            config.instruction_path: {
                "fn": "verifiers.v1.utils.prompt_utils:task_text"
            },
        }
        if config.system_prompt is not None:
            files[config.system_prompt_path] = {
                "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
            }
        channels: vf.ProgramOptionMap | None = None
        if config.install_mcp_adapter:
            command, *args = proxy_command()
            mcp_json = json.dumps(
                {
                    "mcpServers": {
                        "verifiers-tools": {
                            "command": command,
                            "args": args,
                            "lifecycle": "lazy",
                        }
                    }
                },
                indent=2,
            )
            models_json = """\
{
  "providers": {
    "verifiers": {
      "baseUrl": "${OPENAI_BASE_URL}",
      "api": "openai-completions",
      "apiKey": "${OPENAI_API_KEY:-intercepted}",
      "models": [{"id": "model", "name": "${OPENAI_MODEL}"}]
    }
  }
}
"""
            mcp_setup = f"""\
set -e

PI_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$PI_WORKDIR" ]; then
    PI_WORKDIR={shlex.quote(config.agent_workdir)}
fi

mkdir -p "$HOME/.pi/agent" "$PI_WORKDIR"
cat > "$HOME/.pi/agent/models.json" <<EOFMODELS
{models_json}EOFMODELS
cat > "$PI_WORKDIR/.mcp.json" <<'EOFMCP'
{mcp_json}
EOFMCP
cd "$PI_WORKDIR"
pi install npm:pi-mcp-adapter -l
"""
            channels = {
                "mcp": mcp_setup,
            }
        setup = f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates nodejs npm xz-utils > /dev/null 2>&1
npm install -g --ignore-scripts n
n 22.19.0
hash -r
npm install -g --ignore-scripts {shlex.quote(config.package)}
"""
        artifacts: dict[str, vf.ProgramValue] = {
            "pi_log": {
                "path": config.log_path,
                "format": "text",
                "optional": True,
            }
        }
        system_prompt_path = (
            config.system_prompt_path if config.system_prompt is not None else None
        )
        log_dir = str(PurePosixPath(config.log_path).parent)
        system_prompt_arg = (
            f'--system-prompt "$(cat {shlex.quote(system_prompt_path)})"'
            if system_prompt_path is not None
            else ""
        )
        run_script = f"""\
set -eo pipefail

PI_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$PI_WORKDIR" ]; then
    PI_WORKDIR={shlex.quote(config.agent_workdir)}
fi

mkdir -p {shlex.quote(log_dir)} "$PI_WORKDIR"
cd "$PI_WORKDIR"
pi --no-session --no-context-files --provider verifiers --model model \
  {system_prompt_arg} -p @{shlex.quote(config.instruction_path)} 2>&1 | tee {shlex.quote(config.log_path)}
"""
        self.program = vf.ProgramConfig.from_command(
            command=["bash", "-lc", run_script],
            program=config.program,
            default_sandbox=config.sandbox,
            files=files,
            setup=setup,
            env={"OPENAI_MODEL": "runtime.model"},
            artifacts=artifacts,
            channels=channels,
        )
        self.model_fields_set.discard("program")
        return self


class Pi(vf.Harness[PiConfig]):
    config: PiConfig


def load_harness(config: PiConfig) -> Pi:
    return Pi(config=config)
