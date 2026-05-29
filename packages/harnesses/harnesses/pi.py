import json
import shlex
from pathlib import PurePosixPath

import verifiers as vf
from verifiers.v1.utils.mcp_proxy_utils import proxy_command

PI_DEFAULT_PACKAGE = "@earendil-works/pi-coding-agent"
PI_DEFAULT_WORKDIR = "/app"
PI_DEFAULT_INSTRUCTION_PATH = "/pi/instruction.txt"
PI_DEFAULT_SYSTEM_PROMPT_PATH = "/pi/system.txt"
PI_DEFAULT_LOG_PATH = "/logs/agent/pi.txt"
PI_DEFAULT_SYSTEM_PROMPT = "Complete the user's task using the available tools."


class PiProgramConfig(vf.ProgramConfig):
    agent_workdir: str = PI_DEFAULT_WORKDIR
    instruction_path: str = PI_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = PI_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = PI_DEFAULT_LOG_PATH
    package: str = PI_DEFAULT_PACKAGE
    install_mcp_adapter: bool = True
    sandbox: vf.SandboxConfig | None = vf.SandboxConfig()

    def resolve(self) -> vf.ProgramConfig:
        files: dict[str, vf.ProgramValue] = {
            self.instruction_path: {"fn": "verifiers.v1.utils.prompt_utils:task_text"},
            self.system_prompt_path: {
                "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
            },
        }
        channels: dict[str, vf.ProgramValue] | None = None
        if self.install_mcp_adapter:
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
    PI_WORKDIR={shlex.quote(self.agent_workdir)}
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
npm install -g --ignore-scripts {shlex.quote(self.package)}
"""
        artifacts = vf.ArtifactsConfig.model_validate(
            {
                "pi_log": {
                    "path": self.log_path,
                    "format": "text",
                    "optional": True,
                }
            }
        )
        log_dir = str(PurePosixPath(self.log_path).parent)
        system_prompt_path = shlex.quote(self.system_prompt_path)
        run_script = f"""\
set -eo pipefail

PI_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$PI_WORKDIR" ]; then
    PI_WORKDIR={shlex.quote(self.agent_workdir)}
fi

mkdir -p {shlex.quote(log_dir)} "$PI_WORKDIR"
cd "$PI_WORKDIR"
SYSTEM_PROMPT_ARGS=()
if [ -s {system_prompt_path} ]; then
  SYSTEM_PROMPT_ARGS=(--system-prompt "$(cat {system_prompt_path})")
fi
pi --no-session --no-context-files --provider verifiers --model model \
  "${{SYSTEM_PROMPT_ARGS[@]}}" -p @{shlex.quote(self.instruction_path)} 2>&1 | tee {shlex.quote(self.log_path)}
"""
        return self.resolve_command(
            command=["bash", "-lc", run_script],
            default_sandbox=self.sandbox,
            files=files,
            setup=setup,
            env={"OPENAI_MODEL": "runtime.model"},
            artifacts=artifacts,
            channels=channels,
        )


class PiConfig(vf.HarnessConfig):
    system_prompt: vf.PromptInput | vf.SystemPromptConfig | None = (
        PI_DEFAULT_SYSTEM_PROMPT
    )
    program: PiProgramConfig = PiProgramConfig()
    max_turns: int = 4


class Pi(vf.Harness[PiConfig]):
    config: PiConfig


def load_harness(config: PiConfig) -> Pi:
    return Pi(config=config)
