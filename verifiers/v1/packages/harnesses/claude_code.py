import json
import shlex
from pathlib import PurePosixPath
from typing import ClassVar, cast

from .command import configure_command_harness
from .configs import ClaudeCodeConfig
from ...harness import Harness
from ...types import ProgramCommand, ProgramChannels, ProgramOptionMap
from ...utils.mcp_proxy_utils import proxy_command


class ClaudeCode(Harness[ClaudeCodeConfig]):
    _config_aliases: ClassVar[tuple[str, ...]] = ("claude", "claude-code")

    def __init__(self, config: ClaudeCodeConfig | None = None):
        config = cast(ClaudeCodeConfig, self._coerce_config(config))
        super().__init__(config=config.model_copy(update={"program": None}))
        self.config = config
        configure_command_harness(
            self,
            config,
            command=self.command(config),
            setup=self.setup(config),
            env=self.env(config),
            artifacts=self.artifacts(config),
            channels=self.channels(config),
        )

    def command(self, config: ClaudeCodeConfig) -> ProgramCommand:
        return [
            "bash",
            "-lc",
            build_claude_code_run_script(
                agent_workdir=config.agent_workdir,
                instruction_path=config.instruction_path,
                system_prompt_path=config.system_prompt_path
                if config.system_prompt is not None
                else None,
                log_path=config.log_path,
                permission_mode=config.permission_mode,
                max_turns=config.max_turns,
            ),
        ]

    def setup(self, config: ClaudeCodeConfig) -> str:
        return build_claude_code_install_script(package=config.package)

    def env(self, config: ClaudeCodeConfig) -> ProgramOptionMap:
        return {
            "ANTHROPIC_MODEL": "runtime.model",
            "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
        }

    def artifacts(self, config: ClaudeCodeConfig) -> ProgramOptionMap:
        return {
            "claude_code_log": {
                "path": config.log_path,
                "format": "text",
                "optional": True,
            }
        }

    def channels(self, config: ClaudeCodeConfig) -> ProgramChannels:
        return "mcp"


def build_claude_code_install_script(package: str) -> str:
    return f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates nodejs npm > /dev/null 2>&1
npm install -g {shlex.quote(package)}
"""


def build_claude_code_run_script(
    *,
    agent_workdir: str,
    instruction_path: str,
    system_prompt_path: str | None,
    log_path: str,
    permission_mode: str,
    max_turns: int,
) -> str:
    log_dir = str(PurePosixPath(log_path).parent)
    final_path = f"{log_path}.final"
    mcp_config_path = "/tmp/claude-code-mcp.json"
    mcp_config = claude_code_mcp_json()
    system_prompt_arg = (
        f'--append-system-prompt "$(cat {shlex.quote(system_prompt_path)})"'
        if system_prompt_path is not None
        else ""
    )
    return f"""\
set -eo pipefail

CLAUDE_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$CLAUDE_WORKDIR" ]; then
    CLAUDE_WORKDIR={shlex.quote(agent_workdir)}
fi

mkdir -p {shlex.quote(log_dir)} "$CLAUDE_WORKDIR"
cat > {shlex.quote(mcp_config_path)} <<'EOFMCP'
{mcp_config}
EOFMCP

cd "$CLAUDE_WORKDIR"
cat {shlex.quote(instruction_path)} | claude -p \
  --model "$ANTHROPIC_MODEL" \
  --max-turns {int(max_turns)} \
  --permission-mode {shlex.quote(permission_mode)} \
  --mcp-config {shlex.quote(mcp_config_path)} \
  --output-format text \
  {system_prompt_arg} > {shlex.quote(final_path)} 2> {shlex.quote(log_path)}
cat {shlex.quote(final_path)} >> {shlex.quote(log_path)}
cat {shlex.quote(final_path)}
"""


def claude_code_mcp_json() -> str:
    command, *args = proxy_command()
    config = {
        "mcpServers": {
            "verifiers-tools": {
                "type": "stdio",
                "command": command,
                "args": args,
            }
        }
    }
    return json.dumps(config, indent=2)
