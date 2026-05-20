import json
import shlex
from pathlib import PurePosixPath
from typing import ClassVar, cast

from .command import configure_command_harness
from .configs import CodexConfig
from ...harness import Harness
from ...state import State
from ...types import ProgramCommand, ProgramChannels, ProgramOptionMap
from ...utils.mcp_proxy_utils import proxy_command


class Codex(Harness[CodexConfig]):
    _config_aliases: ClassVar[tuple[str, ...]] = ("codex", "codex-cli")

    def __init__(self, config: CodexConfig | None = None):
        config = cast(CodexConfig, self._coerce_config(config))
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

    def command(self, config: CodexConfig) -> ProgramCommand:
        return [
            "bash",
            "-lc",
            build_codex_run_script(
                agent_workdir=config.agent_workdir,
                instruction_path=config.instruction_path,
                system_prompt_path=config.system_prompt_path
                if config.system_prompt is not None
                else None,
                log_path=config.log_path,
                codex_sandbox=config.codex_sandbox,
                model_reasoning_effort=config.model_reasoning_effort,
            ),
        ]

    def setup(self, config: CodexConfig) -> str:
        return build_codex_install_script(package=config.package)

    def env(self, config: CodexConfig) -> ProgramOptionMap:
        return {
            "OPENAI_MODEL": "runtime.model",
            "CODEX_API_KEY": codex_api_key,
            "DISABLE_TELEMETRY": "1",
        }

    def artifacts(self, config: CodexConfig) -> ProgramOptionMap:
        return {
            "codex_log": {
                "path": config.log_path,
                "format": "text",
                "optional": True,
            }
        }

    def channels(self, config: CodexConfig) -> ProgramChannels:
        return "mcp"


def build_codex_install_script(package: str) -> str:
    return f"""\
set -e
apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates nodejs npm > /dev/null 2>&1
npm install -g {shlex.quote(package)}
"""


def build_codex_run_script(
    *,
    agent_workdir: str,
    instruction_path: str,
    system_prompt_path: str | None,
    log_path: str,
    codex_sandbox: str,
    model_reasoning_effort: str | None,
) -> str:
    log_dir = str(PurePosixPath(log_path).parent)
    final_path = f"{log_path}.final"
    prompt_path = f"{log_path}.prompt"
    effort_config = (
        f"model_reasoning_effort = {json.dumps(model_reasoning_effort)}\n"
        if model_reasoning_effort is not None
        else ""
    )
    mcp_toml = codex_mcp_toml()
    system_prompt = (
        f"cat {shlex.quote(system_prompt_path)} > {shlex.quote(prompt_path)}\n"
        f"printf '\\n\\n' >> {shlex.quote(prompt_path)}"
        if system_prompt_path is not None
        else f": > {shlex.quote(prompt_path)}"
    )
    return f"""\
set -eo pipefail

CODEX_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$CODEX_WORKDIR" ]; then
    CODEX_WORKDIR={shlex.quote(agent_workdir)}
fi

mkdir -p {shlex.quote(log_dir)} "$CODEX_WORKDIR" "$CODEX_WORKDIR/.codex"
cat > "$CODEX_WORKDIR/.codex/config.toml" <<EOFCODEX
model = "model"
model_provider = "verifiers"
approval_policy = "never"
sandbox_mode = {json.dumps(codex_sandbox)}
{effort_config}[model_providers.verifiers]
name = "Verifiers"
base_url = "$OPENAI_BASE_URL"
env_key = "CODEX_API_KEY"
wire_api = "responses"

{mcp_toml}
EOFCODEX
{system_prompt}
cat {shlex.quote(instruction_path)} >> {shlex.quote(prompt_path)}

cd "$CODEX_WORKDIR"
CODEX_HOME="$CODEX_WORKDIR/.codex" codex exec \
  --skip-git-repo-check \
  --sandbox {shlex.quote(codex_sandbox)} \
  --model "$OPENAI_MODEL" \
  --output-last-message {shlex.quote(final_path)} \
  - < {shlex.quote(prompt_path)} > {shlex.quote(log_path)} 2>&1
cat {shlex.quote(final_path)}
"""


def codex_mcp_toml() -> str:
    command, *args = proxy_command()
    return (
        "[mcp_servers.verifiers-tools]\n"
        f"command = {json.dumps(command)}\n"
        f"args = {json.dumps(args)}\n"
    )


def codex_api_key(state: State) -> str:
    return state.get_endpoint_config(api="responses")["api_key"]
