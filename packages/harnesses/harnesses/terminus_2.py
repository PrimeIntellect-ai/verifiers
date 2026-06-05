import shlex
from pathlib import PurePosixPath

import verifiers as vf
from verifiers.v1.utils.sandbox_python_utils import SANDBOX_BIN_DIR, uv_setup_command

TERMINUS_2_DEFAULT_AGENT_WORKDIR = "/app"
TERMINUS_2_DEFAULT_INSTRUCTION_PATH = "/terminus_2/instruction.md"
TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH = "/terminus_2/system_prompt.txt"
TERMINUS_2_DEFAULT_LOG_PATH = "/logs/agent/terminus_2.log"
TERMINUS_2_DEFAULT_VERSION = "harbor==0.6.6"
TERMINUS_2_DEFAULT_PYTHON_VERSION = "3.12"
TERMINUS_2_DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"
TERMINUS_2_DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"


class Terminus2ProgramConfig(vf.ProgramConfig):
    agent_workdir: str = TERMINUS_2_DEFAULT_AGENT_WORKDIR
    instruction_path: str = TERMINUS_2_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = TERMINUS_2_DEFAULT_LOG_PATH
    python_version: str = TERMINUS_2_DEFAULT_PYTHON_VERSION
    model_name: str = TERMINUS_2_DEFAULT_MODEL_NAME
    api_base_url: str = TERMINUS_2_DEFAULT_API_BASE_URL
    sandbox: vf.SandboxConfig | None = vf.SandboxConfig()
    max_turns: int = 4

    def resolve(self, version: str = TERMINUS_2_DEFAULT_VERSION) -> vf.ProgramConfig:
        files: dict[str, vf.ProgramValue] = {
            self.instruction_path: {"fn": "verifiers.v1.utils.prompt_utils:task_text"},
            self.system_prompt_path: {
                "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
            },
        }
        artifacts = vf.ArtifactsConfig.model_validate(
            {
                "terminus_2_log": {
                    "path": self.log_path,
                    "format": "text",
                    "optional": True,
                }
            }
        )
        log_dir = str(PurePosixPath(self.log_path).parent)
        system_prompt_block = f"""\
    system_prompt_path = Path({self.system_prompt_path!r})
    if system_prompt_path.exists() and system_prompt_path.stat().st_size > 0:
        instruction = system_prompt_path.read_text() + "\\n\\n" + instruction
"""
        agent_script = f"""\
import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.paths import TrialPaths


class LocalEnvironment(BaseEnvironment):
    def __init__(self, workdir: Path, logs_dir: Path):
        self.workdir = workdir
        self.trial_paths = TrialPaths(trial_dir=logs_dir)
        self.trial_paths.mkdir()
        self.default_user = None
        self.session_id = "local"
        self.logger = logging.getLogger(__name__)

    def type(self) -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    def _validate_definition(self) -> None:
        return None

    async def start(self, force_build: bool) -> None:
        return None

    async def stop(self, delete: bool) -> None:
        return None

    async def prepare_logs_for_host(self) -> None:
        return None

    async def upload_file(self, source_path, target_path):
        shutil.copy(source_path, target_path)

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def download_file(self, source_path, target_path):
        shutil.copy(source_path, target_path)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = user
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or str(self.workdir),
                env={{**os.environ, **(env or {{}})}},
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="Command timed out", return_code=124)
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )


async def main() -> None:
    workdir = Path(os.environ.get("AGENT_WORKDIR") or {TERMINUS_2_DEFAULT_AGENT_WORKDIR!r})
    logs_dir = Path({log_dir!r})
    instruction = Path({self.instruction_path!r}).read_text()
{system_prompt_block}    env = LocalEnvironment(workdir=workdir, logs_dir=logs_dir)
    api_base = os.environ.get("OPENAI_BASE_URL") or {self.api_base_url!r}
    agent = Terminus2(
        logs_dir=logs_dir,
        model_name={self.model_name!r},
        api_base=api_base,
        max_turns={self.max_turns!r},
    )
    await agent.setup(env)
    await agent.run(instruction, env, AgentContext())


asyncio.run(main())
"""
        run_script = f"""\
set -eo pipefail
export PATH={shlex.quote(SANDBOX_BIN_DIR)}:"$HOME/.local/bin:$PATH"

TERMINUS_2_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$TERMINUS_2_WORKDIR" ]; then
    TERMINUS_2_WORKDIR={shlex.quote(self.agent_workdir)}
fi
export AGENT_WORKDIR="$TERMINUS_2_WORKDIR"

mkdir -p {shlex.quote(log_dir)} "$TERMINUS_2_WORKDIR"
cd "$TERMINUS_2_WORKDIR"
uv --no-config run --no-project --quiet \
  --python {shlex.quote(self.python_version)} \
  --with {shlex.quote(version)} \
  python - <<'PY' 2>&1 | tee -a {shlex.quote(self.log_path)}
{agent_script}
PY
"""
        return self.resolve_command(
            command=["bash", "-lc", run_script],
            default_sandbox=self.sandbox,
            files=files,
            setup=uv_setup_command(),
            artifacts=artifacts,
        )


class Terminus2Config(vf.HarnessConfig):
    version: str = TERMINUS_2_DEFAULT_VERSION
    program: Terminus2ProgramConfig = Terminus2ProgramConfig()


class Terminus2(vf.Harness[Terminus2Config]):
    config: Terminus2Config

    def load_program_config(self, config: Terminus2Config) -> vf.ProgramConfig:
        return config.program.resolve(version=config.version)


def load_harness(config: Terminus2Config) -> Terminus2:
    return Terminus2(config=config)
