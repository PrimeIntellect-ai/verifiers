import shlex
from pathlib import PurePosixPath

import verifiers as vf
from pydantic import model_validator
from verifiers.v1.utils.sandbox_python_utils import SANDBOX_BIN_DIR, uv_setup_command

TERMINUS_2_DEFAULT_AGENT_WORKDIR = "/app"
TERMINUS_2_DEFAULT_INSTRUCTION_PATH = "/terminus_2/instruction.md"
TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH = "/terminus_2/system_prompt.txt"
TERMINUS_2_DEFAULT_LOG_PATH = "/logs/agent/terminus_2.log"
TERMINUS_2_DEFAULT_HARBOR_PACKAGE = "harbor==0.6.6"
TERMINUS_2_DEFAULT_PYTHON_VERSION = "3.12"
TERMINUS_2_DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"
TERMINUS_2_DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"


class Terminus2Config(vf.HarnessConfig):
    agent_workdir: str = TERMINUS_2_DEFAULT_AGENT_WORKDIR
    instruction_path: str = TERMINUS_2_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = TERMINUS_2_DEFAULT_LOG_PATH
    harbor_package: str = TERMINUS_2_DEFAULT_HARBOR_PACKAGE
    python_version: str = TERMINUS_2_DEFAULT_PYTHON_VERSION
    model_name: str = TERMINUS_2_DEFAULT_MODEL_NAME
    api_base_url: str = TERMINUS_2_DEFAULT_API_BASE_URL
    system_prompt: vf.PromptInput | None = None
    sandbox: vf.SandboxConfig | None = vf.SandboxConfig()
    max_turns: int = 4

    @model_validator(mode="after")
    def configure_program(self) -> "Terminus2Config":
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
        artifacts: dict[str, vf.ProgramValue] = {
            "terminus_2_log": {
                "path": config.log_path,
                "format": "text",
                "optional": True,
            }
        }
        log_dir = str(PurePosixPath(config.log_path).parent)
        system_prompt_path = (
            config.system_prompt_path if config.system_prompt is not None else None
        )
        system_prompt_block = ""
        if system_prompt_path is not None:
            system_prompt_block = f"""\
    system_prompt_path = Path({system_prompt_path!r})
    if system_prompt_path.exists() and system_prompt_path.stat().st_size > 0:
        instruction = system_prompt_path.read_text() + "\\n\\n" + instruction
"""
        agent_script = f"""\
from __future__ import annotations

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

    @staticmethod
    def type() -> EnvironmentType:
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

    def _validate_definition(self):
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool):
        pass

    async def prepare_logs_for_host(self) -> None:
        pass

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
        del user
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
    instruction = Path({config.instruction_path!r}).read_text()
{system_prompt_block}    env = LocalEnvironment(workdir=workdir, logs_dir=logs_dir)
    api_base = os.environ.get("OPENAI_BASE_URL") or {config.api_base_url!r}
    agent = Terminus2(
        logs_dir=logs_dir,
        model_name={config.model_name!r},
        api_base=api_base,
        max_turns={config.max_turns!r},
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
    TERMINUS_2_WORKDIR={shlex.quote(config.agent_workdir)}
fi
export AGENT_WORKDIR="$TERMINUS_2_WORKDIR"

mkdir -p {shlex.quote(log_dir)} "$TERMINUS_2_WORKDIR"
cd "$TERMINUS_2_WORKDIR"
uv --no-config run --no-project --quiet \
  --python {shlex.quote(config.python_version)} \
  --with {shlex.quote(config.harbor_package)} \
  python - <<'PY' 2>&1 | tee -a {shlex.quote(config.log_path)}
{agent_script}
PY
"""
        self.program = vf.ProgramConfig.from_command(
            command=["bash", "-lc", run_script],
            program=config.program,
            default_sandbox=config.sandbox,
            files=files,
            setup=uv_setup_command(),
            artifacts=artifacts,
        )
        self.model_fields_set.discard("program")
        return self


class Terminus2(vf.Harness[Terminus2Config]):
    config: Terminus2Config


def load_harness(config: Terminus2Config) -> Terminus2:
    return Terminus2(config=config)
