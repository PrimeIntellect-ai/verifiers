import shlex
from pathlib import PurePosixPath

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.program import Program, ProgramCommand, ProgramOptionMap, ProgramValue
from verifiers.v1.sandbox import SandboxConfig
from verifiers.v1.types import (
    ConfigData,
    ConfigMap,
    PromptInput,
)
from verifiers.v1.utils.sandbox_python_utils import SANDBOX_BIN_DIR, uv_setup_command

TERMINUS_2_DEFAULT_AGENT_WORKDIR = "/app"
TERMINUS_2_DEFAULT_INSTRUCTION_PATH = "/terminus_2/instruction.md"
TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH = "/terminus_2/system_prompt.txt"
TERMINUS_2_DEFAULT_LOG_PATH = "/logs/agent/terminus_2.log"
TERMINUS_2_DEFAULT_HARBOR_PACKAGE = "harbor==0.6.6"
TERMINUS_2_DEFAULT_PYTHON_VERSION = "3.12"
TERMINUS_2_DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"
TERMINUS_2_DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"


class Terminus2Config(HarnessConfig):
    agent_workdir: str = TERMINUS_2_DEFAULT_AGENT_WORKDIR
    instruction_path: str = TERMINUS_2_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = TERMINUS_2_DEFAULT_LOG_PATH
    harbor_package: str = TERMINUS_2_DEFAULT_HARBOR_PACKAGE
    python_version: str = TERMINUS_2_DEFAULT_PYTHON_VERSION
    model_name: str = TERMINUS_2_DEFAULT_MODEL_NAME
    api_base_url: str = TERMINUS_2_DEFAULT_API_BASE_URL
    system_prompt: PromptInput | None = None
    sandbox: SandboxConfig | None = SandboxConfig()
    max_turns: int = 4


class Terminus2(Harness[Terminus2Config]):
    config: Terminus2Config

    def load_program(self) -> Program:
        program, _ = terminus_2_program_config(self.config)
        return program

    def load_sandbox(self) -> ConfigMap | None:
        _, sandbox = terminus_2_program_config(self.config)
        return sandbox


def load_harness(config: Terminus2Config) -> Terminus2:
    return Terminus2(config=config)


def terminus_2_program_config(
    config: Terminus2Config,
) -> tuple[Program, ConfigData | None]:
    return Harness.command_program_config(
        config,
        command=terminus_2_command(config),
        files=terminus_2_files(config),
        setup=terminus_2_setup(config),
        artifacts=terminus_2_artifacts(config),
    )


def terminus_2_command(config: Terminus2Config) -> ProgramCommand:
    return [
        "bash",
        "-lc",
        build_terminus_2_run_script(
            agent_workdir=config.agent_workdir,
            instruction_path=config.instruction_path,
            system_prompt_path=config.system_prompt_path
            if config.system_prompt is not None
            else None,
            log_path=config.log_path,
            harbor_package=config.harbor_package,
            python_version=config.python_version,
            model_name=config.model_name,
            api_base_url=config.api_base_url,
            max_turns=config.max_turns,
        ),
    ]


def terminus_2_setup(config: Terminus2Config) -> str:
    _ = config
    return uv_setup_command()


def terminus_2_files(config: Terminus2Config) -> ProgramOptionMap:
    files: dict[str, ProgramValue] = {
        config.instruction_path: {"fn": "verifiers.v1.utils.prompt_utils:task_text"},
    }
    if config.system_prompt is not None:
        files[config.system_prompt_path] = {
            "fn": "verifiers.v1.utils.prompt_utils:state_system_prompt_text"
        }
    return files


def terminus_2_artifacts(config: Terminus2Config) -> ProgramOptionMap:
    return {
        "terminus_2_log": {
            "path": config.log_path,
            "format": "text",
            "optional": True,
        }
    }


def build_terminus_2_run_script(
    *,
    agent_workdir: str,
    instruction_path: str,
    system_prompt_path: str | None,
    log_path: str,
    harbor_package: str,
    python_version: str,
    model_name: str,
    api_base_url: str,
    max_turns: int | None,
) -> str:
    log_dir = str(PurePosixPath(log_path).parent)
    agent_script = terminus_2_agent_script(
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_dir=log_dir,
        model_name=model_name,
        api_base_url=api_base_url,
        max_turns=max_turns,
    )
    return f"""\
set -eo pipefail
export PATH={shlex.quote(SANDBOX_BIN_DIR)}:"$HOME/.local/bin:$PATH"

TERMINUS_2_WORKDIR="${{AGENT_WORKDIR:-}}"
if [ -z "$TERMINUS_2_WORKDIR" ]; then
    TERMINUS_2_WORKDIR={shlex.quote(agent_workdir)}
fi
export AGENT_WORKDIR="$TERMINUS_2_WORKDIR"

mkdir -p {shlex.quote(log_dir)} "$TERMINUS_2_WORKDIR"
cd "$TERMINUS_2_WORKDIR"
uv --no-config run --no-project --quiet \
  --python {shlex.quote(python_version)} \
  --with {shlex.quote(harbor_package)} \
  python - <<'PY' 2>&1 | tee -a {shlex.quote(log_path)}
{agent_script}
PY
"""


def terminus_2_agent_script(
    *,
    instruction_path: str = TERMINUS_2_DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str | None = TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH,
    log_dir: str = "/logs/agent",
    model_name: str = TERMINUS_2_DEFAULT_MODEL_NAME,
    api_base_url: str = TERMINUS_2_DEFAULT_API_BASE_URL,
    max_turns: int | None = 4,
) -> str:
    system_prompt_block = ""
    if system_prompt_path is not None:
        system_prompt_block = f"""\
    system_prompt_path = Path({system_prompt_path!r})
    if system_prompt_path.exists() and system_prompt_path.stat().st_size > 0:
        instruction = system_prompt_path.read_text() + "\\n\\n" + instruction
"""
    return f"""\
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
    instruction = Path({instruction_path!r}).read_text()
{system_prompt_block}    env = LocalEnvironment(workdir=workdir, logs_dir=logs_dir)
    if "OPENAI_API_KEY" not in os.environ and "PRIME_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["PRIME_API_KEY"]
    api_base = os.environ.get("OPENAI_BASE_URL") or {api_base_url!r}
    agent = Terminus2(
        logs_dir=logs_dir,
        model_name={model_name!r},
        api_base=api_base,
        max_turns={max_turns!r},
    )
    await agent.setup(env)
    await agent.run(instruction, env, AgentContext())


asyncio.run(main())
"""


__all__ = [
    "Terminus2",
    "build_terminus_2_run_script",
    "terminus_2_agent_script",
]
