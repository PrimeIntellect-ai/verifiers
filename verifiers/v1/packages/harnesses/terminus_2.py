import shlex
from pathlib import PurePosixPath

from verifiers.types import SamplingArgs

from .command import command_program, command_sandbox
from ...config import SandboxConfig
from ...harness import Harness
from ...utils.prompt_utils import (
    state_system_prompt_text,
    task_text as task_instruction_text,
)
from ...toolset import ToolsetCollection
from ...types import (
    ConfigMap,
    Handler,
    ModelClient,
    ProgramMap,
    ProgramOptionMap,
    ProgramValue,
    PromptInput,
)

DEFAULT_AGENT_WORKDIR = "/app"
DEFAULT_INSTRUCTION_PATH = "/terminus_2/instruction.md"
DEFAULT_SYSTEM_PROMPT_PATH = "/terminus_2/system_prompt.txt"
DEFAULT_LOG_PATH = "/logs/agent/terminus_2.log"
DEFAULT_HARBOR_PACKAGE = "harbor==0.6.6"
DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"
DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"


class Terminus2(Harness):
    def __init__(
        self,
        *,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        instruction_path: str = DEFAULT_INSTRUCTION_PATH,
        system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
        log_path: str = DEFAULT_LOG_PATH,
        harbor_package: str = DEFAULT_HARBOR_PACKAGE,
        python_version: str = DEFAULT_PYTHON_VERSION,
        model_name: str = DEFAULT_MODEL_NAME,
        api_base_url: str = DEFAULT_API_BASE_URL,
        system_prompt: PromptInput | None = None,
        sandbox: bool | ConfigMap | SandboxConfig = True,
        program: ProgramMap | None = None,
        max_turns: int | None = 4,
        user: Handler | str | ConfigMap | None = None,
        client: ModelClient | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
        toolsets: ToolsetCollection | None = None,
        stops: list[Handler] | None = None,
        setups: list[Handler] | None = None,
        updates: list[Handler] | None = None,
        metrics: list[Handler] | None = None,
        rewards: list[Handler] | None = None,
        advantages: list[Handler] | None = None,
        cleanups: list[Handler] | None = None,
    ):
        files: dict[str, ProgramValue] = {
            instruction_path: task_instruction_text,
        }
        if system_prompt is not None:
            files[system_prompt_path] = state_system_prompt_text
        artifacts: ProgramOptionMap = {
            "terminus_2_log": {
                "path": log_path,
                "format": "text",
                "optional": True,
            }
        }
        command = [
            "bash",
            "-lc",
            build_terminus_2_run_script(
                agent_workdir=agent_workdir,
                instruction_path=instruction_path,
                system_prompt_path=system_prompt_path
                if system_prompt is not None
                else None,
                log_path=log_path,
                harbor_package=harbor_package,
                python_version=python_version,
                model_name=model_name,
                api_base_url=api_base_url,
                max_turns=max_turns,
            ),
        ]
        super().__init__(
            program=command_program(
                command=command,
                sandbox=sandbox,
                files=files,
                setup=build_terminus_2_install_script(),
                artifacts=artifacts,
                program=program,
            ),
            sandbox=command_sandbox(sandbox),
            system_prompt=system_prompt,
            max_turns=max_turns,
            user=user,
            client=client,
            model=model,
            sampling_args=sampling_args,
            toolsets=toolsets,
            stops=stops,
            setups=setups,
            updates=updates,
            metrics=metrics,
            rewards=rewards,
            advantages=advantages,
            cleanups=cleanups,
        )


def build_terminus_2_install_script() -> str:
    return """\
set -e
apt-get -o Acquire::Retries=3 update -qq
apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates > /dev/null 2>&1
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
"""


def build_terminus_2_run_script(
    *,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str | None = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    harbor_package: str = DEFAULT_HARBOR_PACKAGE,
    python_version: str = DEFAULT_PYTHON_VERSION,
    model_name: str = DEFAULT_MODEL_NAME,
    api_base_url: str = DEFAULT_API_BASE_URL,
    max_turns: int | None = 4,
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
export PATH="$HOME/.local/bin:$PATH"

TERMINUS_2_WORKDIR="${{AGENT_WORKDIR:-}}"
if [[ -z "$TERMINUS_2_WORKDIR" ]]; then
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
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str | None = DEFAULT_SYSTEM_PROMPT_PATH,
    log_dir: str = "/logs/agent",
    model_name: str = DEFAULT_MODEL_NAME,
    api_base_url: str = DEFAULT_API_BASE_URL,
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
    workdir = Path(os.environ.get("AGENT_WORKDIR") or {DEFAULT_AGENT_WORKDIR!r})
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
    "DEFAULT_AGENT_WORKDIR",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_HARBOR_PACKAGE",
    "DEFAULT_INSTRUCTION_PATH",
    "DEFAULT_LOG_PATH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PYTHON_VERSION",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "Terminus2",
    "build_terminus_2_install_script",
    "build_terminus_2_run_script",
    "terminus_2_agent_script",
]
