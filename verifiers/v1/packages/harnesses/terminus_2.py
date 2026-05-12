from __future__ import annotations

import shlex
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .cli import CLIHarness
from ...config import SandboxConfig
from ...utils.prompt_utils import (
    state_system_prompt_text,
    task_text as task_instruction_text,
)

DEFAULT_AGENT_WORKDIR = "/app"
DEFAULT_INSTRUCTION_PATH = "/terminus_2/instruction.md"
DEFAULT_SYSTEM_PROMPT_PATH = "/terminus_2/system_prompt.txt"
DEFAULT_LOG_PATH = "/logs/agent/terminus_2.log"
DEFAULT_HARBOR_PACKAGE = "harbor==0.6.6"
DEFAULT_PYTHON_VERSION = "3.12"


class Terminus2(CLIHarness):
    def __init__(
        self,
        *,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        instruction_path: str = DEFAULT_INSTRUCTION_PATH,
        system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
        log_path: str = DEFAULT_LOG_PATH,
        harbor_package: str = DEFAULT_HARBOR_PACKAGE,
        python_version: str = DEFAULT_PYTHON_VERSION,
        system_prompt: object | None = None,
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
            "terminus_2_log": {
                "path": log_path,
                "format": "text",
                "optional": True,
            }
        }
        super().__init__(
            command=[
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
                    max_turns=max_turns,
                ),
            ],
            sandbox=sandbox,
            files=files,
            setup=build_terminus_2_install_script(),
            env={"OPENAI_MODEL": "runtime.model"},
            artifacts=artifacts,
            program=program,
            system_prompt=system_prompt,
            max_turns=max_turns,
            **kwargs,
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
    max_turns: int | None = 4,
) -> str:
    log_dir = str(PurePosixPath(log_path).parent)
    agent_script = terminus_2_agent_script(
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_dir=log_dir,
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
    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=os.environ.get("OPENAI_MODEL") or "openai/gpt-4",
        api_base=os.environ["OPENAI_BASE_URL"],
        max_turns={max_turns!r},
    )
    await agent.setup(env)
    await agent.run(instruction, env, AgentContext())


asyncio.run(main())
"""


__all__ = [
    "DEFAULT_AGENT_WORKDIR",
    "DEFAULT_HARBOR_PACKAGE",
    "DEFAULT_INSTRUCTION_PATH",
    "DEFAULT_LOG_PATH",
    "DEFAULT_PYTHON_VERSION",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "Terminus2",
    "build_terminus_2_install_script",
    "build_terminus_2_run_script",
    "terminus_2_agent_script",
]
