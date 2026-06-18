# /// script
# requires-python = ">=3.12"
# dependencies = ["harbor=={version}"]
# ///

import asyncio
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths


class LocalEnvironment:
    default_user = None
    session_id = "verifiers"

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = user
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )


async def main() -> None:
    model, task = sys.argv[1:]
    logs_dir = Path(os.environ["TMUX_TMPDIR"])
    logs_dir.mkdir(mode=0o700, exist_ok=True)
    EnvironmentPaths.agent_dir = PurePosixPath(logs_dir)

    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=model,
        api_base=os.environ["OPENAI_BASE_URL"],
        llm_kwargs={"custom_llm_provider": "openai"},
        record_terminal_session=False,
    )
    environment = LocalEnvironment()
    await agent.setup(environment)
    await agent.run(task, environment, AgentContext())


if __name__ == "__main__":
    asyncio.run(main())
