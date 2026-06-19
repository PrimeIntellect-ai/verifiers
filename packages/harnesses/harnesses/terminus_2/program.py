# /// script
# requires-python = ">=3.12"
# dependencies = ["harbor=={version}"]
# ///

import argparse
import asyncio
import os
import subprocess
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--task", required=True)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    model, system_prompt, task = args.model, args.system_prompt, args.task
    logs_dir = Path(os.environ["TMUX_TMPDIR"])
    logs_dir.mkdir(mode=0o700, exist_ok=True)
    EnvironmentPaths.agent_dir = PurePosixPath(logs_dir)

    # api_base/api_key go to LiteLLM explicitly (not OPENAI_* env), so the agent's terminal
    # commands don't inherit the interception endpoint and get recorded as model turns.
    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=model,
        api_base=args.base_url,
        llm_kwargs={"custom_llm_provider": "openai", "api_key": args.api_key},
        record_terminal_session=False,
    )
    if system_prompt:
        call = agent._llm.call

        async def call_with_system_prompt(*args, message_history, **kwargs):
            return await call(
                *args,
                message_history=[
                    {"role": "system", "content": system_prompt},
                    *message_history,
                ],
                **kwargs,
            )

        agent._llm.call = call_with_system_prompt
    environment = LocalEnvironment()
    await agent.setup(environment)
    await agent.run(task, environment, AgentContext())


if __name__ == "__main__":
    asyncio.run(main())
