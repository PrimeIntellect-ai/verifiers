# /// script
# requires-python = ">=3.12,<3.15"
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

# REMOTE_TOOL_CLIENT


class LocalEnvironment:
    default_user = None
    session_id = "verifiers"

    def __init__(self, args):
        self.args = args

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = user
        if self.args.tool_interception_url:
            content = await asyncio.to_thread(
                remote_tool,  # noqa: F821 - injected shared client
                self.args.tool_interception_url,
                self.args.api_key,
                "bash",
                {
                    "command": command,
                    "cwd": cwd,
                    "env": {"TMUX_TMPDIR": os.environ["TMUX_TMPDIR"], **(env or {})},
                    "timeout": timeout_sec,
                },
            )
            result = remote_shell_result(content)  # noqa: F821
            return ExecResult(
                stdout=result["stdout"],
                stderr=result["stderr"] + result["exception_info"],
                return_code=result["returncode"],
            )
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
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
    parser.add_argument("--tool-interception-url", default="")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    model, system_prompt, task = args.model, args.system_prompt, args.task
    logs_dir = Path(os.environ["TMUX_TMPDIR"])
    logs_dir.mkdir(mode=0o700, exist_ok=True)
    EnvironmentPaths.agent_dir = PurePosixPath(logs_dir)

    agent = Terminus2(
        logs_dir=logs_dir,
        model_name=model,
        api_base=args.base_url,
        llm_kwargs={"custom_llm_provider": "openai", "api_key": args.api_key},
        record_terminal_session=False,
        interleaved_thinking=True,
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
    environment = LocalEnvironment(args)
    await agent.setup(environment)
    await agent.run(task, environment, AgentContext())


if __name__ == "__main__":
    asyncio.run(main())
