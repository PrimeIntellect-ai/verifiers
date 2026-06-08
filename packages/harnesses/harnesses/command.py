import json
from typing import Generic, TypeVar

from pydantic import Field

import verifiers.v1 as vf


class CommandHarnessConfig(vf.HarnessConfig):
    command: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float | None = None


ConfigT = TypeVar("ConfigT", bound=CommandHarnessConfig)


class CommandHarness(vf.Harness[ConfigT], Generic[ConfigT]):
    config: ConfigT

    def command(self, task: vf.Task, state: vf.State) -> list[str]:
        _ = task, state
        if not self.config.command:
            raise ValueError(f"{type(self).__name__} requires config.command.")
        return list(self.config.command)

    def command_env(self, task: vf.Task, state: vf.State) -> dict[str, str]:
        prompt_text = "\n\n".join(
            str(getattr(message, "content", "") or "")
            for message in self.initial_messages(task)
        )
        return {
            **self.config.env,
            "VF_TASK_JSON": json.dumps(
                task.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
                ensure_ascii=False,
            ),
            "VF_STATE_ID": state.id,
            "VF_PROMPT": prompt_text,
        }

    async def run_with_context(self, context: vf.Context) -> None:
        task = context.task
        state = context.state
        runtime = context.runtime
        if runtime is None:
            raise ValueError("CommandHarness requires a runtime.")
        prompt = self.initial_messages(task)

        async def stop_check() -> str | None:
            if await self.is_completed(context):
                return state.stop_condition or "stop"
            return None

        async with vf.InterceptionServer(
            context,
            task,
            state,
            protocols=self.protocols,
            stop_check=stop_check,
        ) as endpoint:
            endpoint_url = await runtime.expose(endpoint.port)
            result = await runtime.run(
                self.command(task, state),
                cwd=self.config.cwd,
                env={
                    **self.command_env(task, state),
                    **endpoint.env(base_url=endpoint_url, model=context.model),
                },
                timeout=self.config.timeout_seconds,
            )
        state.artifacts["command"] = result.model_dump(mode="json")
        content = result.stdout.strip() or result.stderr.strip()
        if not state.transcript:
            message = vf.AssistantMessage(content=content)
            state.transcript.append(vf.Turn(prompt=prompt, completion=[message]))
        if result.returncode == 0:
            state.stop("command_completed")
        else:
            state.stop("command_failed")


def shell_command(command: str) -> list[str]:
    return ["bash", "-lc", command]
