import json
import shlex
from typing import Generic, TypeVar, cast

import verifiers.v1 as vf


class CommandHarnessConfig(vf.HarnessConfig):
    command: list[str] = []
    cwd: str | None = None
    env: dict[str, str] = {}
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
        return {
            **self.config.env,
            "VF_TASK_JSON": json.dumps(task.to_record(), ensure_ascii=False),
            "VF_STATE_ID": state.id,
            "VF_PROMPT": vf.messages_text(self.initial_messages(task)),
        }

    async def _run(
        self,
        task: vf.Task,
        state: vf.State,
        *,
        ctx: vf.RolloutContext,
        runtime: vf.RuntimeSession | None = None,
        tools: vf.MCPToolRegistry | None = None,
        user: vf.MCPToolRegistry | None = None,
    ) -> None:
        if runtime is None:
            raise ValueError("CommandHarness requires a runtime session.")
        _ = tools, user
        prompt = self.initial_messages(task)

        async def stop_check() -> str | None:
            if await self.is_completed(task, state, ctx=ctx):
                return state.stop_condition or "stop"
            return None

        async with vf.InterceptionServer(
            ctx,
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
                    **endpoint.env(base_url=endpoint_url, model=ctx.model),
                },
                timeout=self.config.timeout_seconds,
            )
        state.artifacts["command"] = result.model_dump(mode="json")
        content = result.stdout.strip() or result.stderr.strip()
        if not state.transcript:
            message = vf.AssistantMessage(content=content)
            state.add_turn(vf.Turn(prompt=prompt, completion=[message]))
        if result.returncode == 0:
            state.stop("command_completed")
        else:
            state.stop("command_failed")


def shell_command(command: str) -> list[str]:
    return ["bash", "-lc", command]


def quote_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def command_metrics(state: vf.State, key: str) -> float:
    command = state.artifacts.get("command")
    if not isinstance(command, dict):
        return 0.0
    value = cast(dict[str, object], command).get(key)
    return float(value) if isinstance(value, int | float) else 0.0
