"""parallel: one rollout whose harness sends `n` chat programs at the model at once,
so their requests overlap on one trace. With `same_prompt`, every copy sends the task
prompt; otherwise copy `i` appends its index, so the copies diverge after the system
prompt."""

import asyncio

from pydantic import PositiveInt

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
    launch_chat_program,
)
from verifiers.v1.runtimes import ProgramResult, Runtime

__all__ = ["ParallelHarness", "ParallelTaskset"]


class ParallelHarnessConfig(HarnessConfig):
    n: PositiveInt = 4
    same_prompt: bool = True


class ParallelHarness(Harness[ParallelHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(CHAT_PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: vf.Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(data)

        def copy_prompt(i: int) -> str:
            return prompt if self.config.same_prompt else f"{prompt} ({i})"

        results = await asyncio.gather(
            *(
                launch_chat_program(
                    CHAT_PROGRAM_SOURCE,
                    self.config,
                    ctx,
                    trace,
                    runtime,
                    endpoint,
                    secret,
                    mcp_urls,
                    system_prompt,
                    copy_prompt(i),
                )
                for i in range(self.config.n)
            )
        )
        return next((r for r in results if r.exit_code != 0), results[0])


class ParallelTask(vf.Task):
    @vf.reward(weight=1.0)
    async def answered(self, trace: vf.Trace) -> float:
        return float(trace.num_turns >= 1)


class ParallelTaskset(vf.Taskset[ParallelTask, vf.TasksetConfig]):
    def load(self) -> list[ParallelTask]:
        return [
            ParallelTask(
                vf.TaskData(idx=0, system_prompt="Be brief.", prompt="Reply with OK.")
            )
        ]
