from math import ceil
from pathlib import Path
from typing import Literal

from verifiers.v1._clients import ModelContext
from verifiers.v1._configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1._harness import Harness
from verifiers.v1._runtimes import ProgramResult, Runtime
from verifiers.v1._runtimes.base import current_program
from verifiers.v1._task import TaskData
from verifiers.v1._trace import Trace
from verifiers.v1.harnesses._utils.launch import stage_text

REPEATED_FORMAT_ERROR_EXIT_CODE = 98
PROGRAM_SOURCE = (
    (Path(__file__).resolve().parent / "_program.py")
    .read_text()
    .replace("{repeated_format_error_exit_code}", str(REPEATED_FORMAT_ERROR_EXIT_CODE))
)


class MiniSWEAgentHarnessConfig(HarnessConfig):
    """Config for the mini-swe-agent harness."""

    dialect: Literal["chat_completions"] = "chat_completions"
    version: PinnedVersion = "2.4.6"
    """mini-swe-agent release to install, pinned for reproducibility."""


class MiniSWEAgentHarness(Harness[MiniSWEAgentHarnessConfig]):
    """Run mini-swe-agent."""

    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False
    SUPPORTS_MEMORY_RESTORE = True

    async def setup(self, runtime: Runtime) -> None:
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.resolved_env)

    async def _check_result(
        self, trace: Trace, runtime: Runtime, result: ProgramResult
    ) -> None:
        if result.exit_code == REPEATED_FORMAT_ERROR_EXIT_CODE:
            trace.stop("format_violation")
            return
        await super()._check_result(trace, runtime, result)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("mini-swe-agent does not support disabling tools")
        _, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("mini-swe-agent requires a task prompt")
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        args = [
            "--model",
            ctx.model,
            "--model-class",
            "litellm",
            "--task-file",
            await stage_text(
                runtime, trace, "task.txt", prompt, env=self.config.resolved_env
            ),
            "--exit-immediately",
            "--yolo",
            "-c",
            "mini",
            "-c",
            "agent.cost_limit=0",
            # mini-SWE-agent's timeout is in whole seconds.
            "-c",
            f"environment.timeout={ceil(self.config.tool_timeout)}",
            "-c",
            "model.cost_tracking=ignore_errors",
            "-c",
            "model.model_kwargs.custom_llm_provider=openai",
            "-c",
            "model.model_kwargs.parallel_tool_calls=true",
            "-c",
            f"model.model_kwargs.api_base={endpoint}",
            "-c",
            f"model.model_kwargs.api_key={secret}",
        ]
        env = {
            **self.config.resolved_env,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_SILENT_STARTUP": "true",
        }
        program = await runtime.prepare_uv_script(source, self.config.resolved_env)
        return await runtime.run([*program, *args], env, program=current_program.get())
