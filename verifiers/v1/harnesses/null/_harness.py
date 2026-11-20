from typing import Literal

from verifiers.v1._clients import ModelContext
from verifiers.v1._configs.harness import HarnessConfig
from verifiers.v1._harness import Harness
from verifiers.v1._runtimes import ProgramResult, Runtime
from verifiers.v1._task import TaskData
from verifiers.v1._trace import Trace
from verifiers.v1.harnesses._utils.launch import (
    CHAT_PROGRAM_SOURCE,
    CompactionConfig,
    compaction_args,
    launch_chat_program,
)


class NullHarnessConfig(HarnessConfig):
    """Config for the plain chat harness."""

    dialect: Literal["chat_completions"] = "chat_completions"
    compaction: CompactionConfig | None = None
    """Context compaction, off by default. Set it (e.g. an empty `[compaction]` table for
    auto thresholds, or `summarize_at_tokens`) to summarize the conversation and continue
    instead of ending the run when the context window fills."""


class NullHarness(Harness[NullHarnessConfig]):
    """Plain chat loop with no local tools."""

    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(CHAT_PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(data)
        args = compaction_args(self.config.compaction)
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        return await launch_chat_program(
            CHAT_PROGRAM_SOURCE,
            self.config,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            system_prompt,
            prompt,
            extra_args=args,
        )
