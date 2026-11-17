from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
    CompactionConfig,
    compaction_args,
    has_internal_mcp,
    launch_chat_program,
    run_in_process_chat,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


class NullHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    compaction: CompactionConfig | None = None
    """Context compaction, off by default. Set it (e.g. an empty `[compaction]` table for
    auto thresholds, or `summarize_at_tokens`) to summarize the conversation and continue
    instead of ending the run when the context window fills."""


class NullHarness(Harness[NullHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False
    RUNS_ON_HOST = True

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
        if has_internal_mcp(mcp_urls):
            args = compaction_args(self.config.compaction)
            if tool_interception_url:
                args.append(
                    f"--tool-interception-url={runtime.host_url(tool_interception_url)}"
                )
            return await launch_chat_program(
                CHAT_PROGRAM_SOURCE,
                self.config,
                ctx,
                trace,
                runtime,
                runtime.host_url(endpoint),
                secret,
                mcp_urls,
                system_prompt,
                prompt,
                extra_args=args,
            )
        return await run_in_process_chat(
            self.config,
            ctx,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            system_prompt,
            prompt,
            compaction_cfg=self.config.compaction,
            tool_interception_url=tool_interception_url,
        )
