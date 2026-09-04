from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
    MCP_CHAT_PROGRAM_SOURCE,
    launch_chat_program,
)
from verifiers.v1.interception import prepare_tool_interception
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


class NullHarnessConfig(HarnessConfig):
    pass


class NullHarness(Harness[NullHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    SUPPORTS_PRE_TOOL_INTERCEPTION = True
    SUPPORTS_POST_TOOL_INTERCEPTION = True
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
        tool_interception: tuple[str, str] | None = None,
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(data)
        args: list[str] = []
        tool_interception_secret = (
            prepare_tool_interception(args, runtime, tool_interception, "Null")
            if mcp_urls
            else None
        )
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
            source_with_mcp=MCP_CHAT_PROGRAM_SOURCE,
            extra_args=args,
            activate=tool_interception_secret is None,
            stdin=tool_interception_secret,
        )
