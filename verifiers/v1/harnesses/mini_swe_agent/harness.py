from pathlib import Path

from pydantic import Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.interception import (
    DIRECT_TOOL_SOURCE,
    prepare_tool_interception,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (
    (Path(__file__).resolve().parent / "program.py")
    .read_text()
    .replace("# {tool_interception}", DIRECT_TOOL_SOURCE)
)


class MiniSWEAgentHarnessConfig(HarnessConfig):
    version: str = Field(default="2.4.6", pattern=r"^[A-Za-z0-9._+-]+$")
    """mini-swe-agent release to install, pinned for reproducibility."""


class MiniSWEAgentHarness(Harness[MiniSWEAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False
    SUPPORTS_PRE_TOOL_INTERCEPTION = True
    SUPPORTS_POST_TOOL_INTERCEPTION = True

    async def setup(self, runtime: Runtime) -> None:
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.resolved_env)

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
        if self.config.disabled_tools:
            raise ValueError("mini-swe-agent does not support disabling tools")
        _, prompt = self.resolve_text_prompt(data)
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        args = [
            "--model",
            ctx.model,
            "--model-class",
            "litellm",
            "--task",
            prompt,
            "--exit-immediately",
            "--yolo",
            "-c",
            "mini",
            "-c",
            "agent.cost_limit=0",
            # Effectively unlimited; Verifiers owns the rollout timeout.
            "-c",
            "environment.timeout=86400",
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
        tool_interception_secret = prepare_tool_interception(
            args, runtime, tool_interception, "Mini-SWE"
        )
        if tool_interception_secret is not None:
            args += ["--agent-class", "__main__.InterceptingAgent"]
        env = {
            **self.config.resolved_env,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_SILENT_STARTUP": "true",
        }
        program = await runtime.prepare_uv_script(
            source,
            self.config.resolved_env,
            activate=tool_interception_secret is None,
        )
        return await runtime.run_program(
            [*program, *args], env, stdin=tool_interception_secret
        )
