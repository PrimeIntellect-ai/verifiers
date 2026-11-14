from pathlib import Path
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.remote_runtime import (
    REMOTE_TOOL_CLIENT_SOURCE,
    remote_shell_tools,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class MiniSWEAgentHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    version: PinnedVersion = "2.4.6"
    """mini-swe-agent release to install, pinned for reproducibility."""


class MiniSWEAgentHarness(Harness[MiniSWEAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False
    SUPPORTS_SPLIT_EXECUTION = True
    SUPPORTS_SPLIT_TOOL_INTERCEPTION = True

    def _source(self) -> str:
        return PROGRAM_SOURCE.replace("{version}", self.config.version).replace(
            "# REMOTE_TOOL_CLIENT", REMOTE_TOOL_CLIENT_SOURCE
        )

    async def remote_tools(self, runtime: Runtime, data: TaskData):
        return remote_shell_tools(runtime)

    async def setup(self, runtime: Runtime) -> None:
        source = self._source()
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
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("mini-swe-agent does not support disabling tools")
        _, prompt = self.resolve_text_prompt(data)
        source = self._source()
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
        env = {
            **self.config.resolved_env,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_SILENT_STARTUP": "true",
        }
        if runtime._remote_tools_only:
            if not tool_interception_url or runtime._execution_bridge is None:
                raise ValueError("split mini-swe-agent requires a remote tool gate")
            env.update(
                VF_REMOTE_TOOL_URL=tool_interception_url,
                VF_REMOTE_TOOL_SECRET=secret,
                VF_EXECUTION_CWD=runtime._execution_bridge.runtime.info.workdir,
            )
        program = await runtime.prepare_uv_script(source, self.config.resolved_env)
        return await runtime.run([*program, *args], env)
