"""The mini-swe-agent harness: runs the native bash-tool agent through LiteLLM."""

from pathlib import Path

from verifiers.v1.clients import RolloutContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class MiniSWEAgentHarnessConfig(HarnessConfig):
    """The mini-swe-agent CLI harness."""

    id: str = "mini-swe-agent"
    version: str = "2.2.8"
    """mini-swe-agent release to install, pinned for reproducibility."""


class MiniSWEAgentHarness(Harness[MiniSWEAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_TASK_TOOLS = False

    async def setup(self, runtime: Runtime) -> None:
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.env)

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("mini-swe-agent does not support disabling tools")
        _, prompt = self.resolve_prompt(trace.task)
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
        env = {
            **self.config.env,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_SILENT_STARTUP": "true",
        }
        program = await runtime.prepare_uv_script(source, self.config.env)
        return await runtime.run_program([*program, *args], env)
