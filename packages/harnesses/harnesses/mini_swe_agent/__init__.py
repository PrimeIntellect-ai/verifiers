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

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        _, instruction = self.resolve_prompt(trace.task)
        args = [
            "--model",
            ctx.model,
            "--model-class",
            "litellm",
            "--task",
            instruction,
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
        ]
        env = {
            **self.config.env,
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_SILENT_STARTUP": "true",
        }
        return await runtime.run_uv_script(
            PROGRAM_SOURCE.replace("{version}", self.config.version), args=args, env=env
        )


def load_harness(config: MiniSWEAgentHarnessConfig) -> MiniSWEAgentHarness:
    return MiniSWEAgentHarness(config)


__all__ = [
    "MiniSWEAgentHarness",
    "MiniSWEAgentHarnessConfig",
    "load_harness",
]
