from pathlib import Path
from typing import Literal

from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class MiniSWEAgentHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    version: PinnedVersion = "2.4.6"
    """mini-swe-agent release to install, pinned for reproducibility."""


class MiniSWEAgentHarness(Harness[MiniSWEAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False

    async def setup(self, runtime: Runtime) -> None:
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        await runtime.prepare_uv_script(source, self.config.resolved_env)

    async def launch(self, data: TaskData) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("mini-swe-agent does not support disabling tools")
        _, prompt = self.resolve_text_prompt(data)
        source = PROGRAM_SOURCE.replace("{version}", self.config.version)
        args = [
            "--model",
            self.model.model,
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
            f"model.model_kwargs.api_base={self.wiring.endpoint}",
            "-c",
            f"model.model_kwargs.api_key={self.wiring.secret}",
        ]
        env = {
            **self.config.resolved_env,
            "MSWEA_CONFIGURED": "true",
            "MSWEA_SILENT_STARTUP": "true",
        }
        program = await self.runtime.prepare_uv_script(source, self.config.resolved_env)
        return await self.runtime.run([*program, *args], env)
