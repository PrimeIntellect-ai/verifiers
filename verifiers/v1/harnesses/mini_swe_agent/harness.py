import json
from pathlib import Path
from typing import Literal

from pydantic import PositiveInt

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class MiniSWEAgentHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    version: PinnedVersion = "2.4.6"
    """mini-swe-agent release to install, pinned for reproducibility."""
    command_timeout: PositiveInt = 86400
    """Seconds one shell command may run before mini-swe-agent stops it and tells the
    model it timed out. The default leaves the rollout timeout in charge; a shorter one
    keeps a command that never returns (a server started in the foreground) from holding
    the rollout until then."""
    request_timeout: PositiveInt = 86400
    """Seconds mini-swe-agent's LiteLLM client waits for one model call. LiteLLM's own
    default (600) failed the run with `litellm.Timeout` on any longer turn (GLM 5.3 at
    high effort thinks past ten minutes on hard turns); the interception keeps the
    provider stream alive and retries it, so the rollout timeout stays in charge."""


def with_mcp_servers(prompt: str, mcp_urls: dict[str, str]) -> str:
    """The task prompt with the task's MCP servers listed after it, as Harbor's
    mini-swe-agent adapter does: mini-swe-agent has no MCP client of its own, so the
    model reaches a server from the shell at its URL."""
    if not mcp_urls:
        return prompt
    listing = "".join(
        f"- {name}: streamable-http transport, url: {url}\n"
        for name, url in mcp_urls.items()
    )
    return (
        f"{prompt}\n\nMCP Servers:\n"
        f"The following MCP servers are available for this task.\n{listing}"
    )


class MiniSWEAgentHarness(Harness[MiniSWEAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = True
    """Listed in the task prompt (`with_mcp_servers`), as Harbor's adapter does."""
    SUPPORTS_SKILLS = True
    """Accepted and not installed, as Harbor's adapter does: a task's skills folder
    stays readable where the task keeps it (TB4's cumulative-layout-shift keeps its
    in the repo under .agents/skills)."""

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
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("mini-swe-agent does not support disabling tools")
        _, prompt = self.resolve_text_prompt(data)
        prompt = with_mcp_servers(prompt, mcp_urls)
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
            "-c",
            f"environment.timeout={self.config.command_timeout}",
            "-c",
            "model.cost_tracking=ignore_errors",
            "-c",
            "model.model_kwargs.custom_llm_provider=openai",
            "-c",
            "model.model_kwargs.parallel_tool_calls=true",
            "-c",
            f"model.model_kwargs.timeout={self.config.request_timeout}",
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
        args_file = f".vf-mini-swe-agent-args-{trace.id}.json"
        await runtime.write(args_file, json.dumps(args).encode())
        await runtime.run(["chmod", "600", args_file], {})
        program = await runtime.prepare_uv_script(source, self.config.resolved_env)
        return await runtime.run([*program, args_file], env)
