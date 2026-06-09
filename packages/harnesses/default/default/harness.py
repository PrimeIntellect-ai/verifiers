"""The built-in default harness: runs a small chat-loop program as a uv script.

A growing-message-list chat loop with an optional bash tool (`enable_bash`), plus MCP tools
when the rollout has tool servers (host-side, resolved to URLs by the Environment). The program is a uv script
(deps: openai, mcp), launched via `runtime.run_uv_script` — so it works on any runtime
with `uv` (the harness bootstraps it), with no runtime-specific setup.
"""

import json
from pathlib import Path

from verifiers.v1.harnesses.base import Harness, HarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class DefaultHarnessConfig(HarnessConfig):
    """The built-in harness. A uv script (deps: openai, mcp), so it runs in any runtime that
    has `uv` (the harness bootstraps it) with no other setup."""

    id: str = "default"
    enable_bash: bool = False
    """Offer the model a local `bash` tool. Off by default — a pure-text chat loop where
    the model answers directly (MCP tools from the taskset, if any, are still wired in);
    enable it (`--harness.enable-bash true`) for agents that run shell commands, e.g.
    harbor terminal tasks."""


class DefaultHarness(Harness[DefaultHarnessConfig]):
    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        env = {
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "OPENAI_MODEL": ctx.model,
            "ENABLE_BASH": "1" if self.config.enable_bash else "0",
        }
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            env["MCP_CONFIG"] = json.dumps(
                {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
            )
        return await runtime.run_uv_script(
            PROGRAM_SOURCE, args=[trace.task.instruction], env=env
        )
