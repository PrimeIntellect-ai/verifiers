"""The compacting harness: runs a context-rewrite loop as a uv script.

It carries `notes` across compactions and sends a fresh `[system, user]` each one — the
task on the first turn, then only the carried-over notes — so the prompt is rewritten
rather than appended, and every compaction is its own branch (the deliberate stress test
for branch detection). Within a compaction it can use MCP tools (gather, then summarize
into notes). The program is a uv script (deps: openai, mcp), launched via
`runtime.run_uv_script`, so the harness only needs `uv` in the runtime.
"""

import json
from pathlib import Path

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class CompactingHarnessConfig(HarnessConfig):
    """A context-rewrite harness: it rebuilds its prompt from carried-over notes each
    compaction instead of appending, so the trajectory branches at every compaction."""

    id: str = "compact"


class CompactingHarness(Harness[CompactingHarnessConfig]):
    SUPPORTS_TASK_TOOLS = True

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
        }
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            env["MCP_CONFIG"] = json.dumps(
                {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
            )
        return await runtime.run_uv_script(
            PROGRAM_SOURCE, args=[trace.task.prompt], env=env
        )
