"""The compacting harness: a context-rewrite loop run as a uv script.

Each compaction sends a fresh `[system, user]` — the task on the first turn, then only
the model's saved notes — so every compaction is its own branch. See `program.py` for
the turn protocol.
"""

import json
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class CompactingHarnessConfig(HarnessConfig):
    """A context-rewrite harness: it rebuilds its prompt from carried-over notes each
    compaction instead of appending, so the trajectory branches at every compaction."""


class CompactingHarness(Harness[CompactingHarnessConfig]):
    SUPPORTS_MCP = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)

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
        _, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("Compacting harness requires a string task prompt")
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
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)
        return await runtime.run_program([*program, prompt], env)
