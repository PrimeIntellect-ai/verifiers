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
from verifiers.v1.harnesses.utils import mcp
from verifiers.v1.harnesses.utils.launch import bundle_program
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = bundle_program(
    (Path(__file__).resolve().parent / "program.py").read_text(), mcp
)


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
        if data.mcp_servers:
            env["MCP_CONFIG"] = json.dumps({"mcpServers": data.mcp_servers})
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.env, activate=False
        )
        return await runtime.run_program([*program, prompt], env)
