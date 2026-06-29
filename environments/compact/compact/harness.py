"""The compacting harness: runs a context-rewrite loop as a uv script.

It carries `notes` across compactions and sends a fresh `[system, user]` each one — the
task on the first turn, then only the carried-over notes — so the prompt is rewritten
rather than appended, and every compaction is its own branch (the deliberate stress test
for branch detection). Within a compaction it can use MCP tools (gather, then summarize
into notes). Its uv script (deps: openai, mcp) is prepared during setup, then launched as
the harness program.
"""

import json
from pathlib import Path

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

from compact.annotate import branch_start_nodes, tool_nodes

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class CompactingHarnessConfig(HarnessConfig):
    """A context-rewrite harness: it rebuilds its prompt from carried-over notes each
    compaction instead of appending, so the trajectory branches at every compaction."""


class CompactingHarness(Harness[CompactingHarnessConfig]):
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        tool_log_path = "/tmp/vf_compact_tool_log.json"
        env = {
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "OPENAI_MODEL": ctx.model,
            "TOOL_LOG": tool_log_path,
        }
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            env["MCP_CONFIG"] = json.dumps(
                {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
            )
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)
        result = await runtime.run_program([*program, trace.task.prompt], env)

        # Tag replay resume points on the finished graph (Option A: trace.info side-channel).
        # The harness is the only writer; the program was the sensor for tool failures.
        node_tags = trace.info.setdefault("node_tags", {})
        for nid in branch_start_nodes(trace):  # this harness rewrites context every turn
            node_tags[str(nid)] = "compaction"
        by_id = tool_nodes(trace)
        for rec in json.loads(await runtime.read(tool_log_path)):
            nid = by_id.get(rec["tool_call_id"])
            if nid is not None:
                node_tags[str(nid)] = rec["tag"]
        return result
