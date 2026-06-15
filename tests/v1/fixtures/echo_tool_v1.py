"""echo (v1, tool): the model calls a self-contained tool server and echoes its result.

A single-tool fixture for the e2e server-runtime matrix. The tool server is a self-contained
uv `script` (only dep: mcp), so it runs in EVERY runtime (subprocess / its own docker / its own
prime sandbox) via uv — the right vehicle to cover a tool server in its own runtime (a vf-native
`Toolset` would need `verifiers` + the taskset package installed there). The runtime is set by
`server_runtime`, threaded into the tool's placement `config`.
"""

import verifiers.v1 as vf

PHRASE = "hello world"

TOOL_SCRIPT = b"""# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp"]
# ///
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo", host="127.0.0.1", port=int(os.environ["MCP_PORT"]))


@mcp.tool()
def say(text: str) -> str:
    \"\"\"Return the text you pass, verbatim.\"\"\"
    return text


mcp.run(transport="streamable-http")
"""


def _key(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


class EchoToolTask(vf.Task):
    phrase: str


class EchoToolConfig(vf.TasksetConfig):
    phrase: str = PHRASE
    server_runtime: str = "subprocess"
    """Runtime the tool server runs in (its own) — set by the e2e server-runtime matrix."""


class EchoToolTaskset(vf.Taskset[EchoToolTask, EchoToolConfig]):
    def load_tasks(self) -> list[EchoToolTask]:
        return [
            EchoToolTask(
                idx=0,
                instruction=(
                    f'Call the `echo_say` tool with the text "{self.config.phrase}", '
                    "then reply with exactly what it returns."
                ),
                phrase=self.config.phrase,
            )
        ]

    def tools(self, task: EchoToolTask) -> list[vf.Tools]:
        return [
            vf.Tools(
                name="echo",
                script=TOOL_SCRIPT,
                config=vf.ToolsetConfig(runtime={"type": self.config.server_runtime}),
            )
        ]

    @vf.reward(weight=1.0)
    async def echoed(self, task: EchoToolTask, trace: vf.Trace) -> float:
        last = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(_key(task.phrase) in _key(last or ""))


def load_taskset(config: EchoToolConfig) -> EchoToolTaskset:
    return EchoToolTaskset(config)
