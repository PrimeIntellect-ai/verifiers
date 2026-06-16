"""tool-response-image: an MCP tool returns image content.

The task asks the model to call a tiny colocated MCP tool. The reward ignores the
final answer and checks the v1 trace: the tool result must survive as an image_url
content part on a ToolMessage.
"""

import verifiers.v1 as vf

PNG_DATA = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8DwHwAFAAH/e+m+7wAAAABJRU5ErkJggg=="
EXPECTED_URL = f"data:image/png;base64,{PNG_DATA}"
SYSTEM = "Call the requested tool before answering."

TOOL_SERVER = f'''# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp"]
# ///
"""Image-returning MCP tool server for the v1 e2e fixture."""

import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

PNG_DATA = "{PNG_DATA}"
mcp = FastMCP("vision", host="127.0.0.1", port=int(os.environ["MCP_PORT"]))


@mcp.tool()
def snapshot():
    """Return a tiny PNG image."""
    return [
        TextContent(type="text", text="tiny test image"),
        ImageContent(type="image", data=PNG_DATA, mimeType="image/png"),
    ]


mcp.run(transport="streamable-http")
'''.encode()


class ToolResponseImageTask(vf.Task):
    pass


class ToolResponseImageTaskset(vf.Taskset[ToolResponseImageTask, vf.TasksetConfig]):
    def load_tasks(self) -> list[ToolResponseImageTask]:
        return [
            ToolResponseImageTask(
                idx=0,
                instruction=(
                    "Call the `vision_snapshot` tool exactly once. After it returns, "
                    "reply with exactly `done`."
                ),
                system_prompt=SYSTEM,
            )
        ]

    def tools(self, task: ToolResponseImageTask) -> list[vf.Tools]:
        return [vf.Tools(name="vision", script=TOOL_SERVER)]

    @vf.reward(weight=1.0)
    async def preserved_image_tool_result(self, trace: vf.Trace) -> float:
        for message in trace.tool_messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if part.type == "image_url" and part.image_url.url == EXPECTED_URL:
                        return 1.0
        return 0.0


def load_taskset(config: vf.TasksetConfig) -> ToolResponseImageTaskset:
    return ToolResponseImageTaskset(config)
