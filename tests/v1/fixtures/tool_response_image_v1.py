"""tool-response-image: an MCP tool returns image content.

The task asks the model to call a tiny colocated MCP tool. The reward ignores the
final answer and checks the v1 trace: the tool result must survive as an image_url
content part on a ToolMessage.
"""

import verifiers.v1 as vf

PNG_DATA = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8DwHwAFAAH/e+m+7wAAAABJRU5ErkJggg=="
EXPECTED_URL = f"data:image/png;base64,{PNG_DATA}"
SYSTEM = "Call the requested tool before answering."


class VisionToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "vision"  # the model sees `vision_snapshot`

    @vf.tool
    def snapshot(self) -> list:
        """Return a tiny PNG image."""
        from mcp.types import ImageContent, TextContent

        return [
            TextContent(type="text", text="tiny test image"),
            ImageContent(type="image", data=PNG_DATA, mimeType="image/png"),
        ]


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

    def tools(self, task: ToolResponseImageTask) -> list[vf.Toolset]:
        return [VisionToolset(vf.ToolsetConfig())]

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
