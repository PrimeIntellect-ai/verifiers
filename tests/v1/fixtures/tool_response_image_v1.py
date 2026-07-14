"""tool-response-image: an MCP tool returns image content.

The task asks the model to call a tiny colocated MCP tool. The reward ignores the
final answer and checks the v1 trace: the tool result must survive as an image_url
content part on a ToolMessage.
"""

import verifiers.v1 as vf

PNG_DATA = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKUlEQVRIx+3NMQEAAAjDMMC/52ECvlRA06nUZ/N6BwAAAAAAAAAAAIcttVsCPm9Ue3AAAAAASUVORK5CYII="
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
    tools = (VisionToolset,)

    @vf.reward(weight=1.0)
    async def preserved_image_tool_result(self, trace: vf.Trace) -> float:
        for message in trace.tool_messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if part.type == "image_url" and part.image_url.url == EXPECTED_URL:
                        return 1.0
        return 0.0


class ToolResponseImageTaskset(vf.Taskset[ToolResponseImageTask, vf.TasksetConfig]):
    def load(self) -> list[ToolResponseImageTask]:
        return [
            ToolResponseImageTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "Call the `vision_snapshot` tool exactly once. After it returns, "
                        "reply with exactly `done`."
                    ),
                    system_prompt=SYSTEM,
                ),
                self.config.task,
            )
        ]


__all__ = ["ToolResponseImageTaskset"]


if __name__ == "__main__":
    VisionToolset.run()
