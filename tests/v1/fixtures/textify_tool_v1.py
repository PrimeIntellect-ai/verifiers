"""Textify E2E fixture: score a rendered MCP image result."""

import verifiers.v1 as vf

from tool_response_image_v1 import VisionToolset


class TextifyToolTask(vf.Task):
    tools = (VisionToolset,)

    @vf.reward(weight=1.0)
    async def rendered_image_tool_result(self, trace: vf.Trace) -> float:
        parts = [
            part
            for message in trace.tool_messages
            if isinstance(message.content, list)
            for part in message.content
        ]
        return float(
            any(
                part.type == "text" and part.text.startswith("```image[ascii]")
                for part in parts
            )
            and all(part.type != "image_url" for part in parts)
        )


class TextifyToolTaskset(vf.Taskset[TextifyToolTask, vf.TasksetConfig]):
    def load(self) -> list[TextifyToolTask]:
        return [
            TextifyToolTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "Call the `vision_snapshot` tool exactly once. After it returns, "
                        "reply with exactly `done`."
                    ),
                    system_prompt="Call the requested tool before answering.",
                ),
                self.config.task,
            )
        ]


__all__ = ["TextifyToolTaskset"]
