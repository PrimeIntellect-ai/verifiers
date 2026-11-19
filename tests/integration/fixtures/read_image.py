"""Read the colors from an image file in the agent workspace."""

import base64
from io import BytesIO

from PIL import Image, ImageDraw

import verifiers.v1 as vf

__all__ = ["ReadImageTaskset"]

PATH = "/tmp/swatches.png"
COLORS = ("yellow", "blue", "green", "red")


def swatches() -> bytes:
    image = Image.new("RGB", (320, 80))
    draw = ImageDraw.Draw(image)
    for i, color in enumerate(COLORS):
        draw.rectangle((80 * i, 0, 80 * (i + 1) - 1, 79), fill=color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class ReadImageTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.write(PATH, swatches())

    @vf.reward(weight=1.0)
    async def read_pixels(self, trace: vf.Trace) -> float:
        url = "data:image/png;base64," + base64.b64encode(swatches()).decode()
        saw_pixels = any(
            part.type == "image_url" and part.image_url.url == url
            for message in trace.messages
            if isinstance(message.content, list)
            for part in message.content
        )
        return float(
            saw_pixels and " ".join(COLORS) in (trace.last_reply or "").lower()
        )


class ReadImageTaskset(vf.Taskset[ReadImageTask, vf.TasksetConfig]):
    def load(self) -> list[ReadImageTask]:
        return [
            ReadImageTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        f"Use IPython attach('{PATH}') to view the image. Reply with "
                        "the four colors from left to right, separated by spaces."
                    ),
                ),
                self.config.task,
            )
        ]
