"""Read the colors from two image files in the agent workspace."""

import base64
from io import BytesIO

from PIL import Image, ImageDraw

import verifiers.v1 as vf

__all__ = ["ReadImageTaskset"]

PAGES = {"/tmp/first.png": ("yellow", "blue"), "/tmp/second.png": ("green", "red")}


def swatches(colors: tuple[str, ...]) -> bytes:
    image = Image.new("RGB", (80 * len(colors), 80))
    draw = ImageDraw.Draw(image)
    for i, color in enumerate(colors):
        draw.rectangle((80 * i, 0, 80 * (i + 1) - 1, 79), fill=color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class ReadImageTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        for path, colors in PAGES.items():
            await runtime.write(path, swatches(colors))

    @vf.reward(weight=1.0)
    async def read_pixels(self, trace: vf.Trace) -> float:
        expected = [
            "data:image/png;base64," + base64.b64encode(swatches(colors)).decode()
            for colors in PAGES.values()
        ]
        shown = [
            part.image_url.url
            for message in trace.messages
            if isinstance(message.content, list)
            for part in message.content
            if part.type == "image_url"
        ]
        answer = " ".join(color for colors in PAGES.values() for color in colors)
        return float(shown == expected and answer in (trace.last_reply or "").lower())


class ReadImageTaskset(vf.Taskset[ReadImageTask, vf.TasksetConfig]):
    def load(self) -> list[ReadImageTask]:
        return [
            ReadImageTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        f"Call IPython once with attach_image({', '.join(map(repr, PAGES))}). "
                        "Read both images in file order, from left to right, "
                        "and reply with only the four colors separated by spaces. "
                        "Do not decode the files "
                        "in code or make any further tool calls."
                    ),
                ),
                self.config.task,
            )
        ]
