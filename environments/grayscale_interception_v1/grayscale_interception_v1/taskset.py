import base64
from io import BytesIO

from PIL import Image, ImageOps

import verifiers.v1 as vf
from verifiers.v1.utils.image import image_data_url


class GrayscaleInterceptionTask(vf.Task):
    # Replace inline images before the model and trace receive the user message.
    @vf.on_request
    def grayscale(self, request: vf.Messages) -> vf.UserMessage | None:
        message = request[-1]
        if not isinstance(message, vf.UserMessage) or not isinstance(
            message.content, list
        ):
            return None
        content: list[vf.ContentPart] = []
        changed = False
        for part in message.content:
            if not isinstance(
                part, vf.ImageUrlContentPart
            ) or not part.image_url.url.startswith("data:image/"):
                content.append(part)
                continue
            encoded = part.image_url.url.split(",", 1)[1]
            with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                url = image_data_url(ImageOps.grayscale(image))
            content.append(
                part.model_copy(
                    update={"image_url": part.image_url.model_copy(update={"url": url})}
                )
            )
            changed = True
        if changed:
            return message.model_copy(update={"content": content})

    @vf.reward
    async def intercepted(self, trace: vf.Trace) -> float:
        return float(bool(trace.interceptions))


class GrayscaleInterceptionTaskset(vf.Taskset[GrayscaleInterceptionTask]):
    def load(self) -> list[GrayscaleInterceptionTask]:
        # Keep the example self-contained by creating its input image here.
        image = Image.new("RGB", (64, 64), "orange")
        prompt = [
            vf.UserMessage(
                content=[
                    vf.ImageUrlContentPart(
                        image_url=vf.ImageUrlSource(url=image_data_url(image))
                    ),
                    vf.TextContentPart(text="Describe this image."),
                ]
            )
        ]
        return [
            GrayscaleInterceptionTask(
                vf.TaskData(idx=0, prompt=prompt), self.config.task
            )
        ]
