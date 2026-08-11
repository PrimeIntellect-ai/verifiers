import base64
from io import BytesIO

from PIL import Image, ImageOps

import verifiers.v1 as vf
from verifiers.v1.utils.image import image_data_url


class GrayscaleInterceptionTask(vf.Task):
    @vf.intercept
    def grayscale(self, request: vf.Request) -> vf.Request | None:
        # The last request message is the new user input, before the harness stores it.
        message = request.messages[-1]
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
            metadata, separator, encoded = part.image_url.url.partition(",")
            if not separator or not metadata.lower().endswith(";base64"):
                content.append(part)
                continue
            try:
                with Image.open(
                    BytesIO(base64.b64decode(encoded, validate=True))
                ) as image:
                    alpha = (
                        image.convert("RGBA").getchannel("A")
                        if image.has_transparency_data
                        else None
                    )
                    grayscale_image = ImageOps.grayscale(image)
                    if alpha is not None:
                        grayscale_image.putalpha(alpha)
                    grayscale = image_data_url(grayscale_image)
            except (ValueError, OSError):
                content.append(part)
                continue
            content.append(
                part.model_copy(
                    update={
                        "image_url": part.image_url.model_copy(
                            update={"url": grayscale}
                        )
                    }
                )
            )
            changed = True
        if not changed:
            return None
        messages = [
            *request.messages[:-1],
            message.model_copy(update={"content": content}),
        ]
        return request.model_copy(update={"messages": messages})

    @vf.reward
    async def changed(self, trace: vf.Trace) -> float:
        return float(bool(trace.request_rewrites))


class GrayscaleInterceptionTaskset(vf.Taskset[GrayscaleInterceptionTask]):
    def load(self) -> list[GrayscaleInterceptionTask]:
        # Building the image here keeps the example self-contained.
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
