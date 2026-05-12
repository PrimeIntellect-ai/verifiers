from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from PIL import Image

ImageLike = Union[str, bytes, "Image.Image"]


def add_image(message: dict, image: ImageLike) -> None:
    """
    Append an OpenAI-style image block to a chat message's content.

    `message` uses the multimodal shape: content is a list of parts, each with
    ``type`` and payload. ``image`` may be a filesystem path, a ``data:`` or
    ``http(s):`` URL string, raw PNG/JPEG bytes, or a PIL image.
    """
    if "content" not in message:
        message["content"] = []
    content = message["content"]
    if isinstance(content, str):
        message["content"] = [{"type": "text", "text": content}]
        content = message["content"]
    if not isinstance(content, list):
        raise TypeError("message['content'] must be str or list after normalization")

    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            url = image
        else:
            data = Path(image).expanduser().read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            url = f"data:image/png;base64,{b64}"
    elif isinstance(image, bytes):
        b64 = base64.b64encode(image).decode("ascii")
        url = f"data:image/png;base64,{b64}"
    else:
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        url = f"data:image/png;base64,{b64}"

    content.append({"type": "image_url", "image_url": {"url": url}})
