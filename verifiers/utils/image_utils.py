import base64
from io import BytesIO
from PIL import Image

def _base64_to_pil(data_uri: str) -> Image.Image:
    """Convert a base64 data URI (data:image/...;base64,...) to a PIL Image."""
    if not data_uri.startswith("data:image"):
        raise ValueError(f"Expected base64 image data URI, got: {data_uri[:30]}")
    header, b64data = data_uri.split(",", 1)
    image_data = base64.b64decode(b64data)
    return Image.open(BytesIO(image_data)).convert("RGB")

def pil_to_base64_url(pil_image) -> str:
    """
    Convert a PIL image to a base64 URL string suitable for OpenAI/vLLM messages.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"