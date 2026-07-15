import base64
from io import BytesIO


def image_data_url(image) -> str:
    """Encode a PIL-compatible image as a base64 PNG data URL."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
