"""Pixel-only raster/PDF rendering for tool servers.

Install Pillow for raster images and pypdfium2 for PDFs in the viewer's runtime.
Tool names, asset loading, and per-call limits belong to the calling environment.
"""

import io
from contextlib import ExitStack, closing
from typing import Any

from mcp.server.mcpserver import Image as MCPImage
from mcp.types import ImageContent, TextContent


class PixelDocument:
    """An immutable input rendered as PNGs, with one-based page numbers."""

    metadata: dict[str, Any]

    def __init__(self, payload: bytes, *, pdf_scale: float = 2.0):
        if pdf_scale <= 0:
            raise ValueError("pdf_scale must be positive")
        self.payload = payload
        self.pdf_scale = pdf_scale
        self.is_pdf = payload.startswith(b"%PDF-")
        if self.is_pdf:
            import pypdfium2 as pdfium  # ty: ignore[unresolved-import]

            with pdfium.PdfDocument(payload) as document:
                sizes = [
                    list(document.get_page_size(index))
                    for index in range(len(document))
                ]
            self.metadata = {
                "format": "PDF",
                "pages": len(sizes),
                "page_sizes_points": sizes,
            }
        else:
            from PIL import Image

            with Image.open(io.BytesIO(payload)) as image:
                self.metadata = {
                    "format": image.format,
                    "pages": 1,
                    "width": image.width,
                    "height": image.height,
                }

    def render(self, pages: list[int]) -> list[TextContent | ImageContent]:
        """Render complete pages in request order, without reading a text layer."""
        if not pages or any(
            page < 1 or page > self.metadata["pages"] for page in pages
        ):
            raise ValueError(
                f"Request page numbers in the range 1–{self.metadata['pages']}"
            )
        content: list[TextContent | ImageContent] = []
        with ExitStack() as stack:
            if self.is_pdf:
                import pypdfium2 as pdfium  # ty: ignore[unresolved-import]

                document = stack.enter_context(pdfium.PdfDocument(self.payload))
            else:
                from PIL import Image

                image = stack.enter_context(Image.open(io.BytesIO(self.payload)))
            for number in pages:
                buffer = io.BytesIO()
                if self.is_pdf:
                    with (
                        closing(document[number - 1]) as page,
                        closing(page.render(scale=self.pdf_scale)) as bitmap,
                    ):
                        bitmap.to_pil().save(buffer, format="PNG")
                else:
                    image.convert("RGB").save(buffer, format="PNG")
                content.extend(
                    [
                        TextContent(type="text", text=f"Page {number}"),
                        MCPImage(
                            data=buffer.getvalue(), format="png"
                        ).to_image_content(),
                    ]
                )
        return content
