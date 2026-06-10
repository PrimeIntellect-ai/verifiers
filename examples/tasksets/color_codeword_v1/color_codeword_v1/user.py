"""color-codeword-v1 user simulator: reveals the colored squares one turn at a time.

Launched per-rollout as a `vf.User` subprocess. It holds the episode's per-turn colors
(from `COLOR_CODEWORD_INFO`) and serves a single `respond` tool: after each assistant turn
it renders that turn's squares to PNG images and injects them as a multimodal user message
(OpenAI `image_url` parts), until all turns are done. This is where the images enter the
rollout — the renderer turns them into `multi_modal_data` for training.
"""

import base64
import json
import os
from io import BytesIO

from mcp.server.fastmcp import FastMCP
from PIL import Image

import verifiers.v1 as vf

COLOR_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

INFO = json.loads(os.environ["COLOR_CODEWORD_INFO"])
COLORS_PER_TURN = INFO["colors_per_turn"]
NUM_TURNS = INFO["num_turns"]


def _data_url(color: str) -> str:
    img = Image.new("RGB", (100, 100), COLOR_RGB[color])
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _image_message(colors: list[str], text: str) -> dict:
    content = [
        {"type": "image_url", "image_url": {"url": _data_url(c)}} for c in colors
    ]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


mcp = FastMCP("user")

_turn = 0  # squares-turns delivered so far (one `respond` per assistant turn)


@mcp.tool()
def respond(message: str) -> str:
    """Reveal the next turn's squares, or end the episode once all turns are shown."""
    global _turn
    if _turn >= NUM_TURNS:
        return json.dumps({"messages": [], "done": True})
    colors = COLORS_PER_TURN[_turn]
    _turn += 1
    total = sum(len(COLORS_PER_TURN[i]) for i in range(_turn))
    if _turn == NUM_TURNS:
        text = (
            f"Here {'is' if len(colors) == 1 else 'are'} {len(colors)} more square(s). "
            f"Now output the full {total}-letter codeword (letters only, no spaces)."
        )
    else:
        text = (
            f"Here {'is' if len(colors) == 1 else 'are'} {len(colors)} square(s). "
            "Output the accumulated codeword so far (letters only)."
        )
    return json.dumps({"messages": [_image_message(colors, text)], "done": False})


vf.run_mcp_server(mcp)
