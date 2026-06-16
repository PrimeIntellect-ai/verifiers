"""color-codeword user simulator: a `vf.User` revealing each turn's colored squares.

Self-launching — the framework starts it with `python -m color_codeword_v1.servers.user`.
"""

import base64
from io import BytesIO

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


class ColorCodewordUser(vf.User[vf.UserConfig]):
    """Reveals each turn's colored squares after the prior answer: one `respond` per assistant
    turn, injecting the next turn's squares (image_url parts) as a user message until every
    `max_turns` turn is answered."""

    async def setup_task(self, task) -> None:
        self.colors_per_turn = task.info[
            "colors_per_turn"
        ]  # per-task input, from the task
        self.max_turns = task.info["max_turns"]  # per-task input
        self.turns = 0  # per-rollout mutable state

    def _content(self, colors: list[str], text: str) -> list[dict]:
        parts = []
        for color in colors:
            buf = BytesIO()
            Image.new("RGB", (100, 100), COLOR_RGB[color]).save(buf, format="PNG")
            url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts + [{"type": "text", "text": text}]

    async def respond(self, message: str) -> tuple[vf.Messages, bool]:
        self.turns += 1
        if self.turns >= self.max_turns:
            return [], True
        colors = self.colors_per_turn[self.turns]
        total = sum(len(self.colors_per_turn[t]) for t in range(self.turns + 1))
        last = self.turns == self.max_turns - 1
        text = (
            f"Here are {len(colors)} more squares. Combine your previous answer with these "
            f"new letters to output all {total} letters."
            if last
            else f"Here are {len(colors)} more squares."
        )
        return [{"role": "user", "content": self._content(colors, text)}], False


if __name__ == "__main__":
    ColorCodewordUser.run()
