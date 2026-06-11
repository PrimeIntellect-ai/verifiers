"""color-codeword-v1 — a multi-turn VLM codeword-decoding task (the v1 port of `color-codeword`).

Each turn shows colored squares that map to letters (Red=A, Green=B, ...); the model accumulates
the codeword across turns and, on the final turn, outputs the whole thing. Turn 0's squares are
seeded in the task's `instruction` (a `Messages` prompt carrying images); the later turns are
injected by a colocated `vf.User` (see `user.py`) the interception server drives after each
assistant turn. Reward is an exact match of the final codeword; a partial-match metric tracks
per-position accuracy. Images carry through the v1 message graph as `mm_kwargs` for training.
"""

import base64
import json
import random
import re
import sys
from io import BytesIO

from PIL import Image

import verifiers.v1 as vf

COLOR_MAP = {
    "red": "A",
    "green": "B",
    "blue": "C",
    "yellow": "D",
    "purple": "E",
    "cyan": "F",
    "orange": "G",
    "white": "H",
    "black": "I",
}
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

SYSTEM_PROMPT = """You will be shown colored squares across multiple turns. Each color maps to a letter:

Red=A, Green=B, Blue=C, Yellow=D, Purple=E, Cyan=F, Orange=G, White=H, Black=I

Example: Turn 1 shows Red, Blue. Turn 2 shows Green, Yellow. The full codeword is "ACBD" (all 4 letters in order).

After each turn, output your accumulated codeword so far. Output ONLY the letters with NO spaces."""


def color_data_url(color: str, size: int = 100) -> str:
    """A solid-color PNG square as a base64 `data:` URL."""
    img = Image.new("RGB", (size, size), COLOR_RGB[color])
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def image_content(colors: list[str], text: str) -> list[dict]:
    """An OpenAI content list — one image part per color, then the text. Used by the user
    simulator's later turns (the interception server re-types it, preserving the images)."""
    return [
        {"type": "image_url", "image_url": {"url": color_data_url(c)}} for c in colors
    ] + [{"type": "text", "text": text}]


def turn_text(turn: int, count: int, max_turns: int, total: int) -> str:
    """The user text shown alongside a turn's squares (mirrors the v0 wording)."""
    if turn == 0:
        return f"Here are {count} squares."
    if turn == max_turns - 1:
        return (
            f"Here are {count} more squares. Combine your previous answer with these new "
            f"letters to output all {total} letters."
        )
    return f"Here are {count} more squares."


def extract_codeword(text: str) -> str:
    """The longest standalone A–I run in `text` (case-insensitive); else all A–I characters."""
    text = text.upper()
    matches = re.findall(r"\b[A-I]+\b", text)
    return (
        max(matches, key=len)
        if matches
        else "".join(c for c in text if c in "ABCDEFGHI")
    )


class ColorCodewordConfig(vf.TasksetConfig):
    num_examples: int = 1000
    """Number of synthetic episodes to generate."""
    images_per_turn: int = 2
    """Colored squares shown per turn."""
    max_turns: int = 3
    """Turns per episode; the codeword has `images_per_turn * max_turns` letters."""
    seed: int = 42
    """RNG seed for reproducible color sequences."""


class ColorCodewordTask(vf.Task):
    answer: str
    """The full expected codeword (one letter per square shown, in order)."""
    info: dict
    """The episode the user simulator replays: `colors_per_turn` and `max_turns`."""


class ColorCodewordTaskset(vf.Taskset[ColorCodewordTask, ColorCodewordConfig]):
    def load_tasks(self) -> list[ColorCodewordTask]:
        c = self.config
        assert c.images_per_turn >= 1 and c.max_turns >= 1, (
            "need images_per_turn >= 1 and max_turns >= 1"
        )
        rng = random.Random(c.seed)
        colors = list(COLOR_MAP)
        length = c.images_per_turn * c.max_turns
        tasks: list[ColorCodewordTask] = []
        for idx in range(c.num_examples):
            sequence = [rng.choice(colors) for _ in range(length)]
            answer = "".join(COLOR_MAP[col] for col in sequence)
            colors_per_turn = [
                sequence[t * c.images_per_turn : (t + 1) * c.images_per_turn]
                for t in range(c.max_turns)
            ]
            turn0 = colors_per_turn[0]
            text = turn_text(0, len(turn0), c.max_turns, len(turn0))
            parts = [
                vf.ImageUrlContentPart(
                    image_url=vf.ImageUrlSource(url=color_data_url(col))
                )
                for col in turn0
            ] + [vf.TextContentPart(text=text)]
            tasks.append(
                ColorCodewordTask(
                    idx=idx,
                    instruction=[vf.UserMessage(content=parts)],
                    system_prompt=SYSTEM_PROMPT,
                    answer=answer,
                    info={"colors_per_turn": colors_per_turn, "max_turns": c.max_turns},
                )
            )
        return tasks

    def user(self, task: ColorCodewordTask) -> vf.User:
        return vf.User(
            name="user",
            command=[sys.executable, "-m", "color_codeword_v1.user"],
            env={"COLOR_CODEWORD_INFO": json.dumps(task.info)},
        )

    @vf.reward(weight=1.0)
    async def exact_match(self, task: ColorCodewordTask, trace: vf.Trace) -> float:
        responses = trace.assistant_messages
        last = (responses[-1].content if responses else "") or ""
        return 1.0 if extract_codeword(last) == task.answer else 0.0

    @vf.metric
    async def partial_match(self, task: ColorCodewordTask, trace: vf.Trace) -> float:
        if not task.answer:
            return 0.0
        responses = trace.assistant_messages
        last = (responses[-1].content if responses else "") or ""
        extracted = extract_codeword(last)
        return sum(1 for a, b in zip(task.answer, extracted) if a == b) / len(
            task.answer
        )


def load_taskset(config: ColorCodewordConfig) -> ColorCodewordTaskset:
    return ColorCodewordTaskset(config)
