"""color-codeword-v1 — a multi-turn VLM codeword-decoding task (the v1 port of `color-codeword`).

Each turn shows colored squares that map to letters (Red=A, Green=B, ...); the model accumulates
the codeword across turns and, on the final turn, outputs the whole thing. Turn 0's squares are
seeded in the task's `prompt` (a `Messages` prompt carrying images); the later turns are
injected by a scripted user — a closure the env's `rollout()` passes via `user=` — after each
assistant turn, ending the exchange once every turn is answered. Reward is an exact match of
the final codeword; a partial-match metric tracks per-position accuracy. Images carry through
the v1 message graph as `mm_kwargs` for training.
"""

import base64
import itertools
import random
import re
from collections.abc import Iterator
from io import BytesIO

from PIL import Image
from pydantic import Field

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

SYSTEM_PROMPT = """You will be shown colored squares across multiple turns. Each color maps to a letter:

Red=A, Green=B, Blue=C, Yellow=D, Purple=E, Cyan=F, Orange=G, White=H, Black=I

Example: Turn 1 shows Red, Blue. Turn 2 shows Green, Yellow. The full codeword is "ACBD" (all 4 letters in order).

After each turn, output your accumulated codeword so far. Output ONLY the letters with NO spaces."""

# Turns per episode; the codeword has `images_per_turn * MAX_TURNS` letters.
MAX_TURNS = 3
# RNG seed for reproducible color sequences.
SEED = 42


def color_data_url(color: str, size: int = 100) -> str:
    """A solid-color PNG square as a base64 `data:` URL."""
    img = Image.new("RGB", (size, size), COLOR_RGB[color])
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


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
    images_per_turn: int = Field(2, ge=1)
    """Colored squares shown per turn."""


class ColorCodewordTaskData(vf.TaskData):
    answer: str
    """The full expected codeword (one letter per square shown, in order)."""
    info: dict
    """The episode the env's scripted user replays: `colors_per_turn` and `max_turns`."""


class ColorCodewordTask(vf.Task[ColorCodewordTaskData, vf.State, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def exact_match(self, trace: vf.Trace) -> float:
        responses = trace.assistant_messages
        last = (responses[-1].content if responses else "") or ""
        return 1.0 if extract_codeword(last) == self.data.answer else 0.0

    @vf.metric
    async def partial_match(self, trace: vf.Trace) -> float:
        if not self.data.answer:
            return 0.0
        responses = trace.assistant_messages
        last = (responses[-1].content if responses else "") or ""
        extracted = extract_codeword(last)
        return sum(1 for a, b in zip(self.data.answer, extracted) if a == b) / len(
            self.data.answer
        )


class ColorCodewordEnv(vf.Environment):
    """Reveals each turn's colored squares after the prior answer: one user turn per
    assistant turn, until every `max_turns` turn is answered (then the exchange ends)."""

    async def rollout(self, task, agents):
        colors_per_turn = task.data.info["colors_per_turn"]
        max_turns = task.data.info["max_turns"]
        turns = 0

        async def reveal(message: str) -> vf.Messages:
            nonlocal turns
            turns += 1
            if turns >= max_turns:
                return []  # every turn answered — end the exchange
            colors = colors_per_turn[turns]
            total = sum(len(colors_per_turn[t]) for t in range(turns + 1))
            parts = [
                vf.ImageUrlContentPart(
                    image_url=vf.ImageUrlSource(url=color_data_url(color))
                )
                for color in colors
            ] + [
                vf.TextContentPart(text=turn_text(turns, len(colors), max_turns, total))
            ]
            return [vf.UserMessage(content=parts)]

        return [await agents["main"].run(task, user=reveal)]


class ColorCodewordTaskset(vf.Taskset[ColorCodewordTask, ColorCodewordConfig]):
    INFINITE = True

    def load(self) -> Iterator[ColorCodewordTask]:
        c = self.config
        rng = random.Random(SEED)
        colors = list(COLOR_MAP)
        color_urls = {color: color_data_url(color) for color in colors}
        length = c.images_per_turn * MAX_TURNS
        for idx in itertools.count():
            sequence = [rng.choice(colors) for _ in range(length)]
            answer = "".join(COLOR_MAP[col] for col in sequence)
            colors_per_turn = [
                sequence[t * c.images_per_turn : (t + 1) * c.images_per_turn]
                for t in range(MAX_TURNS)
            ]
            turn0 = colors_per_turn[0]
            text = turn_text(0, len(turn0), MAX_TURNS, len(turn0))
            parts = [
                vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=color_urls[col]))
                for col in turn0
            ] + [vf.TextContentPart(text=text)]
            yield ColorCodewordTask(
                ColorCodewordTaskData(
                    idx=idx,
                    prompt=[vf.UserMessage(content=parts)],
                    system_prompt=SYSTEM_PROMPT,
                    answer=answer,
                    info={
                        "colors_per_turn": colors_per_turn,
                        "max_turns": MAX_TURNS,
                    },
                ),
                c.task,
            )
