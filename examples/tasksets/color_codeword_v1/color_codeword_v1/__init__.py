"""color-codeword-v1 — decode a codeword from colored squares shown as images over turns.

The v1 port of the multimodal `color-codeword` env. Each color maps to a letter
(Red=A … Black=I); over several turns the model is shown colored squares (as images) and
outputs the accumulated codeword, giving the full sequence on the final turn.

The squares are delivered turn by turn by a colocated `vf.User` (see `user.py`) — the
default harness carries only a text instruction, so the images ride the user-simulator path
(which now preserves multimodal content). The whole episode is one native-v1 multimodal
rollout, exercising image input end to end (renderer → `TurnTokens.multi_modal_data`).
"""

import json
import random
import re
import sys

import verifiers.v1 as vf

# Color → letter mapping (9 visually distinct colors). RGB lives in the user simulator,
# which is the only side that renders the squares to images.
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

After each turn, output your accumulated codeword so far. Output ONLY the letters with NO spaces."""

INSTRUCTION = (
    "I'll show you colored squares over the next few turns. After each, reply with the "
    "accumulated codeword so far (letters only, no spaces). Reply 'ready' to begin."
)


def extract_codeword(text: str) -> str:
    """The decoded codeword in `text`: the longest standalone A–I run, else all A–I chars."""
    text = (text or "").upper()
    matches = re.findall(r"\b[A-I]+\b", text)
    return (
        max(matches, key=len)
        if matches
        else "".join(c for c in text if c in "ABCDEFGHI")
    )


class ColorCodewordConfig(vf.TasksetConfig):
    num_examples: int = 1000
    """Number of episodes (random codewords) to generate."""
    images_per_turn: int = 1
    """Colored squares shown per turn."""
    max_turns: int = 3
    """Turns of squares; codeword length is `images_per_turn * max_turns`."""
    seed: int = 42


class ColorCodewordTask(vf.Task):
    answer: str
    """The full codeword (one letter per square, in order)."""
    info: dict
    """`colors_per_turn` (the squares the user simulator reveals) and `num_turns`."""


class ColorCodewordTaskset(vf.Taskset[ColorCodewordTask, ColorCodewordConfig]):
    def load_tasks(self) -> list[ColorCodewordTask]:
        c = self.config
        assert c.images_per_turn >= 1 and c.max_turns >= 1
        rng = random.Random(c.seed)
        colors = list(COLOR_MAP)
        length = c.images_per_turn * c.max_turns
        tasks: list[ColorCodewordTask] = []
        for _ in range(c.num_examples):
            sequence = [rng.choice(colors) for _ in range(length)]
            colors_per_turn = [
                sequence[t * c.images_per_turn : (t + 1) * c.images_per_turn]
                for t in range(c.max_turns)
            ]
            tasks.append(
                ColorCodewordTask(
                    idx=len(tasks),
                    system_prompt=SYSTEM_PROMPT,
                    instruction=INSTRUCTION,
                    answer="".join(COLOR_MAP[x] for x in sequence),
                    info={"colors_per_turn": colors_per_turn, "num_turns": c.max_turns},
                )
            )
        return tasks

    def user(self, task: ColorCodewordTask) -> vf.User:
        info = {
            "colors_per_turn": task.info["colors_per_turn"],
            "num_turns": task.info["num_turns"],
        }
        return vf.User(
            name="user",
            command=[sys.executable, "-m", "color_codeword_v1.user"],
            env={"COLOR_CODEWORD_INFO": json.dumps(info)},
        )

    @vf.reward(weight=1.0)
    async def codeword_exact(self, task: ColorCodewordTask, trace: vf.Trace) -> float:
        responses = [m.content or "" for m in trace.assistant_messages]
        final = responses[-1] if responses else ""
        return 1.0 if extract_codeword(final) == task.answer else 0.0

    @vf.metric
    async def partial_match(self, task: ColorCodewordTask, trace: vf.Trace) -> float:
        if not task.answer:
            return 0.0
        responses = [m.content or "" for m in trace.assistant_messages]
        extracted = extract_codeword(responses[-1] if responses else "")
        return sum(1 for a, b in zip(task.answer, extracted) if a == b) / len(
            task.answer
        )


def load_taskset(config: ColorCodewordConfig) -> ColorCodewordTaskset:
    return ColorCodewordTaskset(config)
