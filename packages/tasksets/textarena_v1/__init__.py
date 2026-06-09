"""textarena_v1 — TextArena games as a v1 taskset, driven by a user simulator.

Each task is one episode of a TextArena game (the working example is Wordle). The model
plays by emitting guesses; the framework's interception server drives a `vf.User` (the game
engine, see `server.py`) that replies with the game's feedback as a user turn — so a whole
game is one rollout of alternating assistant/user turns, and the harness/program never see
the simulator. The user simulator runs colocated in the harness's runtime (host-reachable
for the subprocess/docker runtimes), so this taskset uses the subprocess runtime.

Scoring is a pure function of the trace: the model wins by guessing the secret word.
"""

import re
import sys
from collections.abc import Sequence

import verifiers.v1 as vf

try:
    import nltk
    import textarena as ta
except ImportError as e:
    raise ImportError(
        "textarena_v1 requires nltk and textarena. Install with: uv add 'tasksets[textarena]'"
    ) from e

SYSTEM_PROMPT = (
    "You are a competitive game player. Read the game instructions carefully and always "
    "use the exact answer format the game requires. Think step-by-step first, then give "
    "your move as the only square-bracketed word in your reply (e.g. [crane]) — the game "
    "reads the first bracketed word, so don't put other words in brackets."
)

# TextArena parses the move as the FIRST bracketed token, via re.search(r"\[(\w+)\]") (see
# Wordle/env.py) — match that exactly so the reward scores the same word the game acted on.
_MOVE = re.compile(r"\[(\w+)\]")


class TextArenaConfig(vf.TasksetConfig):
    game: str
    """The TextArena game id (required; the working example is "Wordle-v0")."""


class TextArenaTask(vf.Task):
    answer: str
    """The secret word for this episode; the user simulator's game is seeded with it."""
    game: str


def _word_list(env: object) -> list[str]:
    """The game's word list, flattened (some games expose a dict of difficulty -> words)."""
    words = getattr(env, "word_list", None)
    if isinstance(words, dict):
        words = [
            w
            for values in words.values()
            for w in (values if isinstance(values, (list, tuple)) else [values])
        ]
    if not isinstance(words, Sequence) or isinstance(words, (str, bytes)):
        raise ValueError(
            f"TextArena game {getattr(env, 'env_id', '?')} exposes no word_list"
        )
    return [str(w) for w in words]


def _normalize(word: str) -> str:
    return word.strip().lower()


def _extract_guess(content: str) -> str | None:
    """The word TextArena acts on: the first bracketed token in the message (or None)."""
    match = _MOVE.search(content)
    return match.group(1) if match else None


def _guesses(trace: vf.Trace) -> list[str]:
    """The move per assistant turn that made one, in order."""
    moves = (_extract_guess(m.content or "") for m in trace.assistant_messages)
    return [g for g in moves if g]


def _wordle_marks(guess: str, answer: str) -> tuple[int, int]:
    """(#greens, #yellows) for a guess vs the answer — standard, letter-count-aware Wordle."""
    n = len(answer)
    g = _normalize(guess).ljust(n)[:n]
    a = answer.lower()
    greens = sum(1 for i in range(n) if g[i] == a[i])
    counts: dict[str, int] = {}
    for i in range(n):
        if g[i] != a[i]:
            counts[a[i]] = counts.get(a[i], 0) + 1
    yellows = 0
    for i in range(n):
        if g[i] != a[i] and counts.get(g[i], 0) > 0:
            counts[g[i]] -= 1
            yellows += 1
    return greens, yellows


class TextArenaTaskset(vf.Taskset[TextArenaTask, TextArenaConfig]):
    def load_tasks(self) -> list[TextArenaTask]:
        # One task per word in the game's list; the eval (num_tasks / shuffle) selects.
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        template = ta.make(env_id=self.config.game)
        template.reset(num_players=1)
        _, instruction = template.get_observation()
        return [
            TextArenaTask(
                idx=i,
                name=f"{self.config.game}#{i}",
                instruction=str(instruction),
                system_prompt=SYSTEM_PROMPT,
                answer=word,
                game=self.config.game,
            )
            for i, word in enumerate(_word_list(template))
        ]

    def user(self, task: TextArenaTask) -> vf.User:
        return vf.User(
            name="user",
            command=[sys.executable, "-m", "textarena_v1.server"],
            env={
                "TEXTARENA_GAME": task.game,
                "TEXTARENA_ANSWER": task.answer,
            },
        )

    @vf.reward(weight=1.0)
    async def correct(self, task: TextArenaTask, trace: vf.Trace) -> float:
        answer = _normalize(task.answer)
        return float(any(_normalize(g) == answer for g in _guesses(trace)))

    @vf.reward(weight=0.2)
    async def partial(self, task: TextArenaTask, trace: vf.Trace) -> float:
        # Best-guess shaping (green/yellow letters), recomputed from the trace; skipped once
        # solved (the win already scores 1.0 via `correct`).
        guesses = _guesses(trace)
        answer = _normalize(task.answer)
        n = len(task.answer)
        if not guesses or not n or any(_normalize(g) == answer for g in guesses):
            return 0.0
        scores = (
            (greens + 0.5 * yellows) / n
            for greens, yellows in (_wordle_marks(g, task.answer) for g in guesses)
        )
        return max(scores, default=0.0)

    @vf.metric
    async def num_guesses(self, trace: vf.Trace) -> float:
        return float(len(_guesses(trace)))


def load_taskset(config: TextArenaConfig) -> TextArenaTaskset:
    return TextArenaTaskset(config)
