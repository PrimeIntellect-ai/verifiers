"""textarena_v1 — TextArena games as a v1 taskset, driven by a user simulator.

Each task is one episode of a TextArena game (the working example is Wordle). The model
plays by emitting moves; the framework's interception server drives a `vf.User` (the game
engine, see `server.py`) that replies with the game's feedback as a user turn — so a whole
game is one rollout of alternating assistant/user turns, and the harness/program never see
the simulator. The user simulator runs colocated in the harness's runtime (host-reachable
for the subprocess/docker runtimes), so this taskset uses the subprocess runtime.

Scoring is game-authoritative: when the episode ends the user simulator writes the game's
own outcome (`env.state.rewards`) to `OUTCOME_FILE` in the runtime, and the reward reads it
back — so the taskset needs no per-game guess parsing. Everything the simulator needs to set
up a game is carried in the task's `info` dict (game id + the secret word to seed).
"""

import json
import sys
from collections.abc import Sequence
from typing import Literal

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
    "use the exact move format the game requires. Think step-by-step first, then give your "
    "move as the only square-bracketed token in your reply — the game reads the first "
    "bracketed token, so don't put other words in brackets."
)

# The user simulator writes the game's outcome here (in the runtime workspace) and the
# reward reads it back; shared with `server.py`.
OUTCOME_FILE = "textarena_outcome.json"


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


class TextArenaConfig(vf.TasksetConfig):
    game: Literal[
        "Wordle-v0",
        "Wordle-v0-hardcore",
        "Wordle-v0-long",
        "Wordle-v0-long-hardcore",
        "Hangman-v0",
        "Hangman-v0-hardcore",
    ]
    """The TextArena game (required). Restricted to the tested single-secret-word games (the
    Wordle and Hangman families): the secret is one word drawn from the game's `word_list`,
    and the game reports its own win/partial outcome. The working example is "Wordle-v0";
    variants vary word length (Wordle long = 7) and dictionary size (hardcore = full English
    word list)."""


class TextArenaTask(vf.Task):
    info: dict
    """Everything the user simulator needs to set up the game: the `game` id and the secret
    `answer` to seed."""


class TextArenaTaskset(vf.Taskset[TextArenaTask, TextArenaConfig]):
    def load_tasks(self) -> list[TextArenaTask]:
        # One task per (lowercase) word; the eval (num_tasks / shuffle) selects. Capitalized
        # proper nouns in the hardcore lists are unwinnable (TextArena lowercases the guess
        # but not the stored secret), so drop them.
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        template = ta.make(env_id=self.config.game)
        template.reset(num_players=1)
        _, instruction = template.get_observation()
        words = [w for w in _word_list(template) if w.isalpha() and w == w.lower()]
        return [
            TextArenaTask(
                idx=i,
                name=f"{self.config.game}#{i}",
                instruction=str(instruction),
                system_prompt=SYSTEM_PROMPT,
                info={"game": self.config.game, "answer": word},
            )
            for i, word in enumerate(words)
        ]

    def user(self, task: TextArenaTask) -> vf.User:
        return vf.User(
            name="user",
            command=[sys.executable, "-m", "textarena_v1.server"],
            env={"TEXTARENA_INFO": json.dumps(task.info)},
        )

    @vf.reward(weight=1.0)
    async def game_reward(
        self, task: TextArenaTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        # The simulator wrote the game's own outcome to the runtime when the episode ended;
        # a missing file means the game never finished (e.g. every move was invalid).
        try:
            data = await runtime.read(OUTCOME_FILE)
        except (FileNotFoundError, OSError):
            return 0.0
        return float(json.loads(data)["reward"])


def load_taskset(config: TextArenaConfig) -> TextArenaTaskset:
    return TextArenaTaskset(config)
