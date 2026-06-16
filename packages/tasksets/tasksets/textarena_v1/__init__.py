"""textarena_v1 — TextArena games as a v1 taskset, driven by a user simulator.

Each task is one episode of a TextArena game (the working example is Wordle). The model
plays by emitting moves; the framework's interception server drives a `vf.User` (the game
engine, see `server.py`) that replies with the game's feedback as a user turn — so a whole
game is one rollout of alternating assistant/user turns, and the harness/program never see
the simulator. The user simulator runs colocated in the harness's runtime (host-reachable
for the subprocess/docker runtimes), so this taskset uses the subprocess runtime.

Scoring is game-authoritative: when the episode ends the user simulator writes the game's
own outcome (`env.state.rewards`) to `OUTCOME_FILE` in the runtime, and the reward reads it
back — so the taskset needs no per-game guess parsing. Each task is reproduced from an RNG
seed (carried in `info`): the taskset seeds the game to build the instruction and the
simulator re-seeds to the same episode, so no per-game word-list or state-key knowledge is
needed and any single-player TextArena game fits.
"""

import json
import random
import sys
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


class TextArenaConfig(vf.TasksetConfig):
    game: Literal[
        "Wordle-v0",
        "Wordle-v0-long",
        "Hangman-v0",
        "WordLadder-v0",
        "WordSearch-v0",
    ]
    """The TextArena game (required). The tested single-player games: Wordle / Wordle-long
    and Hangman (guess a hidden word), WordLadder (change one letter at a time to reach a
    target), and WordSearch (find words in a grid)."""
    num_tasks: int = 1000
    """How many seeded episodes to generate; the eval/orchestrator selects from these."""


class TextArenaTask(vf.Task):
    info: dict
    """What the user simulator needs to set up the game: the `game` id and the RNG `seed`
    that reproduces the exact episode this task's instruction was built from."""


class TextArenaTaskset(vf.Taskset[TextArenaTask, TextArenaConfig]):
    def load_tasks(self) -> list[TextArenaTask]:
        # One task per RNG seed; the simulator re-seeds to reproduce the same episode. Games
        # that embed the per-episode setup in the prompt (WordLadder's start/target,
        # WordSearch's grid) need the instruction built under each seed; games whose prompt
        # is seed-invariant (Wordle, Hangman) build it once.
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

        def observation(seed: int) -> str:
            random.seed(seed)
            env = ta.make(env_id=self.config.game)
            env.reset(num_players=1)
            return str(env.get_observation()[1])

        first = observation(0)
        seed_specific = observation(1) != first
        return [
            TextArenaTask(
                idx=i,
                name=f"{self.config.game}#{i}",
                instruction=observation(i) if seed_specific else first,
                system_prompt=SYSTEM_PROMPT,
                info={"game": self.config.game, "seed": i},
            )
            for i in range(self.config.num_tasks)
        ]

    def user(self, task: TextArenaTask) -> vf.User:
        return vf.User(
            name="user",
            command=[sys.executable, "-m", "tasksets.textarena_v1.server"],
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


Config = TextArenaConfig
Taskset = TextArenaTaskset
