"""textarena — TextArena games as a v1 taskset, driven by a user simulator.

Each task is one episode of a TextArena game (the working example is Wordle). The model
plays by emitting moves; the framework's interception server drives a `vf.User` (the game
engine, `TextArenaUser` below) that replies with the game's feedback as a user turn — so a whole
game is one rollout of alternating assistant/user turns, and the harness/program never see
the simulator. The user simulator runs colocated in the harness's runtime (host-reachable
for the subprocess/docker runtimes), so this taskset uses the subprocess runtime.

Scoring is game-authoritative: when the episode ends the user simulator writes the game's
own outcome (`env.state.rewards`) to `OUTCOME_FILE` in the runtime, and the reward reads it
back — so the taskset needs no per-game guess parsing. Each task is reproduced from an RNG
seed (carried in `info`): the taskset seeds the game to build the prompt and the
simulator re-seeds to the same episode, so no per-game word-list or state-key knowledge is
needed and any single-player TextArena game fits.
"""

import copy
import json
import random
from typing import Literal

import verifiers.v1 as vf

try:
    import nltk
    import textarena as ta
except ImportError as e:
    raise ImportError(
        "textarena requires nltk and textarena. Install with: uv add 'verifiers[ta]'"
    ) from e

SYSTEM_PROMPT = (
    "You are a competitive game player. Read the game instructions carefully and always "
    "use the exact move format the game requires. Think step-by-step first, then give your "
    "move as the only square-bracketed token in your reply — the game reads the first "
    "bracketed token, so don't put other words in brackets."
)

OUTCOME_FILE = "textarena_outcome.json"


class TextArenaState(vf.State):
    game_over: bool = False


class TextArenaUser(vf.User[vf.UserConfig, TextArenaState]):
    """The TextArena game engine as a framework-driven conversation partner. Holds one game in
    memory (set up from the task's `game` id + RNG `seed`, reproducing the taskset's episode) and,
    per `respond`, steps the game with the model's move and returns the next observation as a user
    turn plus whether the episode is over. When the game ends it writes the game's own outcome to
    `OUTCOME_FILE` in the runtime, which the reward reads back."""

    async def setup(self) -> None:
        if not self.config.colocated:
            raise ValueError(
                "textarena's user simulator must be colocated: it hands the game outcome to scoring "
                "by writing OUTCOME_FILE into the harness's runtime workspace that `game_reward` reads "
                "back, so a non-colocated user (its own workspace) would always score 0. Set "
                "`--taskset.task.user.colocated true` (the default)."
            )
        import nltk

        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    async def setup_task(self, task) -> None:
        # textarena derives a game's whole setup from the global RNG at reset, so seeding it
        # reproduces the exact episode the taskset built the prompt from — no per-game keys.
        import random

        import textarena

        self.env = textarena.make(
            env_id=task.info["game"]
        )  # per-task input, from the task
        random.seed(task.info["seed"])  # per-task input
        self.env.reset(num_players=1)

    @staticmethod
    def _latest_feedback(observation: str) -> str:
        """Trim feedback to the latest block (after `Feedback:` in the last `[GAME]` message) so
        each injected user turn stays small and doesn't duplicate the history."""
        latest = observation.split("[GAME]")[-1].strip()
        return (
            latest.split("Feedback:")[-1].strip() if "Feedback:" in latest else latest
        )

    async def respond(self, message: str) -> vf.Messages:
        import json

        env = self.env
        env.step(
            message
        )  # TextArena parses the bracketed move out of the message itself
        if env.state.done:
            reward = float((env.state.rewards or {}).get(0, 0.0))
            reason = str(env.state.game_info[0]["reason"])
            with open(OUTCOME_FILE, "w") as f:
                json.dump({"reward": reward, "reason": reason}, f)
            self.state.game_over = True
            return [{"role": "user", "content": reason}]
        _, observation = env.get_observation()
        return [{"role": "user", "content": self._latest_feedback(str(observation))}]


class TextArenaTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig(colocated=True)
    """Colocated is required, not a default: the simulator hands the game outcome to scoring by
    writing `OUTCOME_FILE` into the runtime workspace that `game_reward` reads back, so it must share
    the harness's runtime/workdir (a non-colocated user runs in its own workspace → reward always 0)."""


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
    task: TextArenaTaskConfig = TextArenaTaskConfig()


class TextArenaData(vf.TaskData):
    info: dict
    """What the user simulator needs to set up the game: the `game` id and the RNG `seed`
    that reproduces the exact episode this task's prompt was built from."""


class TextArenaTask(vf.Task[TextArenaData, TextArenaState, TextArenaTaskConfig]):
    user = TextArenaUser
    # Built with the task config's `user` field (resolved by `Task.server_config`);
    # colocated is required — see `TextArenaTaskConfig.user`.

    @vf.stop
    async def game_over(self, trace: vf.Trace) -> bool:
        return trace.state.game_over

    @vf.reward(weight=1.0)
    async def game_reward(self, runtime: vf.Runtime) -> float:
        # The simulator wrote the game's own outcome to the runtime when the episode ended;
        # a missing file means the game never finished (e.g. every move was invalid).
        try:
            data = await runtime.read(OUTCOME_FILE)
        except (FileNotFoundError, OSError):
            return 0.0
        return float(json.loads(data)["reward"])


class TextArenaTaskset(vf.Taskset[TextArenaTask, TextArenaConfig]):
    def load(self) -> list[TextArenaTask]:
        # One task per RNG seed; the simulator re-seeds to reproduce the same episode. Games
        # that embed the per-episode setup in the prompt (WordLadder's start/target,
        # WordSearch's grid) need the prompt built under each seed; games whose prompt
        # is seed-invariant (Wordle, Hangman) build it once.
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        template = ta.make(env_id=self.config.game)

        def observation(seed: int) -> str:
            random.seed(seed)
            env = copy.deepcopy(template)
            env.reset(num_players=1)
            return str(env.get_observation()[1])

        first = observation(0)
        seed_specific = observation(1) != first
        return [
            TextArenaTask(
                TextArenaData(
                    idx=i,
                    name=f"{self.config.game}#{i}",
                    prompt=observation(i) if seed_specific else first,
                    system_prompt=SYSTEM_PROMPT,
                    info={"game": self.config.game, "seed": i},
                ),
                self.config.task,
            )
            for i in range(self.config.num_tasks)
        ]


if __name__ == "__main__":
    TextArenaUser.run()
