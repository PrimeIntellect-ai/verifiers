"""Seeded TextArena games driven by a colocated user simulator.

The task prompt and simulator reset use the same seed so an episode can be
reproduced. The simulator shares the harness runtime and writes the authoritative
outcome there; scoring reads that file instead of trusting conversational text.
"""

import copy
import itertools
import json
import random
from collections.abc import Iterator
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
    """Keep a seeded game alive across user turns in the harness process."""

    EXTRAS = ("ta",)

    async def setup(self) -> None:
        if not self.config.colocated:
            raise ValueError(
                "textarena's user simulator must be colocated: it hands the game outcome to scoring "
                "by writing OUTCOME_FILE into the harness's runtime workspace that `game_reward` reads "
                "back, so a non-colocated user (its own workspace) would always score 0. Set "
                "`--taskset.task.user.colocated true` (the default)."
            )
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    async def setup_task(self, task) -> None:
        # TextArena uses Python's process-global RNG during reset.
        random.seed(task.info["seed"])
        self.env = ta.make(env_id=task.info["game"])
        self.env.reset(num_players=1)

    @staticmethod
    def _latest_feedback(observation: str) -> str:
        """Drop repeated game history before sending the next user turn."""
        latest = observation.split("[GAME]")[-1].strip()
        return (
            latest.split("Feedback:")[-1].strip() if "Feedback:" in latest else latest
        )

    async def respond(self, message: str) -> vf.Messages:
        env = self.env
        env.step(message)
        if env.state.done:
            reward = float((env.state.rewards or {}).get(0, 0.0))
            reason = str(env.state.game_info[0]["reason"])
            with open(OUTCOME_FILE, "w") as f:
                json.dump({"reward": reward, "reason": reason}, f)
            self.state.game_over = True
            return [vf.UserMessage(content=reason)]
        _, observation = env.get_observation()
        return [vf.UserMessage(content=self._latest_feedback(str(observation)))]


class TextArenaTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig(colocated=True)


class TextArenaConfig(vf.TasksetConfig):
    game: Literal[
        "Wordle-v0",
        "Wordle-v0-long",
        "Hangman-v0",
        "WordLadder-v0",
        "WordSearch-v0",
    ]
    task: TextArenaTaskConfig = TextArenaTaskConfig()


class TextArenaData(vf.TaskData):
    info: dict
    """The game id and RNG seed used by the simulator."""


class TextArenaTask(vf.Task[TextArenaData, TextArenaState, TextArenaTaskConfig]):
    user = TextArenaUser

    @vf.stop
    async def game_over(self, trace: vf.Trace) -> bool:
        return trace.state.game_over

    @vf.reward(weight=1.0)
    async def game_reward(self, runtime: vf.Runtime) -> float:
        try:
            data = await runtime.read(OUTCOME_FILE)
        except (FileNotFoundError, OSError):
            # No outcome means the game never reached a terminal state.
            return 0.0
        return float(json.loads(data)["reward"])


class TextArenaTaskset(vf.Taskset[TextArenaTask, TextArenaConfig]):
    INFINITE = True

    def load(self) -> Iterator[TextArenaTask]:
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        template = ta.make(env_id=self.config.game)

        def observation(seed: int) -> str:
            random.seed(seed)
            env = copy.deepcopy(template)
            env.reset(num_players=1)
            return str(env.get_observation()[1])

        # Reuse the first prompt when a game's initial observation is
        # seed-invariant; otherwise each task must expose its seeded board.
        first = observation(0)
        seed_specific = observation(1) != first
        for i in itertools.count():
            yield TextArenaTask(
                TextArenaData(
                    idx=i,
                    name=f"{self.config.game}#{i}",
                    prompt=observation(i) if seed_specific else first,
                    system_prompt=SYSTEM_PROMPT,
                    info={"game": self.config.game, "seed": i},
                ),
                self.config.task,
            )


if __name__ == "__main__":
    TextArenaUser.run()
