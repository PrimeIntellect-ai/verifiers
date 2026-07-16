"""Seeded TextArena games, the game engine playing the user.

The task prompt and the engine reset share a seed so an episode can be reproduced.
The env's `rollout()` steps the real engine host-side, as the run's user: each
assistant move advances the game, the next observation comes back as the user turn,
and a finished game ends the exchange (no more messages) — its outcome scored
directly off the engine, in scope, with nothing crossing a process boundary.
"""

import copy
import itertools
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


def _latest_feedback(observation: str) -> str:
    """Drop repeated game history before sending the next user turn."""
    latest = observation.split("[GAME]")[-1].strip()
    return latest.split("Feedback:")[-1].strip() if "Feedback:" in latest else latest


class TextArenaConfig(vf.TasksetConfig):
    game: Literal[
        "Wordle-v0",
        "Wordle-v0-long",
        "Hangman-v0",
        "WordLadder-v0",
        "WordSearch-v0",
    ]


class TextArenaData(vf.TaskData):
    info: dict
    """The game id and RNG seed the env reproduces the episode from."""


class TextArenaTask(vf.Task[TextArenaData, vf.State, vf.TaskConfig]):
    pass


class TextArenaParams(vf.EnvParams):
    player: vf.AgentConfig = vf.AgentConfig()


class TextArenaEnv(vf.Environment[TextArenaParams]):
    def roles(self):
        return {"player": vf.Role(self.params.player)}

    async def rollout(self, task, agents):
        # TextArena uses Python's process-global RNG during reset; seed + make + reset
        # run with no await between them, so concurrent rollouts can't interleave.
        random.seed(task.data.info["seed"])
        game = ta.make(env_id=task.data.info["game"])
        game.reset(num_players=1)
        outcome: dict = {}

        async def engine(move: str) -> vf.Messages:
            game.step(move)
            if game.state.done:
                outcome["reward"] = float((game.state.rewards or {}).get(0, 0.0))
                outcome["reason"] = str(game.state.game_info[0]["reason"])
                return []  # game over — end the exchange
            _, observation = game.get_observation()
            return [vf.UserMessage(content=_latest_feedback(str(observation)))]

        trace = await agents["player"].run(task, user=engine)
        trace.record_reward("game_reward", outcome.get("reward", 0.0), 1.0)
        if "reason" in outcome:
            trace.info["game_outcome"] = outcome["reason"]
        return [trace]


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
