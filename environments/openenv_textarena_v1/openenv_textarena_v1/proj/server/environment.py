import random
from typing import Any
from uuid import uuid4

import nltk
import textarena as ta
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from server.models import TextArenaAction, TextArenaMessage, TextArenaObservation


class TextArenaEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        self._state = State()
        self.env = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> TextArenaObservation:
        random.seed(seed or 0)
        self.env = ta.make(env_id="Wordle-v0")
        self.env.reset(num_players=1)
        self._state = State(episode_id=episode_id or str(uuid4()))
        _, observation = self.env.get_observation()
        return TextArenaObservation(prompt=str(observation), reward=0.0)

    def step(
        self,
        action: TextArenaAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TextArenaObservation:
        assert self.env is not None
        done, info = self.env.step(action.message)
        _, observation = self.env.get_observation()
        text = str(observation)
        feedback = text.split("[GAME]")[-1].strip() or text
        rewards = self.env.state.rewards or {}
        reward = float(rewards.get(0, 0.0))
        self._state.step_count += 1
        return TextArenaObservation(
            messages=[TextArenaMessage(content=feedback)],
            reward=reward,
            done=done,
            metadata=info or {},
        )

    @property
    def state(self) -> State:
        return self._state
