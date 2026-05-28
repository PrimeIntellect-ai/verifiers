import asyncio
import random
import re
from copy import deepcopy
from typing import Generic, Protocol, TypeVar, cast

from verifiers.types import UserMessage
from verifiers.utils.message_utils import get_messages
from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.types import ConfigData
from verifiers.v1.user import User

try:
    import nltk
    import textarena as ta
except ImportError as e:
    raise ImportError(
        "TextArenaTaskset requires nltk and textarena. Install with: uv add tasksets"
    ) from e


class TextArenaRuntimeState(Protocol):
    game_state: dict[str, object]
    done: bool
    game_info: dict[int, dict[str, object]]


class TextArenaRuntimeEnv(Protocol):
    state: TextArenaRuntimeState


class TextArenaTasksetConfig(TasksetConfig):
    game: str
    num_train_examples: int = 2000
    num_eval_examples: int = 20
    seed: int = 0
    answer_state_key: str


ConfigT = TypeVar("ConfigT", bound=TextArenaTasksetConfig)


class TextArenaTaskset(Taskset[ConfigT], Generic[ConfigT]):
    config: ConfigT

    def __init__(self, config: ConfigT):
        assert isinstance(config, TextArenaTasksetConfig)
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        self.template = ta.make(env_id=config.game)
        assert isinstance(self.template, ta.Env)
        self.template.reset(num_players=1)
        _, self.initial_prompt = self.template.get_observation()
        assert isinstance(self.initial_prompt, str)
        assert self.initial_prompt
        words = self.template.word_list
        if isinstance(words, dict):
            words = [
                word
                for values in words.values()
                for word in (values if isinstance(values, (list, tuple)) else [values])
            ]
        self.word_list = [str(word) for word in words]
        assert self.word_list

        def load_ta_env() -> ta.Env:
            env = deepcopy(self.template)
            env.reset(num_players=1)
            return env

        super().__init__(config=config)
        if "user" not in self.config.model_fields_set:
            self.user = User(
                fn=self.textarena_user,
                objects={"ta_env": load_ta_env},
                bindings={"ta_env": "objects.ta_env"},
            )

    def load_tasks(self) -> list[ConfigData]:
        rng = random.Random(self.config.seed)
        return [
            self.textarena_task(rng, index)
            for index in range(self.config.num_train_examples)
        ]

    def load_eval_tasks(self) -> list[ConfigData]:
        if self.config.num_eval_examples <= 0:
            return []
        rng = random.Random(self.config.seed)
        for _ in range(self.config.num_train_examples):
            rng.choice(self.word_list)
        return [
            self.textarena_task(rng, index + self.config.num_train_examples)
            for index in range(self.config.num_eval_examples)
        ]

    def textarena_task(self, rng: random.Random, index: int) -> ConfigData:
        return {
            "example_id": index,
            "prompt": [UserMessage(content=self.initial_prompt)],
            "answer": rng.choice(self.word_list),
        }

    def format_observation(self, observation: str) -> str:
        return observation

    async def textarena_user(
        self, task: Task, state: State, ta_env: ta.Env
    ) -> list[UserMessage]:
        answer = task["answer"]
        assert isinstance(answer, str)
        assert answer
        runtime_env = cast(TextArenaRuntimeEnv, ta_env)
        runtime_env.state.game_state[self.config.answer_state_key] = answer

        assistant_messages = get_messages(
            cast(list, state.get("completion") or []), role="assistant"
        )
        last_text = assistant_messages[-1].content if assistant_messages else ""
        assert isinstance(last_text, str)
        matches = re.findall(r"<guess>(.*?)</guess>", last_text, re.DOTALL)
        guess = matches[-1].strip() if matches else ""
        await asyncio.to_thread(ta_env.step, guess)
        if runtime_env.state.done:
            reason = str(runtime_env.state.game_info[0]["reason"])
            state["final_env_response"] = reason
            state.stop("textarena_done")
            return [UserMessage(content=reason)]

        _, observation = await asyncio.to_thread(ta_env.get_observation)
        assert isinstance(observation, str)
        return [UserMessage(content=self.format_observation(observation))]
