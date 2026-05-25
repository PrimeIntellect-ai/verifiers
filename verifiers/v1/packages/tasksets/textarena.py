import asyncio
import random
import re
from copy import deepcopy
from typing import Protocol, cast

from verifiers.types import UserMessage
from verifiers.utils.message_utils import get_messages

from ...config import TasksetConfig
from ...task import Task
from ...taskset import Taskset
from ...types import ConfigData
from ...state import State
from ...user import User

try:
    import nltk
    import textarena as ta
except ImportError as e:
    raise ImportError(
        "TextArenaTaskset requires nltk and textarena. "
        "Install with: uv add 'verifiers[ta]'"
    ) from e


class TextArenaState(Protocol):
    game_state: ConfigData
    game_info: list[ConfigData]
    done: bool


class TextArenaEnv(Protocol):
    state: TextArenaState
    word_list: list[str] | dict[str, list[str] | tuple[str, ...] | str]

    def reset(self, *, num_players: int) -> None: ...

    def get_observation(self) -> tuple[int, str]: ...

    def step(self, action: str) -> None: ...


class TextArenaTasksetConfig(TasksetConfig):
    tasks: str | None = "load_tasks"
    eval_tasks: str | None = "load_eval_tasks"
    game: str
    num_train_examples: int = 2000
    num_eval_examples: int = 20
    seed: int = 0
    answer_state_key: str


class TextArenaTaskset(Taskset):
    config: TextArenaTasksetConfig

    def __init__(self, config: TextArenaTasksetConfig):
        assert isinstance(config, TextArenaTasksetConfig)

        self.template, self.initial_prompt, self.word_list = textarena_context(config)

        template = self.template
        shared_memo = {}
        env = template
        while hasattr(env, "env"):
            env = cast(TextArenaEnv, getattr(env, "env"))
        dictionary = getattr(env, "dictionary", None)
        if dictionary is not None:
            shared_memo[id(dictionary)] = dictionary
        word_list = getattr(env, "word_list", None)
        if word_list is not None:
            shared_memo[id(word_list)] = word_list

        def load_ta_env() -> TextArenaEnv:
            env = cast(TextArenaEnv, deepcopy(template, shared_memo.copy()))
            env.reset(num_players=1)
            return env

        super().__init__(config=config)
        if "user" not in self.config.model_fields_set:
            self.user = User(
                fn=self.textarena_user,
                objects={"ta_env": load_ta_env},
                bindings={"ta_env": "objects.ta_env"},
            )

    def format_observation(self, observation: str) -> str:
        return observation

    async def textarena_user(
        self, task: Task, state: State, ta_env: TextArenaEnv
    ) -> list[UserMessage]:
        answer = task["answer"]
        assert isinstance(answer, str)
        assert answer
        ta_env.state.game_state[self.config.answer_state_key] = answer

        assistant_messages = get_messages(
            cast(list, state.get("completion") or []), role="assistant"
        )
        last_text = assistant_messages[-1].content if assistant_messages else ""
        assert isinstance(last_text, str)
        matches = re.findall(r"<guess>(.*?)</guess>", str(last_text), re.DOTALL)
        guess = matches[-1].strip() if matches else ""
        await asyncio.to_thread(ta_env.step, guess)
        if ta_env.state.done:
            reason = str(ta_env.state.game_info[0]["reason"])
            state["final_env_response"] = reason
            state.stop("textarena_done")
            return [UserMessage(content=reason)]

        _, observation = await asyncio.to_thread(ta_env.get_observation)
        assert isinstance(observation, str)
        return [UserMessage(content=self.format_observation(str(observation)))]


def textarena_context(
    config: TextArenaTasksetConfig,
) -> tuple[TextArenaEnv, str, list[str]]:
    nltk.download("words", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    template = cast(TextArenaEnv, ta.make(env_id=config.game))
    assert isinstance(template, ta.Env)
    template.reset(num_players=1)
    _, initial_prompt = template.get_observation()
    assert isinstance(initial_prompt, str)
    assert initial_prompt
    words = template.word_list
    if isinstance(words, dict):
        words = [
            word
            for values in words.values()
            for word in (values if isinstance(values, (list, tuple)) else [values])
        ]
    word_list = [str(word) for word in words]
    assert word_list
    return template, initial_prompt, word_list


def textarena_row(
    initial_prompt: str, word_list: list[str], rng: random.Random, index: int
) -> ConfigData:
    return {
        "example_id": index,
        "prompt": [UserMessage(content=initial_prompt)],
        "answer": rng.choice(word_list),
    }


def load_tasks(config: TextArenaTasksetConfig) -> list[ConfigData]:
    _, initial_prompt, word_list = textarena_context(config)
    rng = random.Random(config.seed)
    return [
        textarena_row(initial_prompt, word_list, rng, index)
        for index in range(config.num_train_examples)
    ]


def load_eval_tasks(config: TextArenaTasksetConfig) -> list[ConfigData]:
    if config.num_eval_examples <= 0:
        return load_tasks(config=config)
    _, initial_prompt, word_list = textarena_context(config)
    rng = random.Random(config.seed)
    for _ in range(config.num_train_examples):
        rng.choice(word_list)
    return [
        textarena_row(initial_prompt, word_list, rng, index + config.num_train_examples)
        for index in range(config.num_eval_examples)
    ]
