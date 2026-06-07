import re
import random
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, cast

from pydantic import BaseModel

import verifiers.v1 as vf

try:
    import nltk
    import textarena as ta
except ImportError as e:
    raise ImportError(
        "TextArenaTaskset requires nltk and textarena. Install with: uv add tasksets"
    ) from e


class TextArenaState(Protocol):
    game_state: vf.JsonData
    game_info: dict[int, vf.JsonData]
    done: bool


class TextArenaRuntimeEnv(Protocol):
    state: TextArenaState

    def reset(self, num_players: int) -> None: ...

    def get_observation(self) -> tuple[int, str]: ...

    def step(self, action: str) -> object: ...


class TextArenaUserConfig(vf.UserConfig):
    pass


class TextArenaTasksetConfig(vf.TasksetConfig):
    id: str | None = "textarena"
    game: str
    user: vf.UserConfig | None = TextArenaUserConfig()
    num_train_examples: int = 2000
    num_eval_examples: int = 20
    seed: int = 0
    answer_state_key: str


class TextArenaSpec(BaseModel, extra="forbid"):
    game: str
    answer_state_key: str


class TextArenaTask(vf.Task):
    answer: str
    textarena: TextArenaSpec


ConfigT = TypeVar("ConfigT", bound=TextArenaTasksetConfig)


class TextArenaTaskset(vf.Taskset[ConfigT], Generic[ConfigT]):
    config: ConfigT
    task_type = TextArenaTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return self.textarena_tasks(
                num_examples=self.config.num_eval_examples,
                first_seed_offset=self.config.num_train_examples,
            )
        return self.textarena_tasks(
            num_examples=self.config.num_train_examples,
            first_seed_offset=0,
        )

    def textarena_tasks(
        self,
        *,
        num_examples: int,
        first_seed_offset: int,
    ) -> vf.Tasks:
        config = self.config
        if num_examples <= 0:
            return []
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        template = ta.make(env_id=config.game)
        assert isinstance(template, ta.Env)
        template.reset(num_players=1)
        _, initial_prompt = template.get_observation()
        if not isinstance(initial_prompt, str) or not initial_prompt:
            raise ValueError("TextArena initial prompt must be a non-empty string.")
        word_list = textarena_word_list(template)
        rng = random.Random(config.seed)
        for _ in range(first_seed_offset):
            rng.choice(word_list)
        return [
            {
                "prompt": [vf.UserMessage(content=initial_prompt)],
                "answer": rng.choice(word_list),
                "textarena": {
                    "game": config.game,
                    "answer_state_key": config.answer_state_key,
                },
            }
            for _ in range(num_examples)
        ]


class TextArenaSession:
    env: TextArenaRuntimeEnv | None

    def __init__(self):
        self.env = None

    def reset(self, game: str) -> TextArenaRuntimeEnv:
        env = ta.make(env_id=game)
        assert isinstance(env, ta.Env)
        self.env = cast(TextArenaRuntimeEnv, env)
        self.env.reset(num_players=1)
        return self.env


def textarena_word_list(env: object) -> list[str]:
    raw_words = getattr(env, "word_list", None)
    if isinstance(raw_words, dict):
        raw_words = [
            word
            for values in raw_words.values()
            for word in (values if isinstance(values, list | tuple) else [values])
        ]
    if not isinstance(raw_words, Sequence) or isinstance(raw_words, str | bytes):
        raise ValueError("TextArena environment must expose a word_list sequence.")
    words = [str(word) for word in raw_words]
    if not words:
        raise ValueError("TextArena word_list must not be empty.")
    return words


def content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(
        content, str | bytes | bytearray
    ):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return ""


class TextArenaUser(vf.User[TextArenaUserConfig]):
    session: TextArenaSession

    def start(self) -> None:
        self.session = TextArenaSession()

    @vf.user(
        args={
            "textarena": "task.textarena",
            "answer": "task.answer",
            "completion": "state.completion",
        },
        sets={
            "final_env_response": "state.extras.final_env_response",
            "stop_condition": "state.stop_condition",
        },
    )
    def respond(self, textarena: dict, answer: str, completion: list[dict]) -> dict:
        return textarena_respond(self.session, textarena, answer, completion)


def textarena_respond(
    session: TextArenaSession, textarena: dict, answer: str, completion: list[dict]
) -> dict:
    game = textarena.get("game")
    answer_state_key = textarena.get("answer_state_key")
    if not isinstance(game, str) or not isinstance(answer_state_key, str):
        raise TypeError("TextArena task config must contain string fields.")
    if not isinstance(answer, str) or not answer:
        raise TypeError("TextArena task requires a non-empty answer.")
    env = session.env or session.reset(game)
    env.state.game_state[answer_state_key] = answer

    assistant_messages = [
        message
        for message in completion
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    last_text = (
        content_text(assistant_messages[-1].get("content"))
        if assistant_messages
        else ""
    )
    matches = re.findall(r"<guess>(.*?)</guess>", last_text, re.DOTALL)
    guess = matches[-1].strip() if matches else ""
    env.step(guess)
    if env.state.done:
        reason = str(env.state.game_info[0]["reason"])
        return {
            "messages": [vf.UserMessage(content=reason).model_dump(mode="json")],
            "final_env_response": reason,
            "stop_condition": "textarena_done",
        }
    _, observation = env.get_observation()
    if not isinstance(observation, str):
        raise TypeError("TextArena observation must be a string.")
    return {"messages": [vf.UserMessage(content=observation).model_dump(mode="json")]}


def load_taskset(config: TextArenaTasksetConfig) -> TextArenaTaskset:
    return TextArenaTaskset(config=config)
