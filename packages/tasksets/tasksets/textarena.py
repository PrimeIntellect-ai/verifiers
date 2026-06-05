import asyncio
import random
import re
from collections.abc import Sequence
from typing import Generic, TypeVar
from typing import Protocol, cast

import verifiers as vf

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
    objects: vf.ObjectsConfig = vf.ObjectsConfig.model_validate(
        {"session": "tasksets.textarena:TextArenaSession"}
    )


class TextArenaTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "textarena"
    game: str
    user: TextArenaUserConfig | None = TextArenaUserConfig()
    num_train_examples: int = 2000
    num_eval_examples: int = 20
    seed: int = 0
    answer_state_key: str


TextArenaConfigT = TypeVar("TextArenaConfigT", bound=TextArenaTasksetConfig)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, vf.TextContentPart):
                chunks.append(part.text)
            elif isinstance(part, dict):
                text = cast(dict[str, object], part).get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return ""


class TextArenaTaskset(vf.Taskset[TextArenaConfigT], Generic[TextArenaConfigT]):
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


class TextArenaUser(vf.User[TextArenaUserConfig]):
    async def get_response(
        self,
        task: vf.Task,
        state: vf.State,
        messages: list[vf.Message],
    ) -> list[vf.UserMessage]:
        session = await self.get_object("session", task, state)
        assert isinstance(session, TextArenaSession)
        textarena_config = task["textarena"]
        assert isinstance(textarena_config, dict)
        game = textarena_config["game"]
        assert isinstance(game, str)
        answer_state_key = textarena_config["answer_state_key"]
        assert isinstance(answer_state_key, str)
        ta_env = session.env or session.reset(game)
        answer = task["answer"]
        assert isinstance(answer, str)
        assert answer
        ta_env.state.game_state[answer_state_key] = answer

        assistant_messages = vf.get_messages(messages, role="assistant")
        last_text = (
            _content_text(assistant_messages[-1].content) if assistant_messages else ""
        )
        matches = re.findall(r"<guess>(.*?)</guess>", last_text, re.DOTALL)
        guess = matches[-1].strip() if matches else ""
        await asyncio.to_thread(ta_env.step, guess)
        if ta_env.state.done:
            reason = str(ta_env.state.game_info[0]["reason"])
            state["final_env_response"] = reason
            state.stop("textarena_done")
            return [vf.UserMessage(content=reason)]

        _, observation = await asyncio.to_thread(ta_env.get_observation)
        assert isinstance(observation, str)
        return [vf.UserMessage(content=observation)]


def load_taskset(
    config: TextArenaTasksetConfig,
) -> TextArenaTaskset:
    return TextArenaTaskset(config=config)
