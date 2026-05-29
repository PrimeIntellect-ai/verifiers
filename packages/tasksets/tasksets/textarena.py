import asyncio
import random
import re
from typing import Generic, TypeVar, cast

import verifiers as vf

try:
    import nltk
    import textarena as ta
except ImportError as e:
    raise ImportError(
        "TextArenaTaskset requires nltk and textarena. Install with: uv add tasksets"
    ) from e


class TextArenaUserConfig(vf.UserConfig):
    game: str
    answer_state_key: str


class TextArenaTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "textarena"
    game: str
    user: TextArenaUserConfig | None = None
    num_train_examples: int = 2000
    num_eval_examples: int = 20
    seed: int = 0
    answer_state_key: str


ConfigT = TypeVar("ConfigT", bound=TextArenaTasksetConfig)


class TextArenaTaskset(vf.Taskset[ConfigT], Generic[ConfigT]):
    config: ConfigT

    def load_user(self) -> vf.UserConfig:
        return self.config.user or TextArenaUserConfig(
            game=self.config.game,
            answer_state_key=self.config.answer_state_key,
        )

    def load_tasks(self, split: vf.TaskSplit = "train") -> list[vf.ConfigData]:
        num_examples = (
            self.config.num_train_examples
            if split == "train"
            else self.config.num_eval_examples
        )
        if num_examples <= 0:
            return []
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        template = ta.make(env_id=self.config.game)
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
        rng = random.Random(self.config.seed)
        first_seed_offset = 0 if split == "train" else self.config.num_train_examples
        for _ in range(first_seed_offset):
            rng.choice(word_list)
        return [
            {
                "prompt": [vf.UserMessage(content=initial_prompt)],
                "answer": rng.choice(word_list),
            }
            for index in range(num_examples)
        ]


class TextArenaUser(vf.User[TextArenaUserConfig]):
    async def get_response(
        self, task: vf.Task, state: vf.State, messages: list[vf.Message]
    ) -> list[vf.UserMessage]:
        ta_env = state.get("textarena_env")
        if ta_env is None:
            ta_env = ta.make(env_id=self.config.game)
            assert isinstance(ta_env, ta.Env)
            ta_env.reset(num_players=1)
            state["textarena_env"] = ta_env
        assert isinstance(ta_env, ta.Env)
        answer = task["answer"]
        assert isinstance(answer, str)
        assert answer
        game_state = cast(dict[str, object], ta_env.state.game_state)
        game_state[self.config.answer_state_key] = answer

        assistant_messages = vf.get_messages(messages, role="assistant")
        last_text = assistant_messages[-1].content if assistant_messages else ""
        assert isinstance(last_text, str)
        matches = re.findall(r"<guess>(.*?)</guess>", last_text, re.DOTALL)
        guess = matches[-1].strip() if matches else ""
        await asyncio.to_thread(ta_env.step, guess)
        if ta_env.state.done:
            game_info = cast(dict[int, dict[str, object]], ta_env.state.game_info)
            reason = str(game_info[0]["reason"])
            state["final_env_response"] = reason
            state.stop("textarena_done")
            return [vf.UserMessage(content=reason)]

        _, observation = await asyncio.to_thread(ta_env.get_observation)
        assert isinstance(observation, str)
        return [vf.UserMessage(content=observation)]


def load_taskset(
    config: TextArenaTasksetConfig,
) -> TextArenaTaskset[TextArenaTasksetConfig]:
    return TextArenaTaskset(config=config)
