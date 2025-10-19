import random
from copy import deepcopy
from typing import Any, Callable

from datasets import Dataset

from verifiers import MultiTurnEnv, Rubric, XMLParser
from verifiers.types import Messages, State

try:
    import nltk  # type: ignore
except ImportError:
    print("nltk is not installed. Please install it with `uv pip install nltk`.")
    exit(1)


# monkey-patch nltk.download to always be quiet before importing textarena
_original_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: _original_nltk_download(  # type: ignore[invalid-assignment]
    *args, **{**kwargs, "quiet": True}
)

try:
    import textarena as ta  # type: ignore
except ImportError:
    print(
        "textarena is not installed. Please install it with `uv pip install textarena`."
    )
    exit(1)


class TextArenaEnv(MultiTurnEnv):
    """
    Wrapper for TextArena environments.
    """

    def __init__(
        self,
        game: str = "Wordle-v0",
        num_train_examples: int = 1000,
        num_eval_examples: int = 0,
        system_prompt: str | None = None,
        parser: XMLParser | None = None,
        rubric: Rubric | None = None,
        feedback_fn: Callable[[str], str] = lambda x: x,
        seed: int = 0,
        **kwargs,
    ):
        # default parser in textarena is XMLParser
        parser = parser or XMLParser(fields=["think", "guess"], answer_field="guess")

        self.game = game
        self.ta_env = ta.make(env_id=game)
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed
        self.feedback_fn = feedback_fn

        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        dataset, eval_dataset = self.ta_to_hf()

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )

    async def is_completed(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> bool:
        completed = await super().is_completed(messages, state, **kwargs)
        if "is_completed" in state and state["is_completed"]:
            state.pop("ta_env")
            return state["is_completed"]
        return False or completed

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> tuple[Messages, State]:
        # load env
        if "ta_env" not in state:
            ta_env = deepcopy(self.ta_env)
            ta_env.reset(num_players=1)
            ta_env.state.game_state["secret_word"] = state["answer"]
            state["ta_env"] = ta_env
        else:
            ta_env = state["ta_env"]
        # parse guess
        assert isinstance(messages[-1], dict)
        guess = self.parser.parse_answer(messages)
        # step env
        is_completed, _ = ta_env.step(str(guess))
        state["is_completed"] = is_completed
        _, observation = ta_env.get_observation()
        feedback = self.feedback_fn(observation)
        return [{"role": "user", "content": str(feedback)}], state

    def ta_to_hf(self) -> tuple[Dataset, Dataset | None]:
        dataset_rows = []
        eval_dataset_rows = []
        ta_env = ta.make(env_id=self.game)
        ta_env.reset(num_players=1)
        _, user_prompt = ta_env.get_observation()
        words = ta_env.word_list
        # set seed
        random.seed(self.seed)
        for i in range(self.num_train_examples + self.num_eval_examples):
            question = user_prompt
            answer = random.choice(words)
            if i < self.num_train_examples:
                dataset_rows.append({"question": question, "answer": answer})
            else:
                eval_dataset_rows.append({"question": question, "answer": answer})
        dataset = Dataset.from_list(dataset_rows)
        if self.num_eval_examples > 0:
            eval_dataset = Dataset.from_list(eval_dataset_rows)
        else:
            eval_dataset = None
        return dataset, eval_dataset
