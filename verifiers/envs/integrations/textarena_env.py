import logging
import random
from copy import deepcopy
from typing import Any, Callable, cast

from datasets import Dataset

import verifiers as vf

try:
    import nltk  # type: ignore
except ImportError as e:
    raise ImportError(
        "TextArenaEnv requires nltk. Install with: uv add 'verifiers[ta]'"
    ) from e


# monkey-patch nltk.download to always be quiet before importing textarena
_original_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: _original_nltk_download(  # type: ignore[invalid-assignment]
    *args, **{**kwargs, "quiet": True}
)

try:
    import textarena as ta  # type: ignore
except ImportError as e:
    raise ImportError(
        "TextArenaEnv requires textarena. Install with: uv add 'verifiers[ta]'"
    ) from e


class TextArenaEnv(vf.MultiTurnEnv):
    """
    Wrapper for TextArena environments.
    """

    def __init__(
        self,
        game: str = "Wordle-v0",
        num_train_examples: int = 1000,
        num_eval_examples: int = 0,
        system_prompt: str | None = None,
        parser: vf.XMLParser | None = None,
        rubric: vf.Rubric | None = None,
        feedback_fn: Callable[[str], str] = lambda x: x,
        seed: int = 0,
        **kwargs,
    ):
        # default parser in textarena is XMLParser
        parser = parser or vf.XMLParser(fields=["think", "guess"], answer_field="guess")

        self.game = game
        self.ta_env = ta.make(env_id=game)
        self.ta_env.reset(num_players=1)
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed
        self.feedback_fn = feedback_fn
        self.logger = logging.getLogger(__name__)

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

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        ta_env = deepcopy(self.ta_env)
        ta_env.state.game_state["secret_word"] = state["answer"]
        state["ta_env"] = ta_env
        return state

    @vf.cleanup
    async def cleanup_ta_env(self, state: vf.State):
        state.pop("ta_env", None)

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        ta_env = state["ta_env"]
        guess = self.parser.parse_answer(messages)
        self.logger.debug(f"Parsed {guess=}")
        ta_env.step(str(guess))

        if ta_env.state.done:
            self.logger.debug(f"Game completed! {ta_env.state.game_info=}")
            response = cast(
                vf.Messages,
                [{"role": "user", "content": ta_env.state.game_info[0]["reason"]}],
            )
            state["final_env_response"] = response
            return response
        else:
            _, observation = ta_env.get_observation()
            self.logger.debug(f"Got {observation=}")
            feedback = self.feedback_fn(observation)
            self.logger.debug(f"Parsed {feedback=}")
            return cast(vf.Messages, [{"role": "user", "content": str(feedback)}])

    def ta_to_hf(self) -> tuple[Dataset, Dataset | None]:
        dataset_rows = []
        eval_dataset_rows = []
        _, user_prompt = self.ta_env.get_observation()
        words = self.ta_env.word_list
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
