from typing import Dict, List, Tuple

from datasets import Dataset
from openai import OpenAI

from verifiers import (
    Messages,
    MultiTurnEnv,
    RewardFunc,
    State,
)
from verifiers.rubrics.math_rubric import MathRubric
from verifiers.utils.data_utils import load_example_dataset

SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


class DoubleCheckEnv(MultiTurnEnv):
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        system_prompt: str = SIMPLE_PROMPT,
        few_shot: List[Dict[str, str]] = [],
        **kwargs,
    ):
        super().__init__(
            client=client,
            model=model,
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            **kwargs,
        )
        self.rubric = MathRubric()

    def get_reward_funcs(self, **kwargs) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self, **kwargs) -> List[float]:
        return self.rubric.get_reward_weights()

    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        return len(state["responses"]) == 1

    def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Tuple[Messages, State]:
        return [{"role": "user", "content": "Are you sure?"}], state


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
):
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    vf_env = DoubleCheckEnv(dataset=dataset, system_prompt=SIMPLE_PROMPT, few_shot=[])
    return vf_env
