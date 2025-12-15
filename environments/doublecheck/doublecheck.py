import verifiers as vf
from verifiers.types import Messages, State
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


class DoubleCheckEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @vf.stop
    async def double_checked(self, state: State) -> bool:
        return len(state["trajectory"]) == 2

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Generate a response from the environment."""
        return [{"role": "user", "content": "Are you sure?"}]


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
):
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    rubric = MathRubric()
    vf_env = DoubleCheckEnv(
        dataset=dataset,
        system_prompt=SIMPLE_PROMPT,
        few_shot=[],
        parser=rubric.parser,
        rubric=rubric,
    )
    return vf_env
