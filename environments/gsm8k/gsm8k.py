import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    load_example_dataset,
)


def load_environment(
    system_prompt: str = BOXED_SYSTEM_PROMPT,
    num_train_examples=-1,
    num_eval_examples=-1,
):
    def build_dataset():
        dataset = load_example_dataset("gsm8k", split="train")
        if num_train_examples != -1:
            dataset = dataset.select(range(num_train_examples))
        return dataset

    def build_eval_dataset():
        eval_dataset = load_example_dataset("gsm8k", split="test")
        if num_eval_examples != -1:
            eval_dataset = eval_dataset.select(range(num_eval_examples))
        return eval_dataset

    rubric = vf.MathRubric()
    vf_env = vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        parser=rubric.parser,
        rubric=rubric,
    )
    return vf_env
