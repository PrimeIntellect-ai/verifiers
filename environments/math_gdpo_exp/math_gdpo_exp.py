from typing import Literal

import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)

AdvantageMode = Literal["grpo", "gdpo"]


def load_environment(
    advantage_mode: AdvantageMode = "grpo",
    length_threshold: int = 500,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
):
    """
    GSM8K environment with GDPO support (arXiv:2601.05242).

    Two rewards: correctness (binary) and length (gated on correctness).
    """
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def math_answer_reward_func(parser, completion, answer, **kwargs):
        """Binary correctness reward."""
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    def length_reward_func(completion, **kwargs):
        """Binary length reward: 1.0 if under threshold, 0.0 otherwise."""
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        return 1.0 if len(text) <= length_threshold else 0.0

    def response_length(completion, **kwargs) -> float:
        """Track response length for logging."""
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        return float(len(text))

    # Gate length reward on correctness (prevents reward hacking)
    gates: dict = (
        {
            "length_reward_func": {
                "func": "math_answer_reward_func",
                "op": ">=",
                "value": 1.0,
            },
        }
        if advantage_mode == "gdpo"
        else {}
    )

    rubric = vf.Rubric(
        parser=parser,
        funcs=[math_answer_reward_func, length_reward_func],
        weights=[1.0, 1.0],
        advantage_mode=advantage_mode,
        gates=gates,
    )
    rubric.add_metric(response_length)

    dataset = load_example_dataset("gsm8k", split="train")
    if num_train_examples > 0:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    eval_dataset = load_example_dataset("gsm8k", split="test")
    if num_eval_examples > 0:
        eval_dataset = eval_dataset.select(
            range(min(num_eval_examples, len(eval_dataset)))
        )

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
