from typing import Literal

import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)

AdvantageMode = Literal["grpo", "gdpo"]

# GSM8K uses shorter solutions than competition math
# ~500 tokens = ~2000 characters
LENGTH_THRESHOLD = 2000


def load_environment(
    advantage_mode: AdvantageMode = "grpo",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
):
    """
    GSM8K environment with GDPO support (correctness + length rewards).

    Uses grade-school math problems which are suitable for smaller models
    like Qwen 2.5 1.5B, while still providing GDPO-testable multi-reward
    optimization with natural tension.

    This environment implements GDPO (arXiv:2601.05242) with 2 rewards:
    - correctness: binary, whether answer matches
    - length: binary, gated on correctness (only counts if correct)

    Args:
        advantage_mode: "grpo" or "gdpo"
        num_train_examples: Number of training examples (-1 for all)
        num_eval_examples: Number of eval examples (-1 for all)
    """
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def math_answer_reward_func(parser, completion, answer, **kwargs):
        """Binary correctness reward."""
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    def length_reward_func(completion, **kwargs):
        """
        Binary length reward: 1.0 if response is concise, 0.0 otherwise.
        Creates tension with correctness - longer reasoning helps accuracy,
        but if correct, shorter is better.
        """
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        return 1.0 if len(text) <= LENGTH_THRESHOLD else 0.0

    # GDPO: gate length on correctness (per paper arXiv:2601.05242)
    # Length reward only counts if the answer is correct
    # This prevents reward hacking where model optimizes length while being wrong
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

    # 2 rewards per GDPO paper: correctness + length (gated)
    rubric = vf.Rubric(
        parser=parser,
        funcs=[
            math_answer_reward_func,
            length_reward_func,
        ],
        weights=[1.0, 1.0],  # Equal weights per paper
        advantage_mode=advantage_mode,
        gates=gates,
    )

    # GSM8K: grade school math (easier than DeepScaleR)
    # Suitable for smaller models like Qwen 2.5 1.5B
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
