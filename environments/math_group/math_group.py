from typing import Literal

from datasets import load_dataset

import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
)

AdvantageMode = Literal["grpo", "gdpo"]

# Length threshold for binary length reward (characters)
# GDPO paper uses 4000 tokens; ~4 chars/token = ~16000 characters
# Responses shorter than this get reward=1, longer get reward=0
LENGTH_THRESHOLD = 16000


def load_environment(
    advantage_mode: AdvantageMode = "grpo",
    num_examples: int = -1,
):
    """
    Load the math_group environment using DeepScaleR-Preview dataset.

    This environment implements GDPO (arXiv:2601.05242) with 2 rewards:
    - correctness: binary, whether answer matches
    - length: binary, gated on correctness (only counts if correct)

    Args:
        advantage_mode: "grpo" or "gdpo"
        num_examples: Number of examples to use (-1 for all ~40k)
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

    # DeepScaleR-Preview: 40k competition-level math problems
    # Same dataset used in GDPO paper (arXiv:2601.05242)
    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")

    # Preprocess to match verifiers format (question -> prompt conversion handled by env)
    def preprocess(example):
        return {
            "question": example["problem"],
            "answer": example["answer"],
        }

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
