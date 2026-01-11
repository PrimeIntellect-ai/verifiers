from typing import Literal

import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)

AdvantageMode = Literal["grpo", "gdpo"]

# Length threshold for binary length reward (characters)
# GDPO paper uses 4000 tokens; ~4 chars/token = ~16000 characters
# Responses shorter than this get reward=1, longer get reward=0
LENGTH_THRESHOLD = 16000


def load_environment(
    advantage_mode: AdvantageMode = "grpo",
):
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def math_answer_reward_func(parser, completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    def format_reward_func(completion, **kwargs):
        """
        Format reward: 1.0 if response contains \\boxed{...} format, 0.0 otherwise.
        This checks if the model followed the expected output format.
        """
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        return 1.0 if "\\boxed{" in text else 0.0

    def length_reward_func(completion, **kwargs):
        """
        Binary length reward: 1.0 if response is concise, 0.0 otherwise.
        This creates tension with correctness - shorter is easier but often wrong.
        """
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        return 1.0 if len(text) <= LENGTH_THRESHOLD else 0.0

    # GDPO: gate format and length on correctness
    # These rewards only count if the answer is correct (harder gates easier)
    # This prevents reward hacking where model games easy rewards while being wrong
    gates: dict = (
        {
            "format_reward_func": {
                "func": "math_answer_reward_func",
                "op": ">=",
                "value": 1.0,
            },
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
        funcs=[
            math_answer_reward_func,
            format_reward_func,
            length_reward_func,
        ],
        weights=[1.0, 0.2, 0.3],
        advantage_mode=advantage_mode,
        gates=gates,
    )

    # Use MATH dataset - competition-level math, much harder than GSM8K
    dataset = load_example_dataset("math", split="train").select(range(2000))

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
