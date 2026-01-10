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
):
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def math_answer_reward_func(parser, completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    # GDPO: gate format_reward on correctness
    # Format only counts if the answer is correct (harder gates easier)
    gates: dict = (
        {
            "format_reward_func": {
                "func": "math_answer_reward_func",
                "op": ">=",
                "value": 1.0,
            }
        }
        if advantage_mode == "gdpo"
        else {}
    )

    rubric = vf.Rubric(
        parser=parser,
        funcs=[math_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
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
