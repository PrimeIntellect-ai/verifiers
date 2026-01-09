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
    # env 1: gsm8k
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def gsm8k_answer_reward_func(parser, completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    # GDPO: gate format_reward on correctness
    gates: dict = (
        {
            "format_reward_func": {
                "func": "gsm8k_answer_reward_func",
                "op": ">=",
                "value": 1.0,
            }
        }
        if advantage_mode == "gdpo"
        else {}
    )

    rubric1 = vf.Rubric(
        parser=parser,
        funcs=[gsm8k_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],
        advantage_mode=advantage_mode,
        gates=gates,
    )
    dataset1 = load_example_dataset("gsm8k", split="train").select(range(1000))
    env1 = vf.SingleTurnEnv(
        dataset=dataset1,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric1,
    )

    # env 2: math
    def math_answer_reward_func(completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    # GDPO: gate format_reward on correctness
    gates2: dict = (
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

    rubric2 = vf.Rubric(
        parser=parser,
        funcs=[math_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        advantage_mode=advantage_mode,
        gates=gates2,
    )
    dataset2 = load_example_dataset("math", split="train").select(range(1000))
    env2 = vf.SingleTurnEnv(
        dataset=dataset2,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric2,
    )

    vf_env = vf.EnvGroup([env1, env2], env_names=["gsm8k", "math"])
    return vf_env
