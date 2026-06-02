import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
    load_example_dataset,
)


def load_environment(**kwargs):
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    # env 1: gsm8k
    rubric1 = vf.MathRubric(parser=parser)
    rubric1.add_metric(parser.get_format_reward_func())

    def build_gsm8k_dataset():
        return load_example_dataset("gsm8k", split="train").select(range(1000))

    env1 = vf.SingleTurnEnv(
        dataset=build_gsm8k_dataset,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric1,
    )

    # env 2: math
    rubric2 = vf.MathRubric(parser=parser)
    rubric2.add_reward_func(parser.get_format_reward_func(), weight=0.2)

    def build_math_dataset():
        return load_example_dataset("math", split="train").select(range(1000))

    env2 = vf.SingleTurnEnv(
        dataset=build_math_dataset,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric2,
    )

    vf_env = vf.EnvGroup([env1, env2], env_names=["gsm8k", "math"])
    return vf_env
