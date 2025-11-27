import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def load_environment(
    use_diamond: bool = True, use_think: bool = True
) -> vf.Environment:
    if use_diamond:
        eval_dataset = load_example_dataset("gpqa_diamond", "train")
    else:
        eval_dataset = load_example_dataset("gpqa_main", "train")
    system_prompt = """Give the letter of the correct answer inside \\boxed{...}."""
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def correct_answer_reward_func(completion, answer, **kwargs) -> float:
        response = parser.parse_answer(completion) or ""
        return 1.0 if response.startswith(str(answer)) else 0.0

    rubric = vf.Rubric(parser=parser, funcs=[correct_answer_reward_func], weights=[1.0])
    vf_env = vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    # judge_rubric = vf.JudgeRubric()

    # async def judge_reward(judge, prompt, completion, answer, state):
    #     judge_response = await judge(prompt, completion, answer, state)
    #     return 1.0 if "yes" in judge_response.lower() else 0.0

    # judge_rubric.add_reward_func(judge_reward, 1.0)
    # vf_env.rubric = vf.RubricGroup([judge_rubric, vf_env.rubric])
    return vf_env
