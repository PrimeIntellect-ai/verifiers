import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    max_turns: int = 100,
    **kwargs,
):
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    system_prompt = (
        "Use python for all calculations. Give your answer inside \\boxed{}."
    )

    parser = vf.Parser(extract_fn=extract_boxed_answer)
    math_rubric = vf.MathRubric(parser=parser)
    vf_env = vf.PythonEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=math_rubric,
        max_turns=max_turns,
        **kwargs,
    )
    assert vf_env.tools is not None
    tool_rubric = vf.ToolRubric(tools=vf_env.tools)
    vf_env.rubric = vf.RubricGroup(rubrics=[tool_rubric, vf_env.rubric])
    return vf_env
