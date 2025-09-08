import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset
from verifiers.utils.tools import python


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    **kwargs,
):
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    system_prompt = "Use python for all calculations (variables do not persist). Give your answer inside \\boxed{}."

    parser = vf.Parser(extract_fn=extract_boxed_answer)

    tool_rubric = vf.ToolRubric(tools=[python])
    math_rubric = vf.MathRubric(parser=parser)
    rubric = vf.RubricGroup(rubrics=[tool_rubric, math_rubric])

    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        tools=[python],
        max_turns=3,
        **kwargs,
    )

    return vf_env
