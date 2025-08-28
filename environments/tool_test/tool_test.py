import random

from datasets import Dataset

import verifiers as vf


# dummy tools for sanity checking parallel tool calls
def tool_A(x: int) -> int:
    """
    Tool for adding 1 to an integer.

    Args:
        x: The integer to add 1 to.

    Returns:
        The integer plus 1.
    """
    return x + 1


def tool_B(x: str) -> str:
    """
    Tool for concatenating a string with "2".

    Args:
        x: The string to concatenate with "2".

    Returns:
        The string concatenated with "2".
    """
    return x + "2"


def tool_C(x: float) -> float:
    """
    Tool for adding 3.0 to a float.

    Args:
        x: The float to add 3.0 to.

    Returns:
        The float plus 3.0.
    """
    return x + 3.0


def tool_D(x: bool) -> bool:
    """
    Tool for negating a boolean.

    Args:
        x: The boolean to negate.

    Returns:
        The negated boolean.
    """
    return not x


tool_list = [tool_A, tool_B, tool_C, tool_D]
tool_name_list = [tool.__name__ for tool in tool_list]


def tool_call_reward_func(completion, info):
    # check if completion tool calls exactly matches info tool calls
    tool_calls = completion[-1].get("tool_calls", [])
    called_tool_names = sorted([call.function.name for call in tool_calls])
    expected_tool_names = sorted(info["tool_names"])
    if called_tool_names == expected_tool_names:
        return 1.0
    else:
        return 0.0


def load_environment(
    num_train_examples: int = 1000, num_eval_examples: int = 100
) -> vf.ToolEnv:
    """
    Loads a custom environment.
    """

    # for each row, choose random non-empty subset of tools
    # prompt: call the following tools
    # completion: call the following tools

    # for each row, choose random non-empty subset of tools
    # prompt: call the following tools
    # completion: call the following tools
    train_rows = []
    eval_rows = []
    for i in range(num_train_examples + num_eval_examples):
        tool_names = random.sample(
            tool_name_list, random.randint(1, len(tool_name_list))
        )

        prompt = [
            {
                "role": "user",
                "content": f"Call the following tools with arguments of your choice: {tool_names}",
            }
        ]
        info = {"tool_names": tool_names}
        if i < num_train_examples:
            train_rows.append({"prompt": prompt, "info": info})
        else:
            eval_rows.append({"prompt": prompt, "info": info})

    dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)
    parser = vf.Parser()
    rubric = vf.ToolRubric(tools=tool_list)
    rubric.add_reward_func(tool_call_reward_func)
    vf_env = vf.ToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        tools=tool_list,
        max_turns=1,
    )
    return vf_env
