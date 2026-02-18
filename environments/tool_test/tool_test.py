import random

from datasets import Dataset

import verifiers as vf
from verifiers.utils.tool_registry import register_tool


# dummy tools for sanity checking parallel tool calls
@register_tool("tool-test", "tool_A")
async def tool_A(x: int) -> int:
    """
    Tool for adding 1 to an integer.

    Args:
        x: The integer to add 1 to.

    Returns:
        The integer plus 1.
    """
    return x + 1


@register_tool("tool-test", "tool_B")
async def tool_B(x: str) -> str:
    """
    Tool for concatenating a string with "2".

    Args:
        x: The string to concatenate with "2".

    Returns:
        The string concatenated with "2".
    """
    return x + "2"


@register_tool("tool-test", "tool_C")
async def tool_C(x: float) -> float:
    """
    Tool for adding 3.0 to a float.

    Args:
        x: The float to add 3.0 to.

    Returns:
        The float plus 3.0.
    """
    return x + 3.0


@register_tool("tool-test", "tool_D")
async def tool_D(x: bool) -> bool:
    """
    Tool for negating a boolean.

    Args:
        x: The boolean to negate.

    Returns:
        The negated boolean.
    """
    return not x


DEFAULT_TOOL_LIST = [tool_A, tool_B, tool_C, tool_D]


def tool_call_reward_func(completion, info):
    # check if completion tool calls exactly matches info tool calls
    tool_calls = completion[-1].get("tool_calls", [])
    called_tool_names = sorted(
        [call.get("function", {}).get("name", "") for call in tool_calls]
    )
    expected_tool_names = sorted(info["tool_names"])
    if called_tool_names == expected_tool_names:
        return 1.0
    else:
        return 0.0


def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    tools: list | None = None,
) -> vf.ToolEnv:
    """
    Loads tool-test environment.
    """

    # Use provided tools or fall back to default
    if tools is None:
        tools = DEFAULT_TOOL_LIST

    # Extract tool names from ACTUAL tools being used (not hardcoded list)
    actual_tool_names = [tool.__name__ for tool in tools]

    # Handle empty tools case
    if not actual_tool_names:
        # Create empty datasets when no tools available
        dataset = Dataset.from_list([])
        eval_dataset = Dataset.from_list([])
        rubric = vf.Rubric(funcs=[tool_call_reward_func])
        vf_env = vf.ToolEnv(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            tools=tools,
            max_turns=1,
        )
        return vf_env

    train_rows = []
    eval_rows = []
    for i in range(num_train_examples + num_eval_examples):
        # Sample from actual available tools only
        tool_names = random.sample(
            actual_tool_names, random.randint(1, len(actual_tool_names))
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
    rubric = vf.Rubric(funcs=[tool_call_reward_func])
    vf_env = vf.ToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        tools=tools,
        max_turns=1,
    )
    return vf_env
