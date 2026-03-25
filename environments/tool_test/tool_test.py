import random

from datasets import Dataset

import verifiers as vf


# dummy tools for sanity checking parallel tool calls
async def tool_A(x: int) -> int:
    """
    Tool for adding 1 to an integer.

    Args:
        x: The integer to add 1 to.

    Returns:
        The integer plus 1.
    """
    return x + 1


async def tool_B(x: str) -> str:
    """
    Tool for concatenating a string with "2".

    Args:
        x: The string to concatenate with "2".

    Returns:
        The string concatenated with "2".
    """
    return x + "2"


async def tool_C(x: float) -> float:
    """
    Tool for adding 3.0 to a float.

    Args:
        x: The float to add 3.0 to.

    Returns:
        The float plus 3.0.
    """
    return x + 3.0


async def tool_D(x: bool) -> bool:
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


def tool_call_reward_func(completion: vf.Messages, info: dict) -> float:
    # check if completion tool calls exactly matches info tool calls
    tool_calls = (
        (completion[-1].tool_calls or []) if completion[-1].role == "assistant" else []
    )
    called_tool_names = sorted([call.name for call in tool_calls])
    expected_tool_names = sorted(info["tool_names"])
    return 1.0 if called_tool_names == expected_tool_names else 0.0


def load_environment(
    num_train_examples: int = 1000, num_eval_examples: int = 100
) -> vf.ToolEnv:
    """
    Loads tool-test environment.
    """

    def build_dataset(count, seed_offset=0):
        rng = random.Random(42 + seed_offset)
        rows = []
        for _ in range(count):
            tool_names = rng.sample(tool_name_list, rng.randint(1, len(tool_name_list)))
            prompt = [
                {
                    "role": "user",
                    "content": f"Call the following tools with arguments of your choice: {tool_names}",
                }
            ]
            info = {"tool_names": tool_names}
            rows.append({"prompt": prompt, "info": info})
        return Dataset.from_list(rows)

    def build_train_dataset():
        return build_dataset(num_train_examples, seed_offset=0)

    def build_eval_dataset():
        return build_dataset(num_eval_examples, seed_offset=1)

    rubric = vf.Rubric(funcs=[tool_call_reward_func])
    vf_env = vf.ToolEnv(
        dataset=build_train_dataset,
        eval_dataset=build_eval_dataset,
        rubric=rubric,
        tools=tool_list,
        max_turns=1,
    )
    return vf_env
