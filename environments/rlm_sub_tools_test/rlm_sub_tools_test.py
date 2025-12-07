"""
RLM Sub-Tools Test Environment.

Tests the sub-agent tools feature of RLMEnv:
1. Tools are documented in the system prompt (root model sees them)
2. Sub-LLMs can use the tools via the tool-calling loop
3. Works correctly without tools (fallback behavior)

The tasks are designed to REQUIRE tool use - sub-LLMs must call tools to get correct answers.
"""

from datasets import Dataset

import verifiers as vf
from verifiers.envs.rlm_env import RLMEnv


# =============================================================================
# Sub-Agent Tools
# =============================================================================


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate (e.g., "17 * 23 + 5")

    Returns:
        The result of the calculation as a string
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Predefined data for lookup tool
_LOOKUP_DATA = {
    "price": "42",
    "quantity": "15",
    "discount": "10",
    "tax_rate": "0.08",
    "base_cost": "100",
    "multiplier": "3",
    "offset": "7",
}


def lookup_data(key: str) -> str:
    """
    Look up a value from the data store.

    Args:
        key: The key to look up (one of: price, quantity, discount, tax_rate,
             base_cost, multiplier, offset)

    Returns:
        The value associated with the key, or an error message if not found
    """
    if key in _LOOKUP_DATA:
        return _LOOKUP_DATA[key]
    return f"Error: Key '{key}' not found. Available keys: {list(_LOOKUP_DATA.keys())}"


# =============================================================================
# Test Dataset
# =============================================================================

# Tasks that require tool use for correct answers
_TEST_TASKS = [
    {
        "query": "What is 17 * 23 + 89? Use the sub-LLM to calculate this.",
        "answer": "480",  # 17 * 23 = 391, 391 + 89 = 480
        "requires_tools": ["calculate"],
    },
    {
        "query": "Look up the 'price' value and multiply it by 3. What is the result?",
        "answer": "126",  # price=42, 42 * 3 = 126
        "requires_tools": ["lookup_data", "calculate"],
    },
    {
        "query": "What is the value of 'base_cost' plus 'offset' from the data store?",
        "answer": "107",  # base_cost=100, offset=7, 100 + 7 = 107
        "requires_tools": ["lookup_data", "calculate"],
    },
    {
        "query": "Calculate (15 * 8) + (22 / 2). Give the exact result.",
        "answer": "131.0",  # 15*8=120, 22/2=11, 120+11=131
        "requires_tools": ["calculate"],
    },
    {
        "query": "Look up 'quantity' and 'multiplier', then compute quantity * multiplier.",
        "answer": "45",  # quantity=15, multiplier=3, 15*3=45
        "requires_tools": ["lookup_data", "calculate"],
    },
]


# =============================================================================
# Environment
# =============================================================================


def load_environment(
    with_tools: bool = True,
    num_samples: int | None = None,
    **kwargs,
) -> vf.Environment:
    """
    Load the RLM sub-tools test environment.

    Args:
        with_tools: Whether to enable sub-agent tools (default: True)
        num_samples: Number of samples to include (default: all)
        **kwargs: Additional arguments passed to RLMEnv

    Returns:
        Configured RLMEnv instance
    """
    tasks = _TEST_TASKS[:num_samples] if num_samples else _TEST_TASKS

    dataset_rows = []
    for i, task in enumerate(tasks):
        dataset_rows.append(
            {
                "example_id": i,
                "prompt": [
                    {
                        "role": "user",
                        "content": f"""Task: {task["query"]}

Instructions:
1. Use llm_batch() to delegate this task to a sub-LLM
2. The sub-LLM has access to tools that can help solve this
3. Return the final numeric answer

Example:
```python
result = llm_batch(["{task["query"]}"])[0]
print(result)
```

After you have the answer, set it:
```python
answer["content"] = "<the numeric answer>"
answer["ready"] = True
```""",
                    }
                ],
                "task": "rlm-sub-tools-test",
                "answer": task["answer"],
                "info": {
                    "requires_tools": task["requires_tools"],
                    "with_tools_enabled": with_tools,
                },
            }
        )

    dataset = Dataset.from_list(dataset_rows)

    # Reward functions
    def exact_match_reward(state: vf.State) -> float:
        """Reward for exact match with expected answer."""
        final_answer = state.get("final_answer", "").strip()
        expected = state.get("answer", "").strip()

        # Normalize numeric comparisons
        try:
            final_num = float(final_answer)
            expected_num = float(expected)
            return 1.0 if abs(final_num - expected_num) < 0.01 else 0.0
        except (ValueError, TypeError):
            pass

        return 1.0 if final_answer == expected else 0.0

    def contains_answer_reward(state: vf.State) -> float:
        """Reward if the final answer contains the expected value."""
        final_answer = state.get("final_answer", "").strip()
        expected = state.get("answer", "").strip()
        return 1.0 if expected in final_answer else 0.0

    def tools_mentioned_reward(state: vf.State) -> float:
        """
        Metric: Check if model mentioned tools in its reasoning.
        Zero-weight - just for observability.
        """
        trajectory = state.get("trajectory", [])
        for step in trajectory:
            for msg in step.get("completion", []):
                content = msg.get("content", "")
                if "calculate" in content.lower() or "lookup" in content.lower():
                    return 1.0
        return 0.0

    rubric = vf.Rubric(
        funcs=[exact_match_reward, contains_answer_reward, tools_mentioned_reward],
        weights=[1.0, 0.0, 0.0],  # Only exact_match contributes to reward
    )

    # Configure tools based on with_tools flag
    sub_tools = [calculate, lookup_data] if with_tools else None

    env = RLMEnv(
        sub_tools=sub_tools,
        sub_tool_max_turns=5,
        max_iterations=20,
        max_output_length=4096,
        dataset=dataset,
        rubric=rubric,
        interception_host=kwargs.get("interception_host"),
    )

    return env
