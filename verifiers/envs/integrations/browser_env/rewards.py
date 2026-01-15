"""
Reward functions for browser environment evaluation.

These functions compute rewards based on agent performance during browser tasks.
"""

from typing import Callable, Literal

import verifiers as vf

__all__ = [
    "efficiency_reward",
    "judge_answer_reward",
    "JUDGE_PROMPT",
    "TASK_JUDGE_PROMPT",
    "get_judge_prompt",
]


# Custom judge prompt for evaluating browser agent answers (GAIA, smoke_test)
# Used when there is a known correct answer to compare against
JUDGE_PROMPT = """You are evaluating a browser automation agent's answer to a question.

Question:
```
{question}
```

Expected Answer:
```
{answer}
```

Agent's Response:
```
{response}
```

Does the agent's response contain the correct answer? The answer may be embedded in a longer response or phrased differently, but should convey the same information as the expected answer.

Respond "yes" if the agent's response contains the correct answer, "no" if it does not."""


# Task-based judge prompt for evaluating task completion (WebVoyager, Mind2Web)
# Used when there is no explicit answer, only a task to complete
TASK_JUDGE_PROMPT = """You are evaluating whether a browser automation agent successfully completed a web task.

Task Description:
```
{question}
```

Starting URL:
```
{start_url}
```

Agent's Actions and Final State:
```
{response}
```

Based on the agent's actions and final state, evaluate whether the task was successfully completed.

Consider:
1. Did the agent navigate to the correct website/page?
2. Did the agent perform the required actions (search, filter, click, fill forms, etc.)?
3. Did the agent reach a state that satisfies the task requirements?
4. Did the agent provide the requested information if applicable?

Respond "yes" if the task was successfully completed, "no" if it was not completed or only partially completed."""


def get_judge_prompt(
    benchmark: Literal["smoke_test", "gaia", "webvoyager", "onlineMind2Web"],
) -> str:
    """
    Get the appropriate judge prompt for a given benchmark.

    Args:
        benchmark: The benchmark type

    Returns:
        The appropriate judge prompt template string
    """
    if benchmark in ("smoke_test", "gaia"):
        # Answer-based evaluation
        return JUDGE_PROMPT
    elif benchmark in ("webvoyager", "onlineMind2Web"):
        # Task-based evaluation
        return TASK_JUDGE_PROMPT
    else:
        # Default to answer-based
        return JUDGE_PROMPT


def efficiency_reward(state: vf.State, **kwargs) -> float:
    """
    Reward for completing task efficiently (fewer actions = higher reward).

    Linear decay from 1.0 at 1 action to 0.0 at max_actions.

    Args:
        state: The current environment state containing trajectory
        **kwargs: Additional arguments, including:
            - max_actions: Maximum number of actions before reward drops to 0 (default: 20)

    Returns:
        float: Efficiency reward between 0.0 and 1.0
    """
    max_actions = kwargs.get("max_actions", 20)
    trajectory = state.get("trajectory", [])
    num_actions = len(trajectory)

    if num_actions == 0:
        return 0.0

    # Linear decay: 1.0 at 1 action, 0.0 at max_actions
    return max(0.0, 1.0 - (num_actions - 1) / max_actions)


async def judge_answer_reward(
    judge: Callable,
    prompt: str | list,
    completion: str | list,
    answer: str,
    state: vf.State,
) -> float:
    """
    LLM judge reward that compares the agent's final answer to the reference answer.

    This function is designed to work with vf.JudgeRubric. The judge callable
    is injected by the rubric and calls an LLM to evaluate correctness.

    Args:
        judge: Callable injected by JudgeRubric that calls the judge LLM
        prompt: The original prompt/question given to the agent
        completion: The agent's full response/trajectory
        answer: The expected/reference answer from the dataset
        state: The current environment state

    Returns:
        float: 1.0 if the judge determines the answer is correct, 0.0 otherwise
    """
    judge_response = await judge(prompt, completion, answer, state)

    # Parse the judge's response - look for "yes" to indicate correctness
    is_correct = "yes" in judge_response.lower()

    return 1.0 if is_correct else 0.0
