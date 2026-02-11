"""
Browser DOM Mode Example Environment.

This example demonstrates using BrowserEnv with DOM mode (Stagehand SDK)
for natural language browser control.

DOM mode uses Stagehand to translate natural language commands into
browser actions like clicking, typing, and navigating.

Usage:
    prime eval run browser-dom-example -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
"""

import verifiers as vf
from verifiers.envs.integrations.browser_env import BrowserEnv
from datasets import Dataset

DOM_SYSTEM_PROMPT = """You are a browser automation agent using Stagehand's AI-driven tools.

Available tools:
- navigate(url): Navigate to a URL
- observe(instruction): Find possible actions matching the instruction
- act(instruction): Execute an action described in natural language
- extract(instruction, schema_json): Extract structured data from the page

Use natural language to describe what you want to do. Stagehand will intelligently
find elements and execute actions without needing CSS selectors or coordinates.

Complete the given task efficiently."""


def create_example_dataset() -> Dataset:
    """
    Create a simple inline dataset for the DOM mode hello world example.

    This dataset tests basic browser navigation and content extraction.
    """
    return Dataset.from_dict(
        {
            "question": [
                "What does the headline say on the primeintellect.ai homepage?"
            ],
            "answer": ["The Open Superintelligence Stack"],
            "start_url": ["https://primeintellect.ai"],
            "task_id": ["dom-example-0"],
        }
    )


# Judge prompt for evaluating answer correctness
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


async def judge_answer(
    judge,
    prompt: str | list,
    completion: str | list,
    answer: str,
    state: vf.State,
) -> float:
    """
    LLM judge reward that compares the agent's final answer to the reference answer.

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
    is_correct = "yes" in judge_response.lower()
    return 1.0 if is_correct else 0.0


def load_environment(
    project_id: str,
    max_turns: int = 10,
    judge_model: str = "gpt-4o-mini",
    system_prompt: str = DOM_SYSTEM_PROMPT,
    browserbase_api_key_var: str = "BROWSERBASE_API_KEY",
    stagehand_model: str = "openai/gpt-4o-mini",
    model_api_key_var: str = "MODEL_API_KEY",
    proxy_model_to_stagehand: bool = False,
    **kwargs,
) -> vf.Environment:
    """
    Load a DOM mode browser example environment.

    This is a self-contained "hello world" example demonstrating how to use
    BrowserEnv with DOM mode for natural language browser control.

    DOM mode uses Stagehand SDK for natural language browser control.
    Operations include: act, observe, extract, navigate.

    Args:
        max_turns: Maximum conversation turns (default: 10)
        judge_model: Model for judging task completion
        project_id: Browserbase project ID (required)
        browserbase_api_key_var: Env var name for Browserbase API key
        stagehand_model: Model for Stagehand operations (default: openai/gpt-4o-mini)
        model_api_key_var: Env var name for model API key
        proxy_model_to_stagehand: Route Stagehand LLM calls through evaluation model
        **kwargs: Additional arguments passed to BrowserEnv

    Returns:
        Configured BrowserEnv instance in DOM mode

    Example:
        >>> env = load_environment()
    """
    import os

    # Check required env vars upfront
    missing = []
    if not os.getenv(browserbase_api_key_var):
        missing.append(browserbase_api_key_var)
    if not os.getenv(model_api_key_var):
        missing.append(model_api_key_var)

    if missing:
        raise ValueError(
            f"Missing required environment variables for browser-dom-example:\n"
            f"  {', '.join(missing)}\n\n"
            f"Set these in your environment or .env file before running."
        )

    # Create inline dataset
    dataset = create_example_dataset()

    # Create judge rubric for evaluation
    rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )
    rubric.add_reward_func(judge_answer, weight=1.0)

    # Create BrowserEnv with DOM mode
    return BrowserEnv(
        mode="dom",
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        system_prompt=system_prompt,
        project_id=project_id,
        browserbase_api_key_var=browserbase_api_key_var,
        stagehand_model=stagehand_model,
        model_api_key_var=model_api_key_var,
        proxy_model_to_stagehand=proxy_model_to_stagehand,
        **kwargs,
    )
