"""
Browser CUA Mode Example Environment.

This example demonstrates using BrowserEnv with CUA (Computer Use Agent) mode
for vision-based browser control.

CUA mode uses screenshots and vision models to interact with the browser,
providing low-level primitives like click, scroll, and type_text.

Usage:
    # Start CUA server first, then:
    prime eval run browser-cua-example -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
"""

from typing import Literal

import verifiers as vf
from verifiers.envs.integrations.browser_env import BrowserEnv
from datasets import Dataset


def create_example_dataset() -> Dataset:
    """
    Create a simple inline dataset for the CUA mode hello world example.

    This dataset tests basic browser navigation and content extraction
    using vision-based interactions.
    """
    return Dataset.from_dict(
        {
            "question": [
                "What does the headline say on the primeintellect.ai homepage?"
            ],
            "answer": ["The Open Superintelligence Stack"],
            "start_url": ["https://primeintellect.ai"],
            "task_id": ["cua-example-0"],
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
    max_turns: int = 15,
    judge_model: str = "gpt-4o-mini",
    server_url: str = "http://localhost:3000",
    browserbase_api_key: str | None = None,
    browserbase_project_id: str | None = None,
    env: Literal["LOCAL", "BROWSERBASE"] = "LOCAL",
    viewport_width: int = 1024,
    viewport_height: int = 768,
    save_screenshots: bool = False,
    keep_recent_screenshots: int | None = 2,
    proxies: bool = False,
    **kwargs,
) -> vf.Environment:
    """
    Load a CUA mode browser example environment.

    This is a self-contained "hello world" example demonstrating how to use
    BrowserEnv with CUA mode for vision-based browser control.

    CUA mode uses vision-based primitives for browser control.
    Requires a running CUA server (see cua-server/).

    Available tools in CUA mode:
    - click(x, y, button): Click at coordinates
    - double_click(x, y): Double-click at coordinates
    - type_text(text): Type text into focused element
    - keypress(keys): Press keyboard keys
    - scroll(x, y, scroll_x, scroll_y): Scroll at position
    - goto(url): Navigate to URL
    - back(): Go back in history
    - forward(): Go forward in history
    - wait(time_ms): Wait for specified milliseconds
    - screenshot(): Capture current page state

    Args:
        max_turns: Maximum conversation turns (default: 15)
        judge_model: Model for judging task completion
        server_url: CUA server URL (default: http://localhost:3000)
        browserbase_api_key: Browserbase API key (or set BROWSERBASE_API_KEY env var)
        browserbase_project_id: Browserbase project ID (or set BROWSERBASE_PROJECT_ID env var)
        env: Environment type - "LOCAL" or "BROWSERBASE"
        viewport_width: Browser viewport width (default: 1024)
        viewport_height: Browser viewport height (default: 768)
        save_screenshots: Save screenshots during execution (default: False)
        keep_recent_screenshots: Number of recent screenshots to keep in context
        proxies: Enable Browserbase proxies
        **kwargs: Additional arguments passed to BrowserEnv

    Returns:
        Configured BrowserEnv instance in CUA mode

    Example:
        >>> env = load_environment(server_url="http://localhost:3000")
    """
    # Create inline dataset
    dataset = create_example_dataset()

    # Create judge rubric for evaluation
    rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )
    rubric.add_reward_func(judge_answer, weight=1.0)

    # Create BrowserEnv with CUA mode (uses default system prompt)
    return BrowserEnv(
        mode="cua",
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        server_url=server_url,
        browserbase_api_key=browserbase_api_key,
        browserbase_project_id=browserbase_project_id,
        env=env,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        save_screenshots=save_screenshots,
        keep_recent_screenshots=keep_recent_screenshots,
        proxies=proxies,
        **kwargs,
    )
