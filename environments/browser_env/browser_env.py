"""
Browser Environment for vision-based browser control.

This environment uses a CUA (Computer Use Agent) server to provide
browser primitives (click, type, scroll, etc.) with screenshot feedback.

Usage:
    1. Start the CUA server:
       cd environments/browser_env/cua-server && pnpm start

    2. Run evaluation:
       vf-eval browser_env -n 5 -m gpt-4o
"""

from typing import Literal

from datasets import Dataset

import verifiers as vf

# Import BrowserEnv - will be available after adding to lazy imports
try:
    from verifiers.envs.browser_env import BrowserEnv
except ImportError:
    raise ImportError(
        "BrowserEnv requires aiohttp. Install with: uv pip install aiohttp"
    )


# ==================== Custom Reward Functions ====================


def efficiency_reward(state: vf.State, **kwargs) -> float:
    """
    Reward for completing task efficiently (fewer actions = higher reward).

    Linear decay from 1.0 at 1 action to 0.0 at max_actions.
    """
    max_actions = kwargs.get("max_actions", 20)
    trajectory = state.get("trajectory", [])
    num_actions = len(trajectory)

    if num_actions == 0:
        return 0.0

    # Linear decay: 1.0 at 1 action, 0.0 at max_actions
    return max(0.0, 1.0 - (num_actions - 1) / max_actions)


async def task_completion_reward(state: vf.State, **kwargs) -> float:
    """
    Placeholder reward for task completion.

    Override this function or add custom reward functions based on your task type:
    - URL matching: Check if browser navigated to target URL
    - Element presence: Check if specific element is visible in screenshot
    - Goal completion: Use a judge model to evaluate task completion
    - Text extraction: Check if model extracted correct information

    Returns:
        float: 0.0 (placeholder - implement based on task requirements)
    """
    # TODO: Implement based on task type
    # Examples:
    # - Check state["browser_state"]["url"] matches target
    # - Use vision model to verify element presence
    # - Compare extracted text to expected answer
    return 0.0


# ==================== Environment Loader ====================


def load_environment(
    server_url: str = "http://localhost:3000",
    env: Literal["LOCAL", "BROWSERBASE"] = "LOCAL",
    browserbase_api_key: str | None = None,
    browserbase_project_id: str | None = None,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    max_turns: int = 20,
    system_prompt: str | None = None,
    efficiency_weight: float = 0.1,
    task_completion_weight: float = 1.0,
    **kwargs,
) -> vf.Environment:
    """
    Load the Browser environment for vision-based browser control.

    Args:
        server_url: URL of the CUA server (default: http://localhost:3000)
        env: Browser environment type ("LOCAL" or "BROWSERBASE")
        browserbase_api_key: API key for Browserbase (if env="BROWSERBASE")
        browserbase_project_id: Project ID for Browserbase (if env="BROWSERBASE")
        viewport_width: Browser viewport width in pixels
        viewport_height: Browser viewport height in pixels
        max_turns: Maximum number of actions per rollout
        system_prompt: Custom system prompt (optional)
        efficiency_weight: Weight for efficiency reward (default: 0.1)
        task_completion_weight: Weight for task completion reward (default: 1.0)
        **kwargs: Additional arguments passed to BrowserEnv

    Returns:
        BrowserEnv instance configured with tools and rubrics
    """
    # Default system prompt for browser agent
    if system_prompt is None:
        system_prompt = """You are a browser automation agent. You can control a web browser using the provided tools.

Available tools:
- click(x, y, button): Click at coordinates
- double_click(x, y): Double-click at coordinates
- type_text(text): Type text into focused element
- keypress(keys): Press keyboard keys (e.g., "Enter", "Tab")
- scroll(x, y, scroll_x, scroll_y): Scroll at position
- goto(url): Navigate to URL
- back(): Go back in history
- forward(): Go forward in history
- wait(time_ms): Wait for specified milliseconds
- screenshot(): Capture current page state

After each action, you will receive a screenshot showing the current page state.
Analyze the screenshot to determine your next action.

Complete the given task efficiently using the minimum number of actions necessary."""

    # Create placeholder dataset
    # TODO: Replace with actual task dataset
    dataset = Dataset.from_dict(
        {
            "prompt": [
                "Navigate to google.com and search for 'weather today'",
                "Go to wikipedia.org and find the main page",
            ],
            "answer": [
                "weather search results",
                "wikipedia main page",
            ],
        }
    )

    # Create parser (no special parsing needed for browser tasks)
    parser = vf.Parser()

    # Create rubrics
    # 1. ToolRubric for basic tool usage metrics
    # Note: tools will be set by BrowserEnv, so we pass empty list here
    # and the BrowserEnv's tools will be used for metrics
    tool_rubric = vf.ToolRubric(tools=[])

    # 2. Custom browser rubric with efficiency and task completion rewards
    browser_rubric = vf.Rubric(
        funcs=[efficiency_reward, task_completion_reward],
        weights=[efficiency_weight, task_completion_weight],
        parser=parser,
    )

    # 3. Combine rubrics
    rubric = vf.RubricGroup(rubrics=[tool_rubric, browser_rubric])

    # Create and return the environment
    browser_env = BrowserEnv(
        server_url=server_url,
        env=env,
        browserbase_api_key=browserbase_api_key,
        browserbase_project_id=browserbase_project_id,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        max_turns=max_turns,
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )

    return browser_env

