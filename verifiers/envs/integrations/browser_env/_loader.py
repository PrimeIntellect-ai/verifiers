"""
Browser Environment loader with load_environment() entry point.

Usage:
    from verifiers.envs.integrations.browser_env import load_environment

    # DOM mode (natural language)
    env = load_environment(mode="dom", benchmark="gaia")

    # CUA mode (vision-based)
    env = load_environment(mode="cua", benchmark="webvoyager")
"""

from typing import Literal
import verifiers as vf
from verifiers import Parser

from .browser_env import BrowserEnv, ModeType
from .browser_datasets import load_benchmark_dataset, BenchmarkType
from .rewards import get_judge_prompt, efficiency_reward, judge_answer_reward


def _get_dom_system_prompt() -> str:
    return """You are a browser automation agent using Stagehand's AI-driven tools.

Available tools:
- navigate(url): Navigate to a URL
- observe(instruction): Find possible actions matching the instruction
- act(instruction): Execute an action described in natural language
- extract(instruction, schema_json): Extract structured data from the page

Use natural language to describe what you want to do. Stagehand will intelligently
find elements and execute actions without needing CSS selectors or coordinates.

Complete the given task efficiently."""


def _get_cua_system_prompt() -> str:
    return """You are a browser automation agent. You can control a web browser using the provided tools.

Available tools:
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

After each action, you will receive a screenshot showing the current page state.
Analyze the screenshot to determine your next action.

Complete the given task efficiently using the minimum number of actions necessary."""


def load_environment(
    # Mode selection
    mode: ModeType = "dom",
    # Shared Browserbase config
    browserbase_api_key: str | None = None,
    browserbase_project_id: str | None = None,
    # DOM mode specific
    model_api_key: str | None = None,
    stagehand_model: str = "openai/gpt-4o-mini",
    proxy_model_to_stagehand: bool = False,
    # CUA mode specific
    server_url: str = "http://localhost:3000",
    env: Literal["LOCAL", "BROWSERBASE"] = "LOCAL",
    viewport_width: int = 1024,
    viewport_height: int = 768,
    save_screenshots: bool = True,
    keep_recent_screenshots: int | None = 2,
    proxies: bool = False,
    # Benchmark & evaluation config
    benchmark: BenchmarkType = "smoke_test",
    num_examples: int = -1,
    difficulty_level: int | str = "easy",
    web_filter: str | None = None,
    max_turns: int = 20,
    system_prompt: str | None = None,
    judge_model: str = "gpt-4o-mini",
    efficiency_weight: float = 0.1,
    task_completion_weight: float = 1.0,
    **kwargs,
) -> vf.Environment:
    """
    Load the unified Browser environment.

    Args:
        mode: Browser control mode
            - "dom": Natural language operations via Stagehand SDK
            - "cua": Vision-based primitives via CUA server

        browserbase_api_key: Browserbase API key (or set BROWSERBASE_API_KEY env var)
        browserbase_project_id: Browserbase project ID (or set BROWSERBASE_PROJECT_ID)

        # DOM mode options
        model_api_key: API key for LLM calls (or set MODEL_API_KEY env var)
        stagehand_model: Model for Stagehand operations
        proxy_model_to_stagehand: If True, route Stagehand internal LLM calls
            through the evaluation model. Default False uses stagehand_model.

        # CUA mode options
        server_url: URL of the CUA server
        env: "LOCAL" or "BROWSERBASE"
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        save_screenshots: Save screenshots to disk
        keep_recent_screenshots: Number of recent screenshots to keep in context
        proxies: Enable Browserbase proxies

        # Benchmark options
        benchmark: Benchmark to use ("smoke_test", "gaia", "webvoyager", "onlineMind2Web")
        num_examples: Number of examples (-1 for all)
        difficulty_level: Difficulty level for benchmarks
        web_filter: Filter tasks by website
        max_turns: Maximum conversation turns
        system_prompt: Custom system prompt (overrides default)
        judge_model: Model for judging task completion (uses OPENAI_API_KEY env var)
        efficiency_weight: Weight for efficiency reward
        task_completion_weight: Weight for task completion reward

    Returns:
        Configured browser environment
    """
    # Get mode-specific system prompt
    if system_prompt is None:
        system_prompt = (
            _get_dom_system_prompt() if mode == "dom" else _get_cua_system_prompt()
        )

    # Load dataset
    dataset = load_benchmark_dataset(
        benchmark=benchmark,
        num_examples=num_examples,
        difficulty_level=difficulty_level,
        web_filter=web_filter,
    )

    # Create parser
    parser = Parser()

    # Create rubrics
    # Note: StatefulToolEnv automatically adds ToolMonitorRubric internally,
    # so we don't need to add it explicitly here.

    # 1. Efficiency rubric for action efficiency
    efficiency_rubric = vf.Rubric(
        funcs=[efficiency_reward],
        weights=[efficiency_weight],
        parser=parser,
    )

    # 2. LLM Judge rubric for evaluating answer/task correctness
    judge_prompt = get_judge_prompt(benchmark)
    judge_rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=judge_prompt,
        parser=parser,
    )
    judge_rubric.add_reward_func(judge_answer_reward, weight=task_completion_weight)

    # 3. Combine rubrics
    rubric = vf.RubricGroup(rubrics=[efficiency_rubric, judge_rubric])

    # Create environment
    browser_env = BrowserEnv(
        mode=mode,
        # Shared
        browserbase_api_key=browserbase_api_key,
        browserbase_project_id=browserbase_project_id,
        # DOM mode
        model_api_key=model_api_key,
        stagehand_model=stagehand_model,
        proxy_model_to_stagehand=proxy_model_to_stagehand,
        # CUA mode
        server_url=server_url,
        env=env,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        save_screenshots=save_screenshots,
        keep_recent_screenshots=keep_recent_screenshots,
        proxies=proxies,
        # Common
        max_turns=max_turns,
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        parser=parser,
        **kwargs,
    )

    return browser_env
