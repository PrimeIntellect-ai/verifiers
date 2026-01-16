"""
WebVoyager Browser Benchmark Environment.

WebVoyager contains web navigation tasks on specific websites.
Tasks are evaluated based on successful completion rather than
explicit ground-truth answers.

Usage:
    prime eval run webvoyager -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import verifiers as vf
from verifiers.envs.integrations.browser_env import BrowserEnv
from datasets import Dataset

_logger = logging.getLogger(__name__)

# Task-based judge prompt for evaluating task completion
# WebVoyager has no explicit answers, only tasks to complete
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


def load_webvoyager_dataset(
    num_examples: int = -1,
    web_filter: Optional[str] = None,
) -> Dataset:
    """
    Load WebVoyager benchmark tasks from local JSONL file.

    Args:
        num_examples: Number of examples to load. Use -1 for all (default: -1)
        web_filter: Filter tasks by website name (e.g., "Allrecipes", "Amazon")

    Returns:
        Dataset with question, answer, start_url, task_id, and website columns
    """
    # Load from local JSONL file
    dataset_path = Path(__file__).parent / "datasets" / "WebVoyager_data.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"WebVoyager dataset not found at {dataset_path}. "
            "Please ensure the dataset file exists."
        )

    # Read JSONL and optionally filter by website
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # Apply web filter if specified
            if web_filter and item.get("web_name") != web_filter:
                continue
            examples.append(
                {
                    "question": item["ques"],
                    "answer": "",  # WebVoyager is task-based, no explicit answers
                    "start_url": item["web"],
                    "task_id": item["id"],
                    "website": item["web_name"],
                }
            )

    if num_examples > 0:
        examples = examples[:num_examples]

    return Dataset.from_list(examples)


async def judge_task_completion(
    judge,
    prompt: str | list,
    completion: str | list,
    answer: str,
    state: vf.State,
) -> float:
    """
    LLM judge reward that evaluates whether the task was completed successfully.

    Args:
        judge: Callable injected by JudgeRubric that calls the judge LLM
        prompt: The original prompt/question given to the agent
        completion: The agent's full response/trajectory
        answer: Not used for task-based evaluation (empty string)
        state: The current environment state

    Returns:
        float: 1.0 if the judge determines the task was completed, 0.0 otherwise
    """
    judge_response = await judge(prompt, completion, answer, state)
    is_complete = "yes" in judge_response.lower()
    return 1.0 if is_complete else 0.0


def load_environment(
    mode: str = "dom",
    max_turns: int = 15,
    judge_model: str = "gpt-4o-mini",
    num_examples: int = -1,
    web_filter: str | None = None,
    browserbase_api_key: str | None = None,
    browserbase_project_id: str | None = None,
    stagehand_model: str = "openai/gpt-4o-mini",
    model_api_key: str | None = None,
    proxy_model_to_stagehand: bool = False,
    server_url: str = "http://localhost:3000",
    env: Literal["LOCAL", "BROWSERBASE"] = "LOCAL",
    **kwargs,
) -> vf.Environment:
    """
    Load the WebVoyager browser benchmark environment.

    WebVoyager contains web navigation tasks across multiple websites.
    Tasks are evaluated based on successful task completion.

    Args:
        mode: Browser control mode ("dom" or "cua")
            - "dom": Natural language operations via Stagehand SDK
            - "cua": Vision-based primitives via CUA server
        max_turns: Maximum conversation turns (default: 15)
        judge_model: Model for judging task completion
        num_examples: Number of examples to load (-1 for all)
        web_filter: Filter tasks by website name (e.g., "Allrecipes", "Amazon", "Apple")
        browserbase_api_key: Browserbase API key (or set BROWSERBASE_API_KEY env var)
        browserbase_project_id: Browserbase project ID (or set BROWSERBASE_PROJECT_ID env var)
        stagehand_model: Model for Stagehand operations (DOM mode only)
        model_api_key: API key for model calls (or set MODEL_API_KEY env var)
        proxy_model_to_stagehand: Route Stagehand LLM calls through evaluation model
        server_url: CUA server URL (CUA mode only)
        env: Environment type - "LOCAL" or "BROWSERBASE"
        **kwargs: Additional arguments passed to BrowserEnv

    Returns:
        Configured BrowserEnv instance for WebVoyager benchmark

    Example:
        >>> env = load_environment()  # all websites
        >>> env = load_environment(web_filter="Amazon", num_examples=10)
    """
    # Load dataset
    dataset = load_webvoyager_dataset(
        num_examples=num_examples,
        web_filter=web_filter,
    )

    # Create judge rubric for task-based evaluation
    rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=TASK_JUDGE_PROMPT,
    )
    rubric.add_reward_func(judge_task_completion, weight=1.0)

    # Create BrowserEnv (uses default system prompt for mode)
    return BrowserEnv(
        mode=mode,
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        browserbase_api_key=browserbase_api_key,
        browserbase_project_id=browserbase_project_id,
        stagehand_model=stagehand_model,
        model_api_key=model_api_key,
        proxy_model_to_stagehand=proxy_model_to_stagehand,
        server_url=server_url,
        env=env,
        **kwargs,
    )
