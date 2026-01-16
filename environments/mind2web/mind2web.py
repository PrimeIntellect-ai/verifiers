"""
onlineMind2Web Browser Benchmark Environment.

Mind2Web contains web navigation tasks with varying difficulty levels.
Tasks are evaluated based on successful completion rather than
explicit ground-truth answers.

Usage:
    prime eval run mind2web -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional, Union

import verifiers as vf
from verifiers.envs.integrations.browser_env import BrowserEnv
from datasets import Dataset

_logger = logging.getLogger(__name__)

# Difficulty level mappings
# Mind2Web has 3 levels: "easy", "medium", "hard"
MIND2WEB_DIFFICULTY_MAP = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
    1: "easy",
    2: "medium",
    3: "hard",
}

# Task-based judge prompt for evaluating task completion
# Mind2Web has no explicit answers, only tasks to complete
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


def _normalize_difficulty(difficulty_level: Union[str, int, None]) -> Optional[str]:
    """Normalize difficulty level for Mind2Web benchmark."""
    if difficulty_level is None:
        return None  # No filter, return all

    mapped = MIND2WEB_DIFFICULTY_MAP.get(difficulty_level)
    if mapped is None:
        valid_options = '"easy" (1), "medium" (2), "hard" (3)'
        raise ValueError(
            f"Invalid difficulty_level '{difficulty_level}' for Mind2Web. "
            f"Valid options: {valid_options}"
        )
    return mapped


def load_mind2web_dataset(
    num_examples: int = -1,
    difficulty_level: Union[str, int, None] = None,
) -> Dataset:
    """
    Load Mind2Web benchmark tasks from local JSONL file.

    Args:
        num_examples: Number of examples to load. Use -1 for all (default: -1)
        difficulty_level: Filter by difficulty:
            - "easy" or 1: Easy tasks
            - "medium" or 2: Medium tasks
            - "hard" or 3: Hard tasks
            - None: All tasks (default)

    Returns:
        Dataset with question, answer, start_url, task_id, and difficulty columns
    """
    # Normalize difficulty
    internal_level = _normalize_difficulty(difficulty_level)

    # Load from local JSONL file
    dataset_path = Path(__file__).parent / "datasets" / "onlineMind2Web.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Mind2Web dataset not found at {dataset_path}. "
            "Please ensure the dataset file exists."
        )

    # Read JSONL and optionally filter by difficulty
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # Apply difficulty filter if specified
            if internal_level and item.get("level") != internal_level:
                continue
            examples.append(
                {
                    "question": item["confirmed_task"],
                    "answer": "",  # Mind2Web is task-based, no explicit answers
                    "start_url": item["website"],
                    "task_id": item["task_id"],
                    "difficulty": item["level"],
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
    difficulty_level: Union[str, int, None] = None,
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
    Load the onlineMind2Web browser benchmark environment.

    Mind2Web contains web navigation tasks with difficulty levels.
    Tasks are evaluated based on successful task completion.

    Args:
        mode: Browser control mode ("dom" or "cua")
            - "dom": Natural language operations via Stagehand SDK
            - "cua": Vision-based primitives via CUA server
        max_turns: Maximum conversation turns (default: 15)
        judge_model: Model for judging task completion
        num_examples: Number of examples to load (-1 for all)
        difficulty_level: Filter by difficulty:
            - "easy" or 1: Easy tasks
            - "medium" or 2: Medium tasks
            - "hard" or 3: Hard tasks
            - None: All tasks (default)
        browserbase_api_key: Browserbase API key (or set BROWSERBASE_API_KEY env var)
        browserbase_project_id: Browserbase project ID (or set BROWSERBASE_PROJECT_ID env var)
        stagehand_model: Model for Stagehand operations (DOM mode only)
        model_api_key: API key for model calls (or set MODEL_API_KEY env var)
        proxy_model_to_stagehand: Route Stagehand LLM calls through evaluation model
        server_url: CUA server URL (CUA mode only)
        env: Environment type - "LOCAL" or "BROWSERBASE"
        **kwargs: Additional arguments passed to BrowserEnv

    Returns:
        Configured BrowserEnv instance for Mind2Web benchmark

    Example:
        >>> env = load_environment()  # all difficulties
        >>> env = load_environment(difficulty_level="easy", num_examples=10)
    """
    # Load dataset
    dataset = load_mind2web_dataset(
        num_examples=num_examples,
        difficulty_level=difficulty_level,
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
