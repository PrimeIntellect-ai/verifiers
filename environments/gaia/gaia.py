"""
GAIA Web Browser Benchmark Environment.

GAIA tasks require web browsing and reasoning to find answers.
The benchmark contains multi-hop questions that need web navigation
to gather information and synthesize answers.

Usage:
    prime eval run gaia -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
"""

import json
import logging
from pathlib import Path
from typing import Literal, Union

import verifiers as vf
from verifiers.envs.integrations.browser_env import BrowserEnv
from datasets import Dataset

_logger = logging.getLogger(__name__)

# Difficulty level mappings
# GAIA has 2 levels: 1 (easier) and 2 (harder)
GAIA_DIFFICULTY_MAP = {
    "easy": 1,
    "hard": 2,
    1: 1,
    2: 2,
}

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


def _normalize_difficulty(difficulty_level: Union[str, int, None]) -> int:
    """Normalize difficulty level for GAIA benchmark."""
    if difficulty_level is None:
        return 1  # Default to easy/level 1

    mapped = GAIA_DIFFICULTY_MAP.get(difficulty_level)
    if mapped is None:
        valid_options = '"easy" (1), "hard" (2)'
        raise ValueError(
            f"Invalid difficulty_level '{difficulty_level}' for GAIA. "
            f"Valid options: {valid_options}"
        )
    return mapped


def load_gaia_dataset(
    num_examples: int = -1,
    difficulty_level: Union[str, int, None] = "easy",
) -> Dataset:
    """
    Load GAIA benchmark questions from local JSONL file.

    Args:
        num_examples: Number of examples to load. Use -1 for all (default: -1)
        difficulty_level: Task difficulty:
            - "easy" or 1: Level 1 tasks (easier)
            - "hard" or 2: Level 2 tasks (harder)
            Default: "easy"

    Returns:
        Dataset with question, answer, start_url, and task_id columns
    """
    # Normalize difficulty
    internal_level = _normalize_difficulty(difficulty_level)

    # Load from local JSONL file
    dataset_path = Path(__file__).parent / "datasets" / "GAIA_web.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"GAIA dataset not found at {dataset_path}. "
            "Please ensure the dataset file exists."
        )

    # Read JSONL and filter by difficulty
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if item.get("Level") == internal_level:
                examples.append(
                    {
                        "question": item["ques"],
                        "answer": item["Final answer"],
                        "start_url": item["web"],
                        "task_id": item["task_id"],
                    }
                )

    if num_examples > 0:
        examples = examples[:num_examples]

    return Dataset.from_list(examples)


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
    mode: str = "dom",
    max_turns: int = 15,
    judge_model: str = "gpt-4o-mini",
    num_examples: int = -1,
    difficulty_level: Union[str, int] = "easy",
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
    Load the GAIA web browser benchmark environment.

    GAIA tasks are multi-hop questions requiring web browsing to find answers.
    Each task has a ground-truth answer for evaluation.

    Args:
        mode: Browser control mode ("dom" or "cua")
            - "dom": Natural language operations via Stagehand SDK
            - "cua": Vision-based primitives via CUA server
        max_turns: Maximum conversation turns (default: 15)
        judge_model: Model for judging task completion
        num_examples: Number of examples to load (-1 for all)
        difficulty_level: Task difficulty:
            - "easy" or 1: Level 1 tasks (easier)
            - "hard" or 2: Level 2 tasks (harder)
        browserbase_api_key: Browserbase API key (or set BROWSERBASE_API_KEY env var)
        browserbase_project_id: Browserbase project ID (or set BROWSERBASE_PROJECT_ID env var)
        stagehand_model: Model for Stagehand operations (DOM mode only)
        model_api_key: API key for model calls (or set MODEL_API_KEY env var)
        proxy_model_to_stagehand: Route Stagehand LLM calls through evaluation model
        server_url: CUA server URL (CUA mode only)
        env: Environment type - "LOCAL" or "BROWSERBASE"
        **kwargs: Additional arguments passed to BrowserEnv

    Returns:
        Configured BrowserEnv instance for GAIA benchmark

    Example:
        >>> env = load_environment()  # easy tasks
        >>> env = load_environment(difficulty_level="hard", num_examples=10)
    """
    # Load dataset
    dataset = load_gaia_dataset(
        num_examples=num_examples,
        difficulty_level=difficulty_level,
    )

    # Create judge rubric for evaluation
    rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )
    rubric.add_reward_func(judge_answer, weight=1.0)

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
