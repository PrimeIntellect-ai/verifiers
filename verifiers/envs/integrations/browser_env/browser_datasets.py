"""
Dataset utilities for browser environment evaluation.

This module provides functions for loading and creating datasets
used in browser task evaluations.

Datasets can be loaded from Prime Environments Hub or fall back to local files.
"""

import json
import logging
from pathlib import Path
from typing import Literal, Union, Optional

from datasets import Dataset

# Prime Environments Hub integration
try:
    from prime import load_dataset as load_prime_dataset

    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False

# Mapping from benchmark names to Prime Hub dataset identifiers
HUB_DATASETS = {
    "gaia": "browserbase/gaia-web",
    "webvoyager": "browserbase/webvoyager",
    "onlineMind2Web": "browserbase/mind2web-online",
}

_logger = logging.getLogger(__name__)

BenchmarkType = Literal["smoke_test", "gaia", "webvoyager", "onlineMind2Web"]
DifficultyLevel = Union[str, int, None]

# Base path for local datasets (fallback)
_LOCAL_DATASETS_PATH = Path(__file__).parent / "datasets"

# ==================== Difficulty Level Mappings ====================
# Unified difficulty levels: "easy", "medium", "hard" (or integers 1, 2, 3)
# These map to benchmark-specific internal values.

# GAIA has 2 levels: 1 (easier) and 2 (harder)
# Mapping: "easy"/1 -> level 1, "hard"/2 -> level 2
GAIA_DIFFICULTY_MAP = {
    "easy": 1,
    "hard": 2,
    # Accept integers for backwards compatibility
    1: 1,
    2: 2,
}

# Mind2Web has 3 levels: "easy", "medium", "hard"
# Mapping: 1 -> "easy", 2 -> "medium", 3 -> "hard"
MIND2WEB_DIFFICULTY_MAP = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
    # Accept integers for convenience
    1: "easy",
    2: "medium",
    3: "hard",
}


def _normalize_gaia_difficulty(difficulty_level: DifficultyLevel) -> int:
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


def _normalize_mind2web_difficulty(difficulty_level: DifficultyLevel) -> str | None:
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


def _load_from_hub(benchmark: str, **kwargs) -> Optional[Dataset]:
    """
    Try to load dataset from Prime Environments Hub.

    Returns None if hub is unavailable or loading fails, allowing
    fallback to local files.

    Args:
        benchmark: Benchmark name (gaia, webvoyager, onlineMind2Web)
        **kwargs: Benchmark-specific filters (difficulty_level, web_filter, etc.)

    Returns:
        Dataset if successfully loaded from Hub, None otherwise
    """
    if not HUB_AVAILABLE:
        return None

    hub_name = HUB_DATASETS.get(benchmark)
    if hub_name is None:
        return None

    try:
        # Load from Prime Environments Hub
        dataset = load_prime_dataset(hub_name)

        # Apply benchmark-specific filters
        if benchmark == "gaia":
            difficulty = kwargs.get("difficulty_level", "easy")
            internal_level = _normalize_gaia_difficulty(difficulty)
            dataset = dataset.filter(lambda x: x.get("Level") == internal_level)
        elif benchmark == "webvoyager":
            web_filter = kwargs.get("web_filter")
            if web_filter:
                dataset = dataset.filter(lambda x: x.get("web_name") == web_filter)
        elif benchmark == "onlineMind2Web":
            difficulty = kwargs.get("difficulty_level")
            if difficulty:
                internal_level = _normalize_mind2web_difficulty(difficulty)
                dataset = dataset.filter(lambda x: x.get("level") == internal_level)

        # Standardize column names to match expected format
        column_mapping = {
            "ques": "question",
            "Final answer": "answer",
            "web": "start_url",
            "id": "task_id",
            "confirmed_task": "question",
            "website": "start_url",
        }
        for old_name, new_name in column_mapping.items():
            if (
                old_name in dataset.column_names
                and new_name not in dataset.column_names
            ):
                dataset = dataset.rename_column(old_name, new_name)

        return dataset

    except Exception as e:
        _logger.warning(
            f"Failed to load {benchmark} from Prime Hub ({hub_name}): {e}. "
            "Falling back to local files."
        )
        return None


__all__ = [
    "load_smoke_test_dataset",
    "load_gaia_dataset",
    "load_webvoyager_dataset",
    "load_mind2web_dataset",
    "load_benchmark_dataset",
    # Keep alias for backwards compatibility
    "load_browser_dataset",
    "BenchmarkType",
    "DifficultyLevel",
]


def load_smoke_test_dataset() -> Dataset:
    """
    Load the smoke test dataset for basic browser evaluation.

    This is a simple single-task dataset for testing browser navigation
    to the Prime Intellect homepage.

    Returns:
        Dataset: A HuggingFace Dataset with 'question', 'answer', 'start_url', and 'task_id' columns

    Example:
        >>> dataset = load_smoke_test_dataset()
        >>> print(dataset[0])
        {'question': 'what does the top green banner say...', 'answer': 'INTELLECT-3', ...}
    """
    return Dataset.from_dict(
        {
            "question": [
                "what does the headline say on the primeintellect.ai homepage?"
            ],
            "answer": ["The Open Superintelligence Stack"],
            "start_url": ["https://primeintellect.ai"],
            "task_id": ["smoke-test-0"],
        }
    )


# Backwards compatibility alias
def load_browser_dataset() -> Dataset:
    """Alias for load_smoke_test_dataset() for backwards compatibility."""
    return load_smoke_test_dataset()


def load_gaia_dataset(
    num_examples: int = -1,
    difficulty_level: DifficultyLevel = "easy",
    use_hub: bool = True,
) -> Dataset:
    """
    Load GAIA benchmark questions for browser evaluation.

    GAIA tasks require web browsing and reasoning to find answers.

    Args:
        num_examples: Number of examples to load. Use -1 to load all examples (default: -1)
        difficulty_level: Difficulty level. Accepts:
            - "easy" or 1: Level 1 tasks (26 tasks, easier)
            - "hard" or 2: Level 2 tasks (65 tasks, harder)
            Default: "easy"
        use_hub: If True, try Prime Hub first, fall back to local (default: True)

    Returns:
        Dataset: A HuggingFace Dataset with 'question', 'answer', 'start_url', and 'task_id' columns

    Example:
        >>> dataset = load_gaia_dataset(num_examples=5, difficulty_level="easy")
        >>> dataset = load_gaia_dataset(difficulty_level="hard")
        >>> dataset = load_gaia_dataset(difficulty_level=1)  # backwards compatible
    """
    # Try hub first if enabled
    if use_hub:
        hub_dataset = _load_from_hub("gaia", difficulty_level=difficulty_level)
        if hub_dataset is not None:
            if num_examples > 0:
                hub_dataset = hub_dataset.select(
                    range(min(num_examples, len(hub_dataset)))
                )
            return hub_dataset

    # Fall back to local file
    internal_level = _normalize_gaia_difficulty(difficulty_level)
    gaia_path = _LOCAL_DATASETS_PATH / "gaia" / "GAIA_web.jsonl"

    questions = []
    answers = []
    start_urls = []
    task_ids = []

    with open(gaia_path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("Level") == internal_level:
                    questions.append(row["ques"])
                    answers.append(row["Final answer"])
                    start_urls.append(row.get("web", "https://www.google.com/"))
                    task_ids.append(row.get("id", row.get("task_id", "")))
                    if num_examples > 0 and len(questions) >= num_examples:
                        break

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "start_url": start_urls,
            "task_id": task_ids,
        }
    )


def load_webvoyager_dataset(
    num_examples: int = -1,
    web_filter: str | None = None,
    use_hub: bool = True,
) -> Dataset:
    """
    Load WebVoyager benchmark tasks for browser evaluation.

    WebVoyager contains web navigation tasks on specific websites.
    These are task-based evaluations without explicit ground-truth answers.

    Args:
        num_examples: Number of examples to load. Use -1 to load all (default: -1)
        web_filter: Optional filter for specific website name (e.g., "Allrecipes", "Amazon")
        use_hub: If True, try Prime Hub first, fall back to local (default: True)

    Returns:
        Dataset: A HuggingFace Dataset with 'question', 'answer', 'start_url',
                 'task_id', and 'website' columns

    Example:
        >>> dataset = load_webvoyager_dataset(num_examples=5, web_filter="Allrecipes")
        >>> print(dataset[0])
        {'question': 'Provide a recipe for...', 'website': 'Allrecipes', ...}
    """
    # Try hub first if enabled
    if use_hub:
        hub_dataset = _load_from_hub("webvoyager", web_filter=web_filter)
        if hub_dataset is not None:
            if num_examples > 0:
                hub_dataset = hub_dataset.select(
                    range(min(num_examples, len(hub_dataset)))
                )
            return hub_dataset

    # Fall back to local file
    webvoyager_path = _LOCAL_DATASETS_PATH / "webvoyager" / "WebVoyager_data.jsonl"

    questions = []
    answers = []
    start_urls = []
    task_ids = []
    websites = []

    with open(webvoyager_path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)

                # Optional website filter
                if web_filter and row.get("web_name") != web_filter:
                    continue

                questions.append(row["ques"])
                # Task-based eval: no explicit answer, use empty string
                answers.append("")
                start_urls.append(row["web"])
                task_ids.append(row["id"])
                websites.append(row["web_name"])

                if num_examples > 0 and len(questions) >= num_examples:
                    break

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "start_url": start_urls,
            "task_id": task_ids,
            "website": websites,
        }
    )


def load_mind2web_dataset(
    num_examples: int = -1,
    difficulty_level: DifficultyLevel = None,
    use_hub: bool = True,
) -> Dataset:
    """
    Load onlineMind2Web benchmark tasks for browser evaluation.

    Mind2Web contains web navigation tasks with difficulty levels.
    These are task-based evaluations without explicit ground-truth answers.

    Args:
        num_examples: Number of examples to load. Use -1 to load all (default: -1)
        difficulty_level: Filter by difficulty. Accepts:
            - "easy" or 1: Easy tasks (83 tasks)
            - "medium" or 2: Medium tasks (143 tasks)
            - "hard" or 3: Hard tasks (74 tasks)
            - None: All tasks (301 tasks)
            Default: None (all)
        use_hub: If True, try Prime Hub first, fall back to local (default: True)

    Returns:
        Dataset: A HuggingFace Dataset with 'question', 'answer', 'start_url',
                 'task_id', and 'difficulty' columns

    Example:
        >>> dataset = load_mind2web_dataset(num_examples=5, difficulty_level="easy")
        >>> dataset = load_mind2web_dataset(difficulty_level=2)  # medium tasks
    """
    # Try hub first if enabled
    if use_hub:
        hub_dataset = _load_from_hub(
            "onlineMind2Web", difficulty_level=difficulty_level
        )
        if hub_dataset is not None:
            if num_examples > 0:
                hub_dataset = hub_dataset.select(
                    range(min(num_examples, len(hub_dataset)))
                )
            return hub_dataset

    # Fall back to local file
    internal_level = _normalize_mind2web_difficulty(difficulty_level)
    mind2web_path = _LOCAL_DATASETS_PATH / "onlineMind2Web" / "onlineMind2Web.jsonl"

    questions = []
    answers = []
    start_urls = []
    task_ids = []
    difficulties = []

    with open(mind2web_path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)

                # Optional difficulty filter
                if internal_level and row.get("level") != internal_level:
                    continue

                questions.append(row["confirmed_task"])
                # Task-based eval: no explicit answer, use empty string
                answers.append("")
                start_urls.append(row["website"])
                task_ids.append(row["task_id"])
                difficulties.append(row["level"])

                if num_examples > 0 and len(questions) >= num_examples:
                    break

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "start_url": start_urls,
            "task_id": task_ids,
            "difficulty": difficulties,
        }
    )


def load_benchmark_dataset(
    benchmark: BenchmarkType,
    num_examples: int = -1,
    **kwargs,
) -> Dataset:
    """
    Unified loader for all browser benchmarks.

    Args:
        benchmark: One of "smoke_test", "gaia", "webvoyager", "onlineMind2Web"
        num_examples: Number of examples to load. Use -1 to load all (default: -1)
        **kwargs: Benchmark-specific arguments:
            - difficulty_level: Unified difficulty levels (works for gaia and onlineMind2Web):
                - "easy" or 1: Easiest tasks
                - "medium" or 2: Medium tasks (Mind2Web only)
                - "hard" or 3: Hardest tasks
                Note: GAIA only has "easy" and "hard" levels.
            - web_filter: For webvoyager only (str, e.g., "Allrecipes", "Amazon")
            - use_hub: Try Prime Hub first (default: True)

    Returns:
        Dataset: A HuggingFace Dataset with columns appropriate for the benchmark

    Example:
        >>> dataset = load_benchmark_dataset("gaia", num_examples=5, difficulty_level="easy")
        >>> dataset = load_benchmark_dataset("gaia", difficulty_level="hard")
        >>> dataset = load_benchmark_dataset("webvoyager", web_filter="Amazon")
        >>> dataset = load_benchmark_dataset("onlineMind2Web", difficulty_level="medium")
        >>> dataset = load_benchmark_dataset("onlineMind2Web", difficulty_level=3)  # hard
    """
    if benchmark == "smoke_test":
        return load_smoke_test_dataset()
    elif benchmark == "gaia":
        difficulty = kwargs.get("difficulty_level", "easy")
        use_hub = kwargs.get("use_hub", True)
        return load_gaia_dataset(
            num_examples=num_examples, difficulty_level=difficulty, use_hub=use_hub
        )
    elif benchmark == "webvoyager":
        web_filter = kwargs.get("web_filter")
        use_hub = kwargs.get("use_hub", True)
        return load_webvoyager_dataset(
            num_examples=num_examples, web_filter=web_filter, use_hub=use_hub
        )
    elif benchmark == "onlineMind2Web":
        difficulty = kwargs.get("difficulty_level")
        use_hub = kwargs.get("use_hub", True)
        return load_mind2web_dataset(
            num_examples=num_examples, difficulty_level=difficulty, use_hub=use_hub
        )
    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. Must be one of: smoke_test, gaia, webvoyager, onlineMind2Web"
        )
