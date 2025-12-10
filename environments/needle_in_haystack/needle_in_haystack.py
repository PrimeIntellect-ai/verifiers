"""
Needle in Haystack Environment.

Tests a model's ability to find a specific piece of information (a "magic number")
hidden within a large body of random text.

Supports two modes:
- RLM mode (use_rlm=True): Uses RLMEnv with context in info["context"], model
  explores using Python code in the REPL
- Standard mode (use_rlm=False): Uses SingleTurnEnv with context directly in the
  prompt, model must search through the text directly

The model must find the line containing "The magic number is X" and return the number.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from filelock import FileLock

import verifiers as vf
from verifiers import RLMEnv
from verifiers import SingleTurnEnv
from verifiers.utils.data_utils import extract_boxed_answer

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics Logging
# =============================================================================


class MetricsLogger:
    """Thread-safe JSON metrics logger for needle-in-haystack experiments.

    Writes metrics incrementally to a JSON file in columnar format (dict of lists)
    for easy loading into pandas. Uses file locking for safe concurrent access.

    Usage:
        logger = MetricsLogger("results/metrics.json")
        logger.log(example_id=1, exact_match=1.0, num_lines=1024, ...)
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.output_path.with_suffix(".lock")
        self._lock = FileLock(self.lock_path)

        # Initialize file with empty structure if it doesn't exist
        if not self.output_path.exists():
            self._write_metrics(self._empty_metrics())

    def _empty_metrics(self) -> dict[str, list]:
        """Return empty metrics structure with all tracked fields."""
        return {
            # Identifiers
            "example_id": [],
            # Experiment parameters
            "num_lines": [],
            "needle_position": [],
            "needle_variance": [],
            "use_rlm": [],
            "seed": [],
            # Performance
            "exact_match": [],
            "contains_answer": [],
            "final_answer": [],
            # Main branch metrics
            "main_turns": [],
            "main_tool_calls": [],
            "main_prompt_tokens": [],
            "main_completion_tokens": [],
            # Sub-LLM metrics (RLM only - will be 0 for standard mode)
            "sub_llm_calls": [],
            "sub_llm_total_tool_calls": [],
            "sub_llm_prompt_tokens": [],
            "sub_llm_completion_tokens": [],
            "sub_llm_total_turns": [],
            # Totals
            "total_prompt_tokens": [],
            "total_completion_tokens": [],
            "total_tokens": [],
            "total_tool_calls": [],
            # Timing
            "generation_ms": [],
            "scoring_ms": [],
            "total_ms": [],
            # Error tracking
            "had_error": [],
            "error_message": [],
        }

    def _read_metrics(self) -> dict[str, list]:
        """Read current metrics from file."""
        with open(self.output_path, "r") as f:
            return json.load(f)

    def _write_metrics(self, metrics: dict[str, list]):
        """Write metrics to file."""
        with open(self.output_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def log(self, **kwargs: Any):
        """Thread-safe logging of a single rollout's metrics."""
        with self._lock:
            metrics = self._read_metrics()
            for key, value in kwargs.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
            self._write_metrics(metrics)


def _create_logging_reward_func(
    metrics_logger: MetricsLogger,
    is_rlm_mode: bool,
    num_lines: int,
    needle_position: float | None,
    needle_variance: float,
    seed: int | None,
):
    """Create a 0-weight reward function that logs metrics.

    Args:
        metrics_logger: MetricsLogger instance to write metrics to.
        is_rlm_mode: Whether this is running in RLM mode.
        num_lines: Number of lines in the haystack (experiment parameter).
        needle_position: Needle position setting (experiment parameter).
        needle_variance: Needle variance setting (experiment parameter).
        seed: Random seed used (experiment parameter).

    Returns:
        A reward function that logs metrics and returns 0.0 (no effect on reward).
    """

    def logging_reward(state: dict, **kwargs) -> float:
        try:
            # Extract main branch metrics from trajectory
            main_tool_calls = 0
            main_prompt_tokens = 0
            main_completion_tokens = 0

            for step in state.get("trajectory", []):
                response = step.get("response")
                if response:
                    # Count tool calls
                    if hasattr(response, "choices"):
                        for choice in response.choices:
                            msg = getattr(choice, "message", None)
                            if msg and getattr(msg, "tool_calls", None):
                                main_tool_calls += len(msg.tool_calls)
                    # Count tokens
                    usage = getattr(response, "usage", None)
                    if usage:
                        main_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                        main_completion_tokens += (
                            getattr(usage, "completion_tokens", 0) or 0
                        )

            # Get sub-LLM metrics (will be 0 for non-RLM mode)
            sub_llm_calls = state.get("sub_llm_call_count", 0)
            sub_llm_total_tool_calls = state.get("sub_llm_total_tool_calls", 0)
            sub_llm_prompt_tokens = state.get("sub_llm_prompt_tokens", 0)
            sub_llm_completion_tokens = state.get("sub_llm_completion_tokens", 0)
            sub_llm_total_turns = state.get("sub_llm_total_turns", 0)

            # Calculate totals
            total_prompt_tokens = main_prompt_tokens + sub_llm_prompt_tokens
            total_completion_tokens = main_completion_tokens + sub_llm_completion_tokens
            total_tokens = total_prompt_tokens + total_completion_tokens
            total_tool_calls = main_tool_calls + sub_llm_total_tool_calls

            # Get timing info
            timing = state.get("timing", {})

            # Get final answer
            if is_rlm_mode:
                final_answer = state.get("final_answer", "")
            else:
                completion = state.get("completion", [])
                if completion and isinstance(completion, list):
                    final_answer = completion[-1].get("content", "")
                else:
                    final_answer = str(completion) if completion else ""

            # Calculate exact_match and contains_answer
            expected = state.get("answer", "").strip()
            if is_rlm_mode:
                extracted = state.get("final_answer", "").strip()
            else:
                extracted = _extract_number(final_answer)

            exact_match = 1.0 if extracted == expected else 0.0
            contains_answer = 1.0 if expected in final_answer else 0.0

            # Log all metrics
            metrics_logger.log(
                example_id=state.get("example_id", -1),
                # Experiment parameters
                num_lines=num_lines,
                needle_position=needle_position,
                needle_variance=needle_variance,
                use_rlm=is_rlm_mode,
                seed=seed,
                # Performance
                exact_match=exact_match,
                contains_answer=contains_answer,
                final_answer=str(final_answer)[:500],  # Truncate for storage
                # Main branch
                main_turns=len(state.get("trajectory", [])),
                main_tool_calls=main_tool_calls,
                main_prompt_tokens=main_prompt_tokens,
                main_completion_tokens=main_completion_tokens,
                # Sub-LLM (RLM only)
                sub_llm_calls=sub_llm_calls,
                sub_llm_total_tool_calls=sub_llm_total_tool_calls,
                sub_llm_prompt_tokens=sub_llm_prompt_tokens,
                sub_llm_completion_tokens=sub_llm_completion_tokens,
                sub_llm_total_turns=sub_llm_total_turns,
                # Totals
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                total_tool_calls=total_tool_calls,
                # Timing
                generation_ms=timing.get("generation_ms", 0),
                scoring_ms=timing.get("scoring_ms", 0),
                total_ms=timing.get("total_ms", 0),
                # Success
                had_error=False,
                error_message="",
            )
            return 0.0  # No effect on reward

        except Exception as e:
            # Log the error but don't crash the evaluation
            try:
                metrics_logger.log(
                    example_id=state.get("example_id", -1),
                    num_lines=num_lines,
                    needle_position=needle_position,
                    needle_variance=needle_variance,
                    use_rlm=is_rlm_mode,
                    seed=seed,
                    exact_match=-1,
                    contains_answer=-1,
                    final_answer="",
                    main_turns=0,
                    main_tool_calls=0,
                    main_prompt_tokens=0,
                    main_completion_tokens=0,
                    sub_llm_calls=0,
                    sub_llm_total_tool_calls=0,
                    sub_llm_prompt_tokens=0,
                    sub_llm_completion_tokens=0,
                    sub_llm_total_turns=0,
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                    total_tokens=0,
                    total_tool_calls=0,
                    generation_ms=0,
                    scoring_ms=0,
                    total_ms=0,
                    had_error=True,
                    error_message=str(e)[:200],
                )
            except Exception:
                pass  # Don't crash if even error logging fails
            return 0.0

    return logging_reward


# =============================================================================
# Haystack Generation
# =============================================================================


def generate_haystack(
    num_lines: int,
    answer: int,
    needle_position: float | None = None,
    needle_variance: float = 0.0,
) -> str:
    """
    Generate a large text context with random lines and a hidden magic number.

    Args:
        num_lines: Total number of lines to generate
        answer: The magic number to hide in the text
        needle_position: Position to place the needle as fraction of total lines (0.0-1.0).
                         If None, places randomly anywhere in the context.
        needle_variance: Variance around needle_position in fraction of total lines.
                         The actual position will be uniformly sampled from
                         [needle_position - needle_variance, needle_position + needle_variance].

    Returns:
        A string with num_lines lines, one of which contains the magic number
    """
    random_words = [
        "blah",
        "random",
        "text",
        "data",
        "content",
        "information",
        "sample",
        "filler",
        "noise",
        "stuff",
    ]

    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))

    # Calculate needle position
    if needle_position is None:
        # Fully random placement
        magic_position = random.randint(0, num_lines - 1)
    else:
        # Calculate position with variance
        pos_min = needle_position - needle_variance
        pos_max = needle_position + needle_variance

        # Clamp to valid range with warning
        if pos_min < 0.0 or pos_max > 1.0:
            original_min, original_max = pos_min, pos_max
            pos_min = max(0.0, pos_min)
            pos_max = min(1.0, pos_max)
            logger.warning(
                f"Needle position range [{original_min:.2f}, {original_max:.2f}] "
                f"exceeds valid bounds [0.0, 1.0], clamping to [{pos_min:.2f}, {pos_max:.2f}]"
            )

        # Sample position within range
        if pos_min == pos_max:
            position_frac = pos_min
        else:
            position_frac = random.uniform(pos_min, pos_max)

        magic_position = int(position_frac * (num_lines - 1))
        magic_position = max(0, min(num_lines - 1, magic_position))

    lines[magic_position] = f"The magic number is {answer}"

    return "\n".join(lines)


def _extract_number(text: str) -> str:
    """Extract a number from text, trying boxed format first, then raw numbers."""
    # Try boxed answer first
    boxed = extract_boxed_answer(text)
    if boxed != text:  # extract_boxed_answer returns original if no boxed found
        return boxed.strip()
    
    # Fall back to finding any 7-digit number in the response
    numbers = re.findall(r"\b\d{7}\b", text)
    if numbers:
        return numbers[-1]  # Return last 7-digit number found
    
    return text.strip()


def load_environment(
    num_samples: int = 10,
    num_lines: int = 10_000,
    needle_position: float | None = None,
    needle_variance: float = 0.0,
    use_rlm: bool = True,
    seed: int | None = None,
    metrics_output_path: str | None = None,
    **kwargs,
) -> vf.Environment:
    """
    Load the needle-in-haystack environment.

    Args:
        num_samples: Number of samples to generate for the dataset
        num_lines: Number of lines in each haystack context
        needle_position: Position to place the needle as fraction of context (0.0-1.0).
                         0.0 = beginning, 0.5 = middle, 1.0 = end.
                         If None (default), places randomly anywhere in the context.
        needle_variance: Variance around needle_position in fraction of context length.
                         The actual position will be uniformly sampled from
                         [needle_position - needle_variance, needle_position + needle_variance].
                         Ignored if needle_position is None.
        use_rlm: If True, use RLMEnv with context in info["context"].
                 If False, use SingleTurnEnv with context in the prompt.
        seed: Random seed for reproducible dataset generation. If None, uses random state.
        metrics_output_path: Optional path to JSON file for logging per-rollout metrics.
                             Metrics are appended, allowing multiple runs to share one file.
                             Useful for ablation studies and generating heatmaps.
        **kwargs: Additional arguments passed to the environment (e.g., interception_host for RLM)

    Returns:
        Configured environment instance (RLMEnv or SingleTurnEnv)
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    # Generate dataset with random answers
    dataset_rows = []
    for i in range(num_samples):
        # Generate a random 7-digit answer for each sample
        answer = random.randint(1_000_000, 9_999_999)
        context = generate_haystack(
            num_lines=num_lines,
            answer=answer,
            needle_position=needle_position,
            needle_variance=needle_variance,
        )

        if use_rlm:
            # RLM mode: context goes in info, short prompt
            dataset_rows.append(
                {
                    "example_id": i,
                    "prompt": [
                        {
                            "role": "user",
                            "content": "I'm looking for a magic number hidden in the context. "
                            "Find it and return just the number.",
                        }
                    ],
                    "task": "needle-in-haystack",
                    "answer": str(answer),
                    "info": {
                        "context": context,
                    },
                }
            )
        else:
            # Standard mode: context goes directly in the prompt
            dataset_rows.append(
                {
                    "example_id": i,
                    "prompt": [
                        {
                            "role": "user",
                            "content": f"I'm looking for a magic number hidden in the following text. "
                            "Find it and return just the number inside \\boxed{}."
                            f"\n\n<text>\n{context}\n</text>",
                        }
                    ],
                    "task": "needle-in-haystack",
                    "answer": str(answer),
                    "info": {},  # Empty info dict for standard mode
                }
            )

    dataset = Dataset.from_list(dataset_rows)

    # Setup metrics logging if path is provided
    metrics_logger = None
    if metrics_output_path:
        metrics_logger = MetricsLogger(metrics_output_path)
        logging_reward = _create_logging_reward_func(
            metrics_logger=metrics_logger,
            is_rlm_mode=use_rlm,
            num_lines=num_lines,
            needle_position=needle_position,
            needle_variance=needle_variance,
            seed=seed,
        )

    if use_rlm:
        # RLM mode: reward functions use state["final_answer"] set by RLMEnv.post_rollout()
        def exact_match_reward(state: vf.State) -> float:
            """Reward for exact match with expected answer."""
            final_answer = state.get("final_answer", "").strip()
            expected = state.get("answer", "").strip()
            return 1.0 if final_answer == expected else 0.0

        def contains_answer_reward(state: vf.State) -> float:
            """Reward if the final answer contains the expected number."""
            final_answer = state.get("final_answer", "").strip()
            expected = state.get("answer", "").strip()
            return 1.0 if expected in final_answer else 0.0

        reward_funcs = [exact_match_reward, contains_answer_reward]
        weights = [1.0, 0.0]

        # Add logging reward if metrics logging is enabled
        if metrics_logger:
            reward_funcs.append(logging_reward)
            weights.append(0.0)

        rubric = vf.Rubric(funcs=reward_funcs, weights=weights)

        env = RLMEnv(
            max_iterations=30,
            max_output_length=8192,
            context_key="context",
            dataset=dataset,
            rubric=rubric,
            interception_host=kwargs.get("interception_host"),
        )
    else:
        # Standard mode: reward functions extract answer from completion
        # Note: completion is a list of messages, not a string
        def exact_match_reward(completion: list, answer: str) -> float:
            """Reward for exact match with expected answer."""
            # Extract content from last assistant message
            if completion and isinstance(completion, list):
                content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
            else:
                content = str(completion) if completion else ""
            extracted = _extract_number(content)
            return 1.0 if extracted == answer.strip() else 0.0

        def contains_answer_reward(completion: list, answer: str) -> float:
            """Reward if the completion contains the expected number."""
            # Extract content from last assistant message
            if completion and isinstance(completion, list):
                content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
            else:
                content = str(completion) if completion else ""
            return 1.0 if answer.strip() in content else 0.0

        reward_funcs = [exact_match_reward, contains_answer_reward]
        weights = [1.0, 0.0]

        # Add logging reward if metrics logging is enabled
        if metrics_logger:
            reward_funcs.append(logging_reward)
            weights.append(0.0)

        rubric = vf.Rubric(funcs=reward_funcs, weights=weights)

        env = SingleTurnEnv(
            dataset=dataset,
            rubric=rubric,
        )

    return env
