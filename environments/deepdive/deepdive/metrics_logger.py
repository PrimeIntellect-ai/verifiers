"""
Metrics logger for RLM vs non-RLM comparison experiments.

Provides thread-safe JSON logging of per-rollout metrics for statistical analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filelock import FileLock


class MetricsLogger:
    """Thread-safe JSON metrics logger for RLM experiments.

    Writes metrics incrementally to a JSON file, with one entry per rollout.
    Uses file locking for safe concurrent access from multiple async tasks.

    Usage:
        logger = MetricsLogger("results/metrics.json")
        logger.log(example_id=1, total_tokens=1234, ...)
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
            "prompt_preview": [],  # First 100 chars of prompt for debugging
            # Performance
            "judge_correct": [],  # From the judge rubric (if available)
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
            # Mode indicator
            "is_rlm_mode": [],
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
        """Thread-safe logging of a single rollout's metrics.

        Args:
            **kwargs: Metric name-value pairs to log.
        """
        with self._lock:
            metrics = self._read_metrics()
            for key, value in kwargs.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
            self._write_metrics(metrics)

    def log_batch(self, records: list[dict[str, Any]]):
        """Thread-safe batch logging of multiple rollouts.

        Args:
            records: List of dicts, each containing metrics for one rollout.
        """
        with self._lock:
            metrics = self._read_metrics()
            for record in records:
                for key, value in record.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            self._write_metrics(metrics)


def create_logging_reward_func(logger: MetricsLogger, is_rlm_mode: bool = False):
    """Create a reward function that logs metrics and returns 1 (success) or 0 (error).

    This function should be added to the rubric with weight=0.0 so it only logs
    metrics without affecting the reward.

    Args:
        logger: MetricsLogger instance to write metrics to.
        is_rlm_mode: Whether this is running in RLM mode (affects which metrics are relevant).

    Returns:
        A reward function that logs metrics and returns 1.0 on success, 0.0 on error.
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

            # Get prompt preview for debugging
            prompt = state.get("prompt", [])
            if isinstance(prompt, list) and len(prompt) > 0:
                # Get the user message content
                for msg in prompt:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        prompt_preview = (
                            content[:100]
                            if isinstance(content, str)
                            else str(content)[:100]
                        )
                        break
                else:
                    prompt_preview = str(prompt[0].get("content", ""))[:100]
            else:
                prompt_preview = str(prompt)[:100]

            # Get final answer
            if is_rlm_mode:
                final_answer = state.get("final_answer", "")
            else:
                final_answer = state.get(
                    "[[deepdive/FINAL_ANSWER]]",
                    state.get("completion", [{}])[-1].get("content", "")
                    if state.get("completion")
                    else "",
                )

            # Log all metrics
            logger.log(
                example_id=state.get("example_id", -1),
                prompt_preview=prompt_preview,
                # Performance - read from state (set by judge_reward_func)
                judge_correct=state.get("judge_reward", -1),
                final_answer=str(final_answer)[:500],  # Truncate for storage
                # Main branch
                main_turns=len(state.get("trajectory", [])),
                main_tool_calls=main_tool_calls,
                main_prompt_tokens=main_prompt_tokens,
                main_completion_tokens=main_completion_tokens,
                # Sub-LLM (RLM only - will be 0 for standard mode)
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
                # Mode
                is_rlm_mode=is_rlm_mode,
            )
            return 1.0  # Success

        except Exception as e:
            # Log the error but don't crash the evaluation
            try:
                logger.log(
                    example_id=state.get("example_id", -1),
                    prompt_preview="",
                    judge_correct=state.get("judge_reward", -1),
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
                    is_rlm_mode=is_rlm_mode,
                )
            except Exception:
                pass  # Don't crash if even error logging fails
            return 0.0  # Error indicator

    return logging_reward
