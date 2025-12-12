"""
Oolong Long-Context Evaluation Environment.

Implements the Oolong benchmark for evaluating long-context understanding
capabilities of language models.

Supports two modes:
- RLM mode (use_rlm=True): Uses RLMEnv with context in info["context"], model
  explores using Python code in the REPL
- Standard mode (use_rlm=False): Uses SingleTurnEnv with context directly in the
  prompt, model must process the text directly

The Oolong benchmark consists of two datasets:
- oolong-synth: Synthetic long-context evaluation tasks
- oolong-real: Real-world long-context evaluation tasks
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset
from filelock import FileLock
from openai import AsyncOpenAI

import verifiers as vf
from verifiers import RLMEnv, SingleTurnEnv
from verifiers.utils.data_utils import extract_boxed_answer


# =============================================================================
# Metrics Logging
# =============================================================================


class MetricsLogger:
    """Thread-safe JSON metrics logger for oolong experiments.

    Writes metrics incrementally to a JSON file in columnar format (dict of lists)
    for easy loading into pandas. Uses file locking for safe concurrent access.

    Usage:
        logger = MetricsLogger("results/metrics.json")
        logger.log(example_id=1, judge_correct=1.0, total_tokens=1234, ...)
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
            "subset": [],
            "prompt_preview": [],  # First 100 chars of question for debugging
            # Performance
            "judge_correct": [],
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
            # Mode and error tracking
            "is_rlm_mode": [],
            "had_error": [],
            "error_message": [],
        }

    def _read_metrics(self) -> dict[str, list]:
        """Read current metrics from file."""
        with open(self.output_path) as f:
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
    subset: str,
):
    """Create a 0-weight reward function that logs metrics.

    Args:
        metrics_logger: MetricsLogger instance to write metrics to.
        is_rlm_mode: Whether this is running in RLM mode.
        subset: Which oolong subset is being used.

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

            # Get prompt preview for debugging
            prompt = state.get("prompt", [])
            if isinstance(prompt, list) and len(prompt) > 0:
                for msg in prompt:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        # For standard mode, extract just the question part
                        if "<context>" in content:
                            content = content.split("<context>")[0].strip()
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
                completion = state.get("completion", [])
                if completion and isinstance(completion, list):
                    content = (
                        completion[-1].get("content", "")
                        if isinstance(completion[-1], dict)
                        else str(completion[-1])
                    )
                    final_answer = extract_boxed_answer(content)
                else:
                    final_answer = str(completion) if completion else ""

            # Check for rollout errors (set by MultiTurnEnv on unrecoverable errors)
            rollout_error = state.get("error")
            had_error = rollout_error is not None
            error_message = str(rollout_error)[:200] if rollout_error else ""

            # Get context length for analysis by context size
            context_length = state.get("info", {}).get("context_length", 0)

            # Log all metrics
            metrics_logger.log(
                example_id=state.get("example_id", -1),
                subset=subset,
                prompt_preview=prompt_preview,
                context_length=context_length,
                # Performance - read from state (set by reward functions)
                judge_correct=state.get("_judge_reward", -1),
                exact_match=state.get("_exact_match", -1),
                contains_answer=state.get("_contains_answer", -1),
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
                # Mode and error tracking
                is_rlm_mode=is_rlm_mode,
                had_error=had_error,
                error_message=error_message,
            )
            return 0.0  # No effect on reward

        except Exception as e:
            # Log the error but don't crash the evaluation
            try:
                metrics_logger.log(
                    example_id=state.get("example_id", -1),
                    subset=subset,
                    prompt_preview="",
                    context_length=state.get("info", {}).get("context_length", 0),
                    judge_correct=-1,
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
                    is_rlm_mode=is_rlm_mode,
                    had_error=True,
                    error_message=str(e)[:200],
                )
            except Exception:
                pass  # Don't crash if even error logging fails
            return 0.0

    return logging_reward


# =============================================================================
# Environment Loading
# =============================================================================


def load_environment(
    subset: Literal["synth", "synth_with_labels", "real"] = "synth",
    split: Literal["validation", "test"] = "validation",
    shuffle: bool = False,
    seed: int = 42,
    use_rlm: bool = True,
    max_iterations: int = 30,
    max_output_length: int = 8192,
    judge_model: str = "gpt-5-mini",
    judge_api_key_var: str = "OPENAI_API_KEY",
    metrics_output_path: str | None = None,
    **kwargs,
) -> vf.Environment:
    """
    Load the Oolong long-context evaluation environment.

    Args:
        subset: Which subset to use:
            - "synth": Synthetic dataset with context_window_text
            - "synth_with_labels": Synthetic dataset with context_window_text_with_labels
            - "real": Real-world dataset with context_window_text
        split: Dataset split to use ("validation" or "test")
        shuffle: Whether to shuffle the dataset (useful for sampling different examples)
        use_rlm: If True, use RLMEnv with context in info["context"].
                 If False, use SingleTurnEnv with context directly in the prompt.
        max_iterations: Maximum REPL iterations before stopping (RLM mode only)
        max_output_length: Maximum length of code execution output (RLM mode only)
        judge_model: Model to use for judging answer correctness
        judge_api_key_var: Environment variable containing the API key for the judge model
        metrics_output_path: Optional path to JSON file for logging per-rollout metrics.
                             Metrics are appended, allowing multiple runs to share one file.
        **kwargs: Additional arguments passed to the environment (e.g., interception_host for RLM)

    Returns:
        Configured environment instance (RLMEnv or SingleTurnEnv)
    """
    # Determine HuggingFace dataset name and context column based on subset
    if subset in ("synth", "synth_with_labels"):
        hf_dataset_name = "oolongbench/oolong-synth"
        context_column = (
            "context_window_text"
            if subset == "synth"
            else "context_window_text_with_labels"
        )
    else:  # "real"
        hf_dataset_name = "oolongbench/oolong-real"
        context_column = "context_window_text"

    # Load the dataset from HuggingFace
    raw_dataset = load_dataset(hf_dataset_name, split=split)

    # Transform dataset into the required format
    def transform_example(example, idx):
        question = example["question"]
        context = example[context_column]
        answer = example["answer"]
        context_length = len(context)  # Character count for analysis

        if use_rlm:
            # RLM mode: context goes in info, short prompt
            return {
                "example_id": idx,
                "prompt": [{"role": "user", "content": question}],
                "task": "oolong",
                "answer": answer,
                "info": {"context": context, "context_length": context_length},
            }
        else:
            # Standard mode: context goes directly in the prompt
            prompt_content = (
                f"{question}\n\n"
                f"<context>\n{context}\n</context>\n\n"
                "Provide your answer inside \\boxed{}."
            )
            return {
                "example_id": idx,
                "prompt": [{"role": "user", "content": prompt_content}],
                "task": "oolong",
                "answer": answer,
                "info": {"context_length": context_length},
            }

    dataset = raw_dataset.map(
        transform_example, with_indices=True, remove_columns=raw_dataset.column_names
    )

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    # Setup judge for evaluating answer correctness
    judge_client = AsyncOpenAI(api_key=os.getenv(judge_api_key_var))

    judge_prompt_template = """Given a ground truth answer and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{ground_truth}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""

    # Helper to extract response based on mode
    def _get_response(state: vf.State, completion: vf.Messages) -> str:
        if use_rlm:
            return state.get("final_answer", "")
        else:
            # Standard mode: extract from completion using boxed answer
            if completion and isinstance(completion, list):
                content = (
                    completion[-1].get("content", "")
                    if isinstance(completion[-1], dict)
                    else str(completion[-1])
                )
            else:
                content = str(completion) if completion else ""
            return extract_boxed_answer(content)

    # Helper to extract question (RLM mode has clean question, standard has context in prompt)
    def _get_question(prompt: vf.Messages, state: vf.State) -> str:
        if use_rlm:
            # RLM mode: prompt is just the question
            return prompt[-1]["content"] if prompt else ""
        else:
            # Standard mode: extract question from the beginning of the prompt
            # The question comes before the <context> tag
            if prompt and isinstance(prompt, list):
                content = (
                    prompt[-1].get("content", "")
                    if isinstance(prompt[-1], dict)
                    else ""
                )
                if "<context>" in content:
                    return content.split("<context>")[0].strip()
                return content
            return ""

    # Reward function using model-as-judge
    async def judge_reward(
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **_kwargs,
    ) -> float:
        """Reward based on judge model evaluation."""
        question = _get_question(prompt, state)
        response = _get_response(state, completion)
        ground_truth = state.get("answer", answer)

        judge_prompt = judge_prompt_template.format(
            question=question,
            ground_truth=ground_truth,
            response=response,
        )

        judge_response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        judge_answer = judge_response.choices[0].message.content or ""
        result = 1.0 if "yes" in judge_answer.lower() else 0.0
        # Store in state for metrics logging
        state["_judge_reward"] = result
        return result

    # Metrics-only reward functions (0-weight) for comparison
    def exact_match_reward(
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **_kwargs,
    ) -> float:
        """Metric: exact match with expected answer."""
        response = _get_response(state, completion).strip()
        expected = state.get("answer", answer).strip()
        result = 1.0 if response == expected else 0.0
        # Store in state for metrics logging
        state["_exact_match"] = result
        return result

    def contains_answer_reward(
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **_kwargs,
    ) -> float:
        """Metric: final answer contains the expected value."""
        response = _get_response(state, completion).strip()
        expected = state.get("answer", answer).strip()
        result = 1.0 if expected in response else 0.0
        # Store in state for metrics logging
        state["_contains_answer"] = result
        return result

    # Build reward functions list
    reward_funcs = [judge_reward, exact_match_reward, contains_answer_reward]
    weights = [1.0, 0.0, 0.0]

    # Setup metrics logging if path is provided
    if metrics_output_path:
        metrics_logger = MetricsLogger(metrics_output_path)
        logging_reward = _create_logging_reward_func(
            metrics_logger=metrics_logger,
            is_rlm_mode=use_rlm,
            subset=subset,
        )
        reward_funcs.append(logging_reward)
        weights.append(0.0)

    rubric = vf.Rubric(funcs=reward_funcs, weights=weights)

    if use_rlm:
        env = RLMEnv(
            max_iterations=max_iterations,
            max_output_length=max_output_length,
            context_key="context",
            dataset=dataset,
            rubric=rubric,
            **kwargs,
        )
    else:
        env = SingleTurnEnv(
            dataset=dataset,
            rubric=rubric,
        )

    return env
