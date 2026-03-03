"""Browserbase monitoring rubric for tracking infrastructure metrics.

This rubric provides metric functions for monitoring Browserbase-related
errors, retries, and session creation without affecting training rewards.
All metrics have weight=0.0 by default.
"""

from typing import Any

import verifiers as vf


def browserbase_session_created(state: vf.State, **kwargs: Any) -> float:
    """Track whether a Browserbase session was successfully created.

    Returns:
        1.0 if session was created successfully, 0.0 otherwise.
    """
    metrics = state.get("browserbase_metrics", {})
    return 1.0 if metrics.get("session_created", False) else 0.0


def browserbase_session_failed(state: vf.State, **kwargs: Any) -> float:
    """Track whether session creation failed.

    Returns:
        1.0 if session creation failed, 0.0 otherwise.
    """
    metrics = state.get("browserbase_metrics", {})
    return 0.0 if metrics.get("session_created", False) else 1.0


def browserbase_rate_limit_errors(state: vf.State, **kwargs: Any) -> float:
    """Track number of rate limit errors (429/503) encountered.

    Returns:
        Count of rate limit errors as a float.
    """
    metrics = state.get("browserbase_metrics", {})
    return float(metrics.get("rate_limit_errors", 0))


def browserbase_had_rate_limit(state: vf.State, **kwargs: Any) -> float:
    """Track whether any rate limit errors occurred.

    Returns:
        1.0 if at least one rate limit error occurred, 0.0 otherwise.
    """
    metrics = state.get("browserbase_metrics", {})
    return 1.0 if metrics.get("rate_limit_errors", 0) > 0 else 0.0


def browserbase_retries(state: vf.State, **kwargs: Any) -> float:
    """Track number of retries during session creation.

    Returns:
        Number of retries as a float.
    """
    metrics = state.get("browserbase_metrics", {})
    return float(metrics.get("session_creation_retries", 0))


def browserbase_had_retries(state: vf.State, **kwargs: Any) -> float:
    """Track whether any retries were needed.

    Returns:
        1.0 if retries occurred, 0.0 otherwise.
    """
    metrics = state.get("browserbase_metrics", {})
    return 1.0 if metrics.get("session_creation_retries", 0) > 0 else 0.0


def browserbase_error_occurred(state: vf.State, **kwargs: Any) -> float:
    """Track whether any error occurred (even if session eventually succeeded).

    Returns:
        1.0 if an error was recorded, 0.0 otherwise.
    """
    metrics = state.get("browserbase_metrics", {})
    return 1.0 if metrics.get("error") is not None else 0.0


class BrowserbaseMonitoringRubric(vf.Rubric):
    """Rubric for monitoring Browserbase infrastructure metrics.

    All metrics have weight=0.0 by default, so they are tracked in
    state["metrics"] but do not affect the training reward.

    Example usage:
        ```python
        from verifiers.envs.integrations.browser_env.browserbase_rubric import (
            BrowserbaseMonitoringRubric
        )

        # Create monitoring rubric
        bb_rubric = BrowserbaseMonitoringRubric()

        # Combine with your task rubric using RubricGroup
        rubric = vf.RubricGroup([task_rubric, bb_rubric])
        ```

    Tracked metrics (all weight=0.0):
        - browserbase_session_created: 1.0 if session created, 0.0 otherwise
        - browserbase_session_failed: 1.0 if session failed, 0.0 otherwise
        - browserbase_rate_limit_errors: count of 429/503 errors
        - browserbase_had_rate_limit: 1.0 if any rate limits hit
        - browserbase_retries: count of retry attempts
        - browserbase_had_retries: 1.0 if any retries needed
        - browserbase_error_occurred: 1.0 if any error recorded
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Add all monitoring metrics with weight=0.0
        self.add_metric(browserbase_session_created)
        self.add_metric(browserbase_session_failed)
        self.add_metric(browserbase_rate_limit_errors)
        self.add_metric(browserbase_had_rate_limit)
        self.add_metric(browserbase_retries)
        self.add_metric(browserbase_had_retries)
        self.add_metric(browserbase_error_occurred)
