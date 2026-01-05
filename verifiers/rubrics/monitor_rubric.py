from typing import Callable

from verifiers.rubrics.rubric import Rubric
from verifiers.types import RewardFunc

StateKey = str
RenamedStateKey = tuple[StateKey, str]
RenamedTransformedStateKey = tuple[StateKey, str, Callable[..., float]]


class MonitorRubric(Rubric):
    """Simple rubric that only contains metrics for logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert all(weight == 0.0 for weight in self.weights)

    def add_reward_func(self, *args, **kwargs):
        """Cannot add reward func to monitor rubric."""
        self.logger.warning("Cannot add reward func to monitor rubric. Ignoring.")

    def add_metric(self, func: RewardFunc, *args, **kwargs):
        """Ensure that the metric has weight 0.0"""
        super().add_metric(func, weight=0.0)
