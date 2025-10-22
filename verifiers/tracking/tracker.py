# ABOUTME: Base abstraction for experiment tracking across different backends.
# ABOUTME: Provides unified interface for logging metrics, completions, and artifacts.

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional



class Tracker(ABC):
    """Base class for experiment tracking."""

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self.project = project
        self.name = name
        self.config = config or {}
        self._initialized = False
        self.logger = logging.getLogger(f"verifiers.tracking.{self.__class__.__name__}")

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def init(self, **kwargs) -> None:
        """Initialize the tracker. Called once before training/evaluation."""
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None, **kwargs
    ) -> None:
        """Log scalar metrics."""
        pass

    @abstractmethod
    def log_table(
        self,
        table_name: str,
        data: dict[str, list[Any]],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Log tabular data (e.g., completions with prompts and rewards)."""
        pass

    def log_completions(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Log completions with prompts and rewards. Default implementation uses log_table."""
        data: dict[str, list[Any]] = {
            "prompt": prompts,
            "completion": completions,
            "reward": rewards,
        }
        self.log_table("completions", data, step=step, **kwargs)

    def log_config(self, config: dict[str, Any], **kwargs) -> None:
        """Log configuration. Default implementation stores in self.config."""
        self.config.update(config)

    def finish(self, **kwargs) -> None:
        """Clean up tracker resources. Called at end of training/evaluation."""
        pass

    def __enter__(self):
        if not self._initialized:
            self.init()
            self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class CompositeTracker(Tracker):
    """Tracker that forwards calls to multiple backends."""

    def __init__(self, trackers: list[Tracker], **kwargs):
        super().__init__(**kwargs)
        self.trackers = trackers

    def init(self, **kwargs) -> None:
        for tracker in self.trackers:
            try:
                tracker.init(**kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to initialize tracker {tracker.__class__.__name__}: {e}")

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None, **kwargs
    ) -> None:
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step=step, **kwargs)
            except Exception as e:
                self.logger.warning(
                    f"Failed to log metrics to {tracker.__class__.__name__}: {e}"
                )

    def log_table(
        self,
        table_name: str,
        data: dict[str, list[Any]],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        for tracker in self.trackers:
            try:
                tracker.log_table(table_name, data, step=step, **kwargs)
            except Exception as e:
                self.logger.warning(
                    f"Failed to log table to {tracker.__class__.__name__}: {e}"
                )

    def log_completions(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        for tracker in self.trackers:
            try:
                tracker.log_completions(prompts, completions, rewards, step=step, **kwargs)
            except Exception as e:
                self.logger.warning(
                    f"Failed to log completions to {tracker.__class__.__name__}: {e}"
                )

    def log_config(self, config: dict[str, Any], **kwargs) -> None:
        for tracker in self.trackers:
            try:
                tracker.log_config(config, **kwargs)
            except Exception as e:
                self.logger.warning(
                    f"Failed to log config to {tracker.__class__.__name__}: {e}"
                )

    def finish(self, **kwargs) -> None:
        for tracker in self.trackers:
            try:
                tracker.finish(**kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to finish tracker {tracker.__class__.__name__}: {e}")


class NullTracker(Tracker):
    """No-op tracker for disabling tracking."""

    def init(self, **kwargs) -> None:
        self._initialized = True

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None, **kwargs
    ) -> None:
        pass

    def log_table(
        self,
        table_name: str,
        data: dict[str, list[Any]],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        pass
