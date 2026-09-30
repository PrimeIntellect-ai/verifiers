"""Retry settings for rollouts, episodes, and upstream client requests."""

from pydantic import Field
from pydantic_config import BaseConfig


class RetryConfig(BaseConfig):
    """Retry count and optional rollout error filters.

    Client requests use `max_retries` only; `include` and `exclude` name exception
    classes for agent and episode retries.
    """

    max_retries: int = Field(0, ge=0)
    """Retries beyond the first attempt. Off by default."""
    include: list[str] = Field(default_factory=list)
    """Only retry errors whose type is listed. Empty = retry anything not excluded."""
    exclude: list[str] = Field(default_factory=list)
    """Never retry errors whose type is listed (wins over `include`)."""
