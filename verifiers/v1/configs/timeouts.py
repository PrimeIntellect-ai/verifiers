"""Timeouts for agent stages and upstream client requests, in seconds."""

from pydantic import Field
from pydantic_config import BaseConfig


class TimeoutConfig(BaseConfig):
    setup: float | None = None
    """Agent task/harness setup through session preparation."""
    rollout: float | None = None
    """Agent solve attempt. Unset: the task's own timeout, else 4 hours; 0 disables it."""
    finalize: float | None = None
    """Agent task/harness finalization."""
    scoring: float | None = None
    """Agent task/harness metrics and scoring."""
    connect: float | None = Field(default=None, gt=0)
    """Upstream client connection timeout; unset uses the client default."""
    read: float | None = Field(default=None, gt=0)
    """Upstream client timeout between received bytes; unset uses the client default."""
