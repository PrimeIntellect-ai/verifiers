"""Timeouts for agent stages and upstream client requests, in seconds."""

from pydantic import Field
from pydantic_config import BaseConfig


class TimeoutConfig(BaseConfig):
    """Agent stage and upstream client timeouts. Unset agent stages use the task's
    timeout for that stage, then no limit; unset client fields use client defaults."""

    # one shared budget: task setup + provisioning
    setup: float | None = Field(None, gt=0)
    """Agent task/harness setup through session preparation."""
    rollout: float | None = Field(None, gt=0)
    """Agent solve attempt."""
    finalize: float | None = Field(None, gt=0)
    """Agent task/harness finalization."""
    scoring: float | None = Field(None, gt=0)
    """Agent task/harness metrics and scoring."""
    connect: float | None = Field(default=None, gt=0)
    """Upstream client connection timeout; unset uses the client default."""
    read: float | None = Field(default=None, gt=0)
    """Upstream client timeout between received bytes; unset uses the client default."""
