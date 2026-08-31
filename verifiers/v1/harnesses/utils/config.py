"""Shared harness configuration."""

from pydantic import PositiveInt
from pydantic_config import BaseConfig


class CompactionConfig(BaseConfig):
    """Context compaction policy for the shared chat loop."""

    summarize_at_tokens: PositiveInt | None = None
    """Compact at this token count. When unset, compact when 16k tokens remain below the
    model context window when the provider advertises it."""
