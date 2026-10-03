"""Native model and compaction settings shared by Pi and Prime Agent."""

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class PiCompactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    reserve_tokens: PositiveInt | None = Field(
        None, serialization_alias="reserveTokens"
    )
    """Compact when fewer than this many context tokens remain."""
    keep_recent_tokens: PositiveInt | None = Field(
        None, serialization_alias="keepRecentTokens"
    )
    """Recent conversation tokens to retain when compacting."""
