"""Retry rules shared by episode, agent rollout, and isolated scoring boundaries."""

import re
from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator
from pydantic_config import BaseConfig

StatusCode = (
    Annotated[StrictInt, Field(ge=100, le=599)]
    | Literal["1xx", "2xx", "3xx", "4xx", "5xx"]
)


class RetryRule(BaseConfig):
    """Match all specified fields. First matching rule wins for each error."""

    type: str | None = None
    """Exact recorded exception class name, e.g. ``ProviderError``."""
    status_code: list[StatusCode] | None = None
    """Match any listed HTTP status or status class; absent status never matches."""
    message: str | None = None
    """Regex search of the error message; plain text matches a substring."""
    max_retries: int = Field(ge=0)
    """Required retry budget across the run. Explicit zero excludes matching errors."""

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"Invalid message regex: {exc}") from exc
        return value


class RetryConfig(BaseConfig):
    """Ordered rules for retries at the owning config’s execution boundary.

    Fields within a rule are ANDed. An exhausted rule never falls through to a
    later rule for that error; another captured error can still trigger a retry.
    """

    max_retries: int = Field(0, ge=0)
    """Default budget for errors that match no rule. Matching rules override it."""
    rules: list[RetryRule] = Field(default_factory=list)
    """Ordered overrides; a matching rule may enable retries or deny them with zero."""
    checkpoint: bool = True
    """Experimental. With a retry budget, checkpoint the runtime's filesystem as model
    turns go upstream (every `checkpoint_interval`-th turn; on a runtime that supports
    it: prime; the snapshot runs alongside the model call, and the reply waits only if
    the snapshot is not captured yet) and restore a retry from the newest one:
    the trace keeps its turns up to there, its typed state as of then, and a new
    harness process continues the conversation in a box restored to that state,
    told that the box was restored (`HarnessSession.restore`). False takes no
    checkpoints and retries from scratch; so does a run with no retry budget, or a
    harness without `SUPPORTS_CHECKPOINT_RESTORE`. Snapshots are freed when the run
    is final. Known limits: a harness with concurrent sub-agents can snapshot while
    one of them is mid-tool, and processes the agent left running (background jobs,
    sub-agents, a REPL's variables) are gone after a restore, which cold boots."""
    checkpoint_interval: int = Field(1, ge=1)
    """Checkpoint before every this-many-th model turn (the 1st, the 1+n-th, ...). A
    larger interval takes fewer snapshots, and a restore loses up to this many turns."""
