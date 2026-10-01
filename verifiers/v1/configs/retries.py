"""Whole-rollout retry rules — each agent's own and the env's episode fallback."""

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
    """Ordered rules for opt-in whole-rollout retries.

    Fields within a rule are ANDed. An exhausted rule never falls through to a
    later rule for that error; another captured error can still trigger a retry.
    """

    max_retries: int = Field(0, ge=0)
    """Overall retry cap across all rules. Zero disables retries regardless of rule budgets."""
    rules: list[RetryRule] | None = None
    """Omitted rules retry any captured error up to the overall cap. Explicit rules
    restrict matching; an empty list disables matching."""
