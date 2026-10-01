"""Whole-rollout retry rules — each agent's own and the env's episode fallback."""

import re
from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator
from pydantic_config import BaseConfig


class RetryRule(BaseConfig):
    """Match all specified fields. First matching rule wins for each error."""

    type: str | None = None
    """Exact recorded exception class name, e.g. ``ProviderError``."""
    status_code: (
        list[
            Annotated[StrictInt, Field(ge=100, le=599)]
            | Literal["1xx", "2xx", "3xx", "4xx", "5xx"]
        ]
        | None
    ) = None
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


def _default_rules() -> list[RetryRule]:
    """Fresh, conservative policies for transport and provider failures."""
    return [
        RetryRule(type="ProviderError", status_code=[408, 429, "5xx"], max_retries=3),
        RetryRule(type="InterceptionError", max_retries=3),
        RetryRule(type="TunnelError", max_retries=3),
        RetryRule(
            type="SandboxError",
            message="(?i)connection reset by peer|connection timed out",
            max_retries=2,
        ),
        RetryRule(
            type="HarnessError",
            message="Tunnel not found or no longer active",
            max_retries=2,
        ),
    ]


class RetryConfig(BaseConfig):
    """Ordered rules for whole-rollout retries. No matching rule means no retry.

    Fields within a rule are ANDed. An exhausted rule never falls through to a
    later rule for that error; another captured error can still trigger a retry.
    """

    max_retries: int = Field(0, ge=0)
    """Overall retry cap across all rules. Off by default."""
    rules: list[RetryRule] = Field(default_factory=_default_rules)
    """Conservative defaults; an explicit list replaces them, and [] disables matching."""
