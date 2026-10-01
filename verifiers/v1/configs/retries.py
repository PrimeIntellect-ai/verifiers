"""Whole-rollout retry policy — each agent's own and the env's episode fallback."""

from pydantic import Field
from pydantic_config import BaseConfig


class RetryConfig(BaseConfig):
    """Retry a whole rollout when it ends with a captured error. `include`/`exclude`
    name exception classes (e.g. ``ProviderError``, ``SandboxError``)."""

    max_retries: int = Field(0, ge=0)
    """Whole-rollout retries beyond the first attempt. Off by default — the SDKs
    already retry transient per-call faults; rerunning a whole trajectory is opt-in."""
    include: list[str] = Field(default_factory=list)
    """Only retry errors whose type is listed. Empty = retry anything not excluded."""
    exclude: list[str] = Field(default_factory=list)
    """Never retry errors whose type is listed (wins over `include`)."""
    retry_nonretryable: bool = False
    """Also rerun faults the error itself declares deterministic (`retryable is
    False` — e.g. a 4xx `ProviderError` or a `TaskError`). Off by default, since a
    rerun usually just reproduces them; opt in when you accept the (often wasted)
    cost — e.g. a flaky authored grader. `include`/`exclude` still scope which
    types this applies to."""
