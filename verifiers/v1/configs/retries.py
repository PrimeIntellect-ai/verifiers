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
    checkpoint: bool = True
    """Checkpoint the runtime's filesystem before every model turn (on runtimes that
    support it, e.g. prime and docker) and resume a retry from the newest checkpoint:
    the trace keeps its turns up to there and the harness continues the conversation
    in a box restored to that state. False retries from scratch. Only the box is
    restored — state kept outside it (a tool server's typed rollout state) stays as
    the failed attempt left it."""
