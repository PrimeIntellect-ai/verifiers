from typing import ClassVar

from pydantic import BaseModel, Field

from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.serve.delta import TraceSummary
from verifiers.v1.types import SamplingConfig


class BaseRequest(BaseModel):
    """`method` is sent as its own route frame, not as payload data."""

    method: ClassVar[str]


class BaseResponse(BaseModel):
    success: bool = True
    error: str | None = None


class HealthRequest(BaseRequest):
    method: ClassVar[str] = "health"


class HealthResponse(BaseResponse):
    pass


class CancelRequest(BaseRequest):
    """Abort an in-flight run by its wire ``request_id``. Idempotent — a
    finished or unknown run cancels successfully with ``cancelled=False``."""

    method: ClassVar[str] = "cancel"
    request_id: str


class CancelResponse(BaseResponse):
    cancelled: bool = False
    """Whether a live run was found and aborted."""


class RunRequest(BaseRequest):
    """One env-rollout, shipping the task itself: `task_data` is the dumped
    `TaskData` the server validates into the taskset's declared type."""

    method: ClassVar[str] = "run"
    task_data: dict
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunResponse(BaseResponse):
    """The end of a run whose traces already streamed as deltas (`serve.delta`)."""

    head: dict | None = None
    """The rollout's episode without its traces — its standing (`id`/`env`/`errors`,
    carrying episode-level errors even when no trace minted); task-specific data
    preserved in `model_extra`."""

    traces: list[TraceSummary] = Field(default_factory=list)
    """The finished traces in episode order, with the sizes the client's assembly
    must match."""
