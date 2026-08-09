from typing import ClassVar

from pydantic import BaseModel, SerializeAsAny

from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.episode import WireEpisode
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


class RunRequest(BaseRequest):
    """One env-rollout, shipping the task itself: `task_data` is the dumped
    `TaskData` the server validates into the taskset's declared type."""

    method: ClassVar[str] = "run"
    task_data: dict
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunResponse(BaseResponse):
    episode: SerializeAsAny[WireEpisode] | None = None
    """The rollout's episode — its standing (`id`/`env`/`errors`, carrying
    episode-level errors even when no trace minted) inlined next to its flat,
    self-contained traces; task-specific data preserved in `model_extra`."""
