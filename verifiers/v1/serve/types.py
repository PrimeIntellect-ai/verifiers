from typing import ClassVar

from pydantic import BaseModel, Field, field_serializer, model_validator

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.task import WireTaskData  # noqa: F401  (docstring reference)
from verifiers.v1.episode import WireEpisode
from verifiers.v1.trace import WireTrace
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


class InfoRequest(BaseRequest):
    method: ClassVar[str] = "info"


class InfoResponse(BaseResponse):
    num_tasks: int | None = None
    """Task count. Only the legacy bridge (whose dataset lives server-side) reports
    one; a v1 server is stateless — its tasks live on the client — so this stays
    `None`."""
    requires_group_scoring: bool = False
    """Whether tasks must be run as whole groups — legacy (v0) envs only; a v1
    server always reports False (sibling-dependent signals run inside the env's
    own rollout)."""


class RunRequest(BaseRequest):
    """One env-rollout. v1 ships the task itself (`task_data`, the dumped `TaskData`
    the server validates into the taskset's declared type); the legacy bridge
    addresses its server-side dataset by row (`task_idx`)."""

    method: ClassVar[str] = "run"
    task_data: dict | None = None
    task_idx: int | None = Field(None, ge=0)
    client: ClientConfig
    model: str
    sampling: SamplingConfig

    @model_validator(mode="after")
    def _exactly_one(self) -> "RunRequest":
        if (self.task_data is None) == (self.task_idx is None):
            raise ValueError(
                "exactly one of task_data (v1) or task_idx (legacy) must be set"
            )
        return self


class RunResponse(BaseResponse):
    episode: WireEpisode | None = None
    """The rollout's episode — its standing (`id`/`env`/`errors`, carrying
    episode-level errors even when no trace minted) inlined next to its flat,
    self-contained traces; task-specific data preserved in `model_extra`."""

    @field_serializer("episode")
    def _ser_episode(self, episode: "WireEpisode | None") -> dict | None:
        return episode.model_dump() if episode is not None else None


class RunGroupRequest(BaseRequest):
    """Legacy (v0) route: group-scored v0 envs run a task's n rollouts together."""

    method: ClassVar[str] = "run_group"
    task_idx: int = Field(ge=0)
    n: int
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunGroupResponse(BaseResponse):
    traces: list[WireTrace] | None = None

    @field_serializer("traces")
    def _ser_traces(self, traces: "list[WireTrace] | None") -> list[dict] | None:
        return [t.model_dump() for t in traces] if traces is not None else None
