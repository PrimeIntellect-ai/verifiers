from typing import ClassVar

from pydantic import BaseModel, Field, field_serializer

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.task import WireTaskData
from verifiers.v1.trace import Trace, WireEpisode
from verifiers.v1.types import SamplingConfig

PROTOCOL_VERSION = 2
"""The serve wire protocol: bumped when response shapes change (consumers —
prime-rl's orchestrator — read it off `info` to detect a mismatched server). In
protocol 2, the rollout route is `run` and answers the episode's flat traces
plus its shared stamp — the renamed route makes a stale server fail loudly
instead of validating to an empty response."""


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
    """Task count; `None` means the taskset is infinite (bound runs with `num_tasks`)."""
    requires_group_scoring: bool = False
    """Whether tasks must be run as whole groups — legacy (v0) envs only; a v1
    server always reports False (sibling-dependent signals run inside the env's
    own rollout)."""
    protocol: int = PROTOCOL_VERSION
    """The server's wire protocol version (`PROTOCOL_VERSION`)."""


class RunRequest(BaseRequest):
    method: ClassVar[str] = "run"
    task_idx: int = Field(ge=0)
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunResponse(BaseResponse):
    episode: WireEpisode | None = None
    """The rollout's episode record — flat, self-contained traces plus the shared
    stamp (which carries episode-level errors even when no trace minted);
    task-specific data preserved in `model_extra`."""

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
    traces: list[Trace[WireTaskData]] | None = None

    @field_serializer("traces")
    def _ser_traces(
        self, traces: "list[Trace[WireTaskData]] | None"
    ) -> list[dict] | None:
        return [t.model_dump() for t in traces] if traces is not None else None
