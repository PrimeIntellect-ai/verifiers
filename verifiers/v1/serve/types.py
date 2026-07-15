from typing import ClassVar

from pydantic import BaseModel, Field, field_serializer, model_validator

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.task import WireTaskData
from verifiers.v1.trace import Trace
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
    """Task count. Only the legacy bridge (whose dataset lives server-side) reports one;
    a v1 server is stateless — its tasks live on the client — so this stays `None`."""
    requires_group_scoring: bool = False
    """Whether tasks must be run and resumed as whole groups."""


class TaskAddressing(BaseModel):
    """How a run request names its task: v1 ships the task itself (`task_data`, a
    `TaskData.full_dump()` the server validates into the taskset's declared type); the
    legacy bridge addresses its server-side dataset by row (`task_idx`)."""

    task_data: dict | None = None
    """The task's wire data (v1). The server rebuilds and pydantic-validates it."""
    task_idx: int | None = Field(None, ge=0)
    """Dataset row index (legacy v0 bridge only)."""

    @model_validator(mode="after")
    def _exactly_one(self) -> "TaskAddressing":
        if (self.task_data is None) == (self.task_idx is None):
            raise ValueError(
                "exactly one of task_data (v1) or task_idx (legacy) must be set"
            )
        return self


class RunRolloutRequest(TaskAddressing, BaseRequest):
    method: ClassVar[str] = "run_rollout"
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunRolloutResponse(BaseResponse):
    trace: Trace[WireTaskData] | None = None
    """A trace whose task-specific data is preserved in `model_extra`."""

    @field_serializer("trace")
    def _ser_trace(self, trace: "Trace[WireTaskData] | None") -> dict | None:
        return trace.model_dump() if trace is not None else None


class RunGroupRequest(TaskAddressing, BaseRequest):
    method: ClassVar[str] = "run_group"
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
