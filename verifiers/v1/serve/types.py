from typing import ClassVar

from pydantic import BaseModel, field_serializer

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
    num_tasks: int = 0
    requires_group_scoring: bool = False
    """Whether tasks must be run and resumed as whole groups."""


class RunRolloutRequest(BaseRequest):
    method: ClassVar[str] = "run_rollout"
    task_idx: int
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunRolloutResponse(BaseResponse):
    trace: Trace[WireTaskData] | None = None
    """A trace whose task-specific data is preserved in `model_extra`."""

    @field_serializer("trace")
    def _ser_trace(self, trace: "Trace[WireTaskData] | None") -> dict | None:
        return trace.model_dump() if trace is not None else None


class RunGroupRequest(BaseRequest):
    method: ClassVar[str] = "run_group"
    task_idx: int
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
