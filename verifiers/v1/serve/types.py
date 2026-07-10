"""Wire types for the env-server RPC (msgpack over ZMQ)."""

from typing import ClassVar

from pydantic import BaseModel, Field, field_serializer

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.task import WireTaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import SamplingConfig


class BaseRequest(BaseModel):
    """Marker base for requests. ``method`` is the RPC route — a class var sent as its
    own ZMQ frame, not a model field — so the request type never rides as payload data
    and the request models stay pure data."""

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
    """Number of tasks in the taskset — the index range the caller samples from."""
    task_data_idxs: list[int] = Field(default_factory=list)
    """`TaskData.idx` for each task position, used to map server requests to persisted
    traces when the stable data index differs from the task's list position."""
    group_idxs: list[int] = Field(default_factory=list)
    """Task positions that define `@group_reward`s. The caller runs only these through
    `run_group` and plans their resume whole-group; other tasks run per rollout."""


class RunRolloutRequest(BaseRequest):
    method: ClassVar[str] = "run_rollout"
    task_idx: int
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunRolloutResponse(BaseResponse):
    trace: Trace[WireTaskData] | None = None
    """A typed `Trace` with a non-strict `WireTaskData` (taskset-specific task fields ride in
    `model_extra`), so the server needn't assume the caller imports the taskset. A caller
    that imports it resolves `trace.task_class` within `task_types(taskset_id)`."""

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
    """Typed `Trace`s with non-strict `WireTaskData`, like `RunRolloutResponse.trace`."""

    @field_serializer("traces")
    def _ser_traces(
        self, traces: "list[Trace[WireTaskData]] | None"
    ) -> list[dict] | None:
        return [t.model_dump() for t in traces] if traces is not None else None
