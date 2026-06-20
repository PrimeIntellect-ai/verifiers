"""Wire types for the env-server RPC (msgpack over ZMQ)."""

from typing import ClassVar

from pydantic import BaseModel, Field, field_serializer

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.task import WireTask
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
    requires_group_scoring: bool = False
    """Whether the taskset defines `@group_reward`s (caller must use `run_group`)."""


class ModelRuntimeConfig(BaseModel):
    client: ClientConfig
    model: str
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


class RunRolloutRequest(BaseRequest):
    method: ClassVar[str] = "run_rollout"
    task_idx: int
    actor: ModelRuntimeConfig


class RunRolloutResponse(BaseResponse):
    trace: Trace[WireTask] | None = None
    """A typed `Trace` with a non-strict `WireTask` (taskset-specific task fields ride in
    `model_extra`), so the server needn't assume the caller imports the taskset. A caller
    that *does* import it upgrades via `Trace[task_type(taskset_id)].model_validate(...)`."""

    @field_serializer("trace")
    def _ser_trace(self, trace: "Trace[WireTask] | None") -> dict | None:
        return trace.model_dump() if trace is not None else None


class RunGroupRequest(BaseRequest):
    method: ClassVar[str] = "run_group"
    task_idx: int
    n: int
    actor: ModelRuntimeConfig


class RunGroupResponse(BaseResponse):
    traces: list[Trace[WireTask]] | None = None
    """Typed `Trace`s with non-strict `WireTask`, like `RunRolloutResponse.trace`."""

    @field_serializer("traces")
    def _ser_traces(self, traces: "list[Trace[WireTask]] | None") -> list[dict] | None:
        return [t.model_dump() for t in traces] if traces is not None else None


class AdvantageBranch(BaseModel):
    branch_index: int
    loss: str
    advantages: list[float]
    mask: list[bool]


class TraceAdvantages(BaseModel):
    branches: list[AdvantageBranch]


class RunAdvantagesRequest(BaseRequest):
    method: ClassVar[str] = "run_advantages"
    refs: list[str]
    traces: list[Trace[WireTask]]
    models: dict[str, ModelRuntimeConfig] = Field(default_factory=dict)

    @field_serializer("traces")
    def _ser_traces(self, traces: "list[Trace[WireTask]]") -> list[dict]:
        return [t.model_dump() for t in traces]


class RunAdvantagesResponse(BaseResponse):
    advantages: list[TraceAdvantages] | None = None
