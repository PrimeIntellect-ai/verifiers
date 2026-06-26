"""Wire types for the env-server RPC (msgpack over ZMQ)."""

from typing import ClassVar

from pydantic import BaseModel, field_serializer

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
    num_tasks: int | None = None
    """How many distinct tasks the taskset has — the caller pulls (it doesn't address by index),
    and uses this to bound a finite eval (pull `num_tasks`, never wrapping). `None` only when the
    taskset is `INFINITE` (the server streams a never-ending order); the caller then bounds the
    run with `--num-tasks`. A finite taskset (list or finite generator) is materialized and reports
    its count; training pulls past it (the server loops, reshuffling each epoch)."""
    requires_group_scoring: bool = False
    """Whether the taskset defines `@group_reward`s (caller must use `run_group`)."""


class SampleRequest(BaseRequest):
    method: ClassVar[str] = "sample"


class SampleResponse(BaseResponse):
    task: WireTask | None = None
    """The next task the server pulls (cursor + shuffle/epoch live on the server). The caller
    echoes it back to `run_rollout` to run rollouts of it — it never addresses tasks by index."""

    @field_serializer("task")
    def _ser_task(self, task: "WireTask | None") -> dict | None:
        return task.model_dump() if task is not None else None


class RunRolloutRequest(BaseRequest):
    method: ClassVar[str] = "run_rollout"
    task: WireTask
    """The task to run, as returned by `sample()`. Echoed back so the server runs this exact
    task — the caller never addresses the dataset by index."""
    client: ClientConfig
    model: str
    sampling: SamplingConfig


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
    task: WireTask
    """The task to run the group of, as returned by `sample()`. Echoed back so `sample()` stays
    the only place the cursor advances."""
    n: int
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunGroupResponse(BaseResponse):
    traces: list[Trace[WireTask]] | None = None
    """Typed `Trace`s with non-strict `WireTask`, like `RunRolloutResponse.trace`."""

    @field_serializer("traces")
    def _ser_traces(self, traces: "list[Trace[WireTask]] | None") -> list[dict] | None:
        return [t.model_dump() for t in traces] if traces is not None else None
