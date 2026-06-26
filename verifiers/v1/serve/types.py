"""Wire types for the env-server RPC (msgpack over ZMQ)."""

from typing import ClassVar

from pydantic import BaseModel, SerializeAsAny

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.task import WireTask
from verifiers.v1.trace import WireTrace
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
    index: int | None = None
    """Global cursor index, stamped by the pool broker so workers stay coherent without pinning
    `sample` to one worker. `None` for a lone server, which uses its own cursor."""


class SampleResponse(BaseResponse):
    task: SerializeAsAny[WireTask] | None = None
    """The next task the server pulls (cursor + shuffle/epoch live on the server). The caller
    echoes it back to `run_rollout` to run rollouts of it — it never addresses tasks by index.
    `SerializeAsAny` so a concrete taskset `Task` dumps its own fields, not the base schema."""


class RunRolloutRequest(BaseRequest):
    method: ClassVar[str] = "run_rollout"
    task: WireTask
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunRolloutResponse(BaseResponse):
    trace: SerializeAsAny[WireTrace] | None = None
    """A non-strict `WireTrace` (taskset-specific task fields ride in `model_extra`), so the server
    needn't assume the caller imports the taskset; a caller that does upgrades via
    `Trace[task_type(taskset_id)].model_validate(...)`. `SerializeAsAny` dumps the concrete trace's
    own fields, not the base schema."""


class RunGroupRequest(BaseRequest):
    method: ClassVar[str] = "run_group"
    task: WireTask
    n: int
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunGroupResponse(BaseResponse):
    traces: list[SerializeAsAny[WireTrace]] | None = None
    """`WireTrace`s, like `RunRolloutResponse.trace`."""
