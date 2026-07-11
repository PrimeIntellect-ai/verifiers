"""Typed native v1 env-server protocol: one request produces one agent graph."""

from typing import ClassVar

from pydantic import BaseModel, Field, field_serializer, field_validator

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.task import WireTaskData
from verifiers.v1.topology import AgentGraph
from verifiers.v1.trace import Trace
from verifiers.v1.types import SamplingConfig


class BaseRequest(BaseModel):
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
    task_ids: list[int] = Field(default_factory=list)


class LegacyInfoResponse(InfoResponse):
    requires_group_scoring: bool = False


class RunRequest(BaseRequest):
    method: ClassVar[str] = "run"
    task_idx: int
    client: ClientConfig
    model: str
    sampling: SamplingConfig


class RunResponse(BaseResponse):
    graph: AgentGraph | None = None

    @field_validator("graph", mode="before")
    @classmethod
    def _load_graph(cls, graph):
        return AgentGraph.load(graph) if isinstance(graph, dict) else graph

    @field_serializer("graph")
    def _serialize_graph(self, graph: AgentGraph | None) -> dict | None:
        return graph.to_record() if graph is not None else None


# Legacy v0 bridge protocol. Native v1 never dispatches these routes.
class RunRolloutRequest(RunRequest):
    method: ClassVar[str] = "run_rollout"


class RunRolloutResponse(BaseResponse):
    trace: Trace[WireTaskData] | None = None


class RunGroupRequest(RunRequest):
    method: ClassVar[str] = "run_group"
    n: int


class RunGroupResponse(BaseResponse):
    traces: list[Trace[WireTaskData]] | None = None
