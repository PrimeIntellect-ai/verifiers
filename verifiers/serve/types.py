from asyncio import Future
from enum import Enum
from typing import Annotated, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict
from pydantic import SkipValidation  # nosemgrep: verifiers-no-skip-validation

from verifiers.types import (
    ClientConfig,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
)

CoercedRolloutOutput = Annotated[
    RolloutOutput, BeforeValidator(lambda v: RolloutOutput(v))
]
RunInput: TypeAlias = (  # nosemgrep: verifiers-no-skip-validation
    SkipValidation[RolloutInput]
)
GroupInput: TypeAlias = (  # nosemgrep: verifiers-no-skip-validation
    SkipValidation[list[RolloutInput]]
)


class BaseRequest(BaseModel):
    # needed for RolloutInput to work
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_type: str


class BaseResponse(BaseModel):
    # needed for RolloutOutput to work
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    error: str | None = None  # TODO: type errors later


class HealthRequest(BaseRequest):
    request_type: Literal["health"] = "health"


class HealthResponse(BaseResponse): ...


class RunRolloutRequest(BaseRequest):
    request_type: Literal["run_rollout"] = "run_rollout"

    # Pydantic rejects some provider message shapes at this protocol boundary.
    input: RunInput
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    max_retries: int
    state_columns: list[str] | None


class RunRolloutResponse(BaseResponse):
    output: CoercedRolloutOutput | None = None


class RunGroupRequest(BaseRequest):
    request_type: Literal["run_group"] = "run_group"

    # Pydantic rejects some provider message shapes at this protocol boundary.
    group_inputs: GroupInput
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    max_retries: int
    state_columns: list[str] | None


class RunGroupResponse(BaseResponse):
    outputs: list[CoercedRolloutOutput] | None = None


BaseRequestT = TypeVar("BaseRequestT", bound=BaseRequest)
BaseResponseT = TypeVar("BaseResponseT", bound=BaseResponse)


class ServerState(str, Enum):
    STARTUP = "startup"  # Initial state, before first successful health check
    HEALTHY = "healthy"  # Server is responsive and working normally
    UNHEALTHY = "unhealthy"  # Server failed health checks


class ServerError(RuntimeError): ...


class PendingRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    request: BaseRequest
    future: Future[dict]
    timeout: float | None = None
    submitted_at: float  # timestamp
