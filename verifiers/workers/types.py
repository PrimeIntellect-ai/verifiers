from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict

from verifiers.types import (
    ClientConfig,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
)

CoercedRolloutOutput = Annotated[
    RolloutOutput, BeforeValidator(lambda v: RolloutOutput(v))
]


class BaseRequest(BaseModel):
    request_type: str


class BaseResponse(BaseModel):
    # needed for RolloutInput and RolloutOutput to work
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    error: str | None = None  # TODO: type errors later


class HealthRequest(BaseRequest):
    request_type: Literal["health"] = "health"  # type: ignore[override]


class HealthResponse(BaseResponse): ...


class RunRolloutRequest(BaseRequest):
    request_type: Literal["run_rollout"] = "run_rollout"  # type: ignore[override]

    input: RolloutInput
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs


class RunRolloutResponse(BaseResponse):
    output: CoercedRolloutOutput


class RunGroupRequest(BaseRequest):
    request_type: Literal["run_group"] = "run_group"  # type: ignore[override]

    group_inputs: list[RolloutInput]
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs


class RunGroupResponse(BaseResponse):
    outputs: list[CoercedRolloutOutput]


BaseRequestT = TypeVar("BaseRequestT", bound=BaseRequest)
BaseResponseT = TypeVar("BaseResponseT", bound=BaseResponse)
