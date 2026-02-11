from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, SkipValidation

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

    # skip validation because multi-modal content type + tool calls validate weirdly
    # (https://github.com/PrimeIntellect-ai/prime-rl/pull/1249)
    input: SkipValidation[RolloutInput]
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    max_retries: int
    state_columns: list[str] | None


class RunRolloutResponse(BaseResponse):
    output: CoercedRolloutOutput | None = None


class RunGroupRequest(BaseRequest):
    request_type: Literal["run_group"] = "run_group"

    # skip validation because multi-modal content type + tool calls validate weirdly
    # (https://github.com/PrimeIntellect-ai/prime-rl/pull/1249)
    group_inputs: SkipValidation[list[RolloutInput]]
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    max_retries: int
    state_columns: list[str] | None


class RunGroupResponse(BaseResponse):
    outputs: list[CoercedRolloutOutput] | None = None


BaseRequestT = TypeVar("BaseRequestT", bound=BaseRequest)
BaseResponseT = TypeVar("BaseResponseT", bound=BaseResponse)
