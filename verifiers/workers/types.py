from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

from verifiers.types import (
    ClientConfig,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
)


class BaseRequest(BaseModel):
    request_type: str


class BaseResponse(BaseModel):
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
    score: bool = True


class RunRolloutResponse(BaseResponse):
    output: RolloutOutput | None = None

    @field_validator("output", mode="before")
    @classmethod
    def convert_output(cls, v: dict | RolloutOutput | None) -> RolloutOutput | None:
        if v is None:
            return None
        if isinstance(v, RolloutOutput):
            return v
        return RolloutOutput(**v)


class RunGroupRequest(BaseRequest):
    request_type: Literal["run_group"] = "run_group"  # type: ignore[override]

    group_inputs: list[RolloutInput]
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    score: bool = True


class RunGroupResponse(BaseResponse):
    outputs: list[RolloutOutput] | None = None

    @field_validator("outputs", mode="before")
    @classmethod
    def convert_outputs(
        cls, v: list[dict | RolloutOutput] | None
    ) -> list[RolloutOutput] | None:
        if v is None:
            return []
        return [o if isinstance(o, RolloutOutput) else RolloutOutput(**o) for o in v]


BaseRequestT = TypeVar("BaseRequestT", bound=BaseRequest)
BaseResponseT = TypeVar("BaseResponseT", bound=BaseResponse)
