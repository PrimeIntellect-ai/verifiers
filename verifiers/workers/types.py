from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

from verifiers.types import (
    ClientConfig,
    GenerateOutputs,
    RolloutInput,
    SamplingArgs,
    State,
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
    state: State | None = None

    @field_validator("state", mode="before")
    @classmethod
    def convert_state(cls, v: dict | None) -> State | None:
        if v is None:
            return None
        return State(**v)


class RunGroupRequest(BaseRequest):
    request_type: Literal["run_group"] = "run_group"  # type: ignore[override]

    group_inputs: list[RolloutInput]
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    score: bool = True


class RunGroupResponse(BaseResponse):
    states: list[State] | None = None

    @field_validator("states", mode="before")
    @classmethod
    def convert_states(cls, v: list[dict] | None) -> list[State] | None:
        if v is None:
            return []
        return [State(**s) for s in v]


class EvaluateRequest(BaseRequest):
    request_type: Literal["evaluate"] = "evaluate"  # type: ignore[override]

    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    num_examples: int = -1
    rollouts_per_example: int = 1
    max_concurrent: int = -1
    results_path: str | None = None
    state_columns: list[str] | None = None
    save_results: bool = False
    save_every: int = -1
    independent_scoring: bool = False


class EvaluateResponse(BaseResponse):
    results: GenerateOutputs | None = None

    @field_validator("results", mode="before")
    @classmethod
    def convert_results_state(cls, v: dict | None) -> GenerateOutputs | None:
        if v is None:
            return None
        if isinstance(v, dict) and "state" in v:
            v["state"] = [State(**s) for s in v["state"]]
        return GenerateOutputs(**v)


BaseRequestT = TypeVar("BaseRequestT", bound=BaseRequest)
BaseResponseT = TypeVar("BaseResponseT", bound=BaseResponse)
