from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

from verifiers.types import (
    ClientConfig,
    GenerateOutputs,
    RolloutInput,
    SamplingArgs,
    State,
)


def _dict_to_state(v: Any) -> State:
    """Convert a dict to a State object if needed."""
    if isinstance(v, State):
        return v
    if isinstance(v, dict):
        return State(v)
    raise ValueError(f"Expected State or dict, got {type(v)}")


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
    def convert_state(cls, v: Any) -> State | None:
        if v is None:
            return None
        return _dict_to_state(v)


class RunGroupRequest(BaseRequest):
    request_type: Literal["run_group"] = "run_group"  # type: ignore[override]

    group_inputs: list[RolloutInput]
    client_config: ClientConfig
    model: str
    sampling_args: SamplingArgs
    score: bool = True


class RunGroupResponse(BaseResponse):
    states: list[State]

    @field_validator("states", mode="before")
    @classmethod
    def convert_states(cls, v: Any) -> list[State]:
        if not isinstance(v, list):
            raise ValueError(f"Expected list, got {type(v)}")
        return [_dict_to_state(s) for s in v]


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
    results: GenerateOutputs

    @field_validator("results", mode="before")
    @classmethod
    def convert_results_state(cls, v: Any) -> GenerateOutputs:
        if isinstance(v, dict) and "state" in v:
            v["state"] = [_dict_to_state(s) for s in v["state"]]
        return v


BaseRequestT = TypeVar("BaseRequestT", bound=BaseRequest)
BaseResponseT = TypeVar("BaseResponseT", bound=BaseResponse)
