import json
import sys
import time
import uuid
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    TypeAlias,
    TypeVar,
    overload,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

if TYPE_CHECKING:
    from anthropic.types import RedactedThinkingBlock
    from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
    from datasets import Dataset

    from verifiers.clients import Client
    from verifiers.errors import Error
else:
    RedactedThinkingBlock = Any
    AnthropicThinkingBlock = Any

if sys.version_info < (3, 12):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict

# Client / message type literals
ClientType = Literal[
    "openai_completions",
    "openai_chat_completions",
    "openai_chat_completions_token",
    "openai_responses",
    "renderer",
    "anthropic_messages",
    "nemorl_chat_completions",
]
EndpointApi = Literal[
    "chat",
    "chat_completions",
    "completions",
    "responses",
    "messages",
    "openai_chat_completions",
    "openai_completions",
    "openai_responses",
    "anthropic_messages",
]
MessageType = Literal["chat", "completion"]  # deprecated


# Provider-agnostic message + response types
class CustomBaseModel(BaseModel):
    """Allow extras and dict-like attribute access."""

    model_config = ConfigDict(extra="allow")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.model_dump() == dict(other)
        return super().__eq__(other)


class TextMessage(CustomBaseModel):
    role: Literal["text"] = "text"
    content: str


class TextContentPart(CustomBaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageUrlSource(CustomBaseModel):
    url: str


class ImageUrlContentPart(CustomBaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlSource


class InputAudioSource(CustomBaseModel):
    data: str
    format: str


class InputAudioContentPart(CustomBaseModel):
    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudioSource


class GenericContentPart(CustomBaseModel):
    type: str


ContentPart: TypeAlias = (
    TextContentPart
    | ImageUrlContentPart
    | InputAudioContentPart
    | GenericContentPart
    | dict[str, Any]
)
MessageContent: TypeAlias = str | list[ContentPart]


class SystemMessage(CustomBaseModel):
    role: Literal["system"] = "system"
    content: MessageContent

    @classmethod
    def from_path(cls, path: str | Path) -> "SystemMessage":
        return cls(content=Path(path).read_text(encoding="utf-8"))


class UserMessage(CustomBaseModel):
    role: Literal["user"] = "user"
    content: MessageContent


class ToolCall(CustomBaseModel):
    id: str
    name: str
    arguments: str


ThinkingBlock: TypeAlias = AnthropicThinkingBlock | RedactedThinkingBlock


class AssistantMessage(CustomBaseModel):
    role: Literal["assistant"] = "assistant"
    content: MessageContent | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[ThinkingBlock] | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(CustomBaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: MessageContent


Message: TypeAlias = (
    SystemMessage | UserMessage | AssistantMessage | ToolMessage | TextMessage
)
Messages = list[Message]


class Tool(CustomBaseModel):
    name: str
    description: str
    parameters: dict[str, object]
    strict: bool | None = None


class Usage(CustomBaseModel):
    prompt_tokens: int
    reasoning_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseTokens(CustomBaseModel):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    routed_experts: str | None = None  # base64 NumPy [seq_len, layers, topk]
    tool_output_train_mask: list[bool] | None = None
    # Renderer-emitted multimodal sidecar (renderers.base.MultiModalData)
    # carrying processed pixel_values / placeholder ranges per modality.
    # Populated by the renderer client when the rollout went through a
    # multimodal-aware renderer; ``None`` otherwise. Stored as ``Any`` to
    # avoid a hard import dependency on ``renderers`` at this layer.
    multi_modal_data: Any | None = None


FinishReason = Literal["stop", "length", "tool_calls"] | None


class ResponseMessage(AssistantMessage):
    finish_reason: FinishReason
    is_truncated: bool | None
    tokens: ResponseTokens | None = None


class Response(CustomBaseModel):
    id: str
    created: int
    model: str
    usage: Usage | None = None
    message: ResponseMessage  # can call tools


# Core data types
Info = dict[str, Any]
SamplingArgs = dict[str, Any]
IndividualRewardFunc = Callable[..., float | Awaitable[float]]
GroupRewardFunc = Callable[..., list[float] | Awaitable[list[float]]]
RewardFunc = IndividualRewardFunc | GroupRewardFunc
DatasetBuilder: TypeAlias = "Callable[[], Dataset]"


class TrajectoryStepTokens(TypedDict):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    overlong_prompt: bool
    is_truncated: bool
    routed_experts: str | None  # base64 NumPy [seq_len, layers, topk]
    # Renderer-emitted multimodal sidecar (renderers.base.MultiModalData)
    # carrying processed pixel_values / placeholder ranges per modality.
    # ``NotRequired`` because text-only rollouts (and non-renderer client
    # types) never populate it.
    multi_modal_data: NotRequired[Any]


class TokenUsage(TypedDict):
    input_tokens: float
    output_tokens: float
    final_input_tokens: NotRequired[float]
    final_output_tokens: NotRequired[float]


class ModelPricing(TypedDict):
    input_usd_per_mtok: float
    output_usd_per_mtok: float


class EvalCost(TypedDict):
    input_usd: float
    output_usd: float
    total_usd: float


class VersionInfo(TypedDict):
    vf_version: str
    vf_commit: str | None
    env_version: str | None
    env_commit: str | None


class TrajectoryStep(TypedDict):
    prompt: Messages
    completion: Messages
    response: Response
    tokens: TrajectoryStepTokens | None
    reward: float | None
    advantage: float | None
    is_truncated: bool
    trajectory_id: str
    extras: dict[str, Any]


class BaseRolloutInput(TypedDict):
    prompt: Messages
    example_id: int


class RolloutInput(BaseRolloutInput, total=False):
    # required: prompt, example_id
    # optional: answer, info
    answer: str
    info: Info | str


class TimeSpan(CustomBaseModel):
    """A timed span. ``duration`` derives from start/end Unix timestamps.

    ``start`` and ``end`` are wall-clock seconds since the epoch (i.e.
    ``time.time()``). Downstream display can convert directly to a
    human-readable timestamp via ``datetime.fromtimestamp(span.start)``.
    """

    start: float = 0.0
    end: float = 0.0

    @computed_field
    @property
    def duration(self) -> float:
        if self.end <= 0.0:
            return 0.0
        return self.end - self.start


class TimeSpans(CustomBaseModel):
    """A list of ``TimeSpan``s. ``duration`` is the sum over children."""

    spans: list[TimeSpan] = Field(default_factory=list)

    @computed_field
    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.spans)


class RolloutTiming(CustomBaseModel):
    """Rollout-level timing. All values in seconds (Unix timestamps).

    Each measured phase (``setup``, ``generation``, ``scoring``) is a
    ``TimeSpan`` carrying wall-clock start/end timestamps. ``model`` and
    ``env`` are ``TimeSpans`` collections of the corresponding step slices
    (each appended in execution order).
    """

    start_time: float = Field(default_factory=time.time)

    setup: TimeSpan = Field(default_factory=TimeSpan)
    generation: TimeSpan = Field(default_factory=TimeSpan)
    scoring: TimeSpan = Field(default_factory=TimeSpan)
    model: TimeSpans = Field(default_factory=TimeSpans)
    env: TimeSpans = Field(default_factory=TimeSpans)

    @computed_field
    @property
    def total(self) -> float:
        if self.scoring.end <= 0.0:
            return 0.0
        return self.scoring.end - self.generation.start

    @computed_field
    @property
    def overhead(self) -> float:
        return (
            self.total
            - self.setup.duration
            - self.model.duration
            - self.env.duration
            - self.scoring.duration
        )


class ErrorInfo(TypedDict):
    error: str
    error_chain_repr: str
    error_chain_str: str


class RolloutOutput(dict):
    """Serialized output from a rollout (mirrors RolloutInput).

    A dict subclass that allows typed access to known fields while supporting
    arbitrary additional fields from state_columns. All values must be
    JSON-serializable.

    Required fields: example_id, prompt, completion, reward, timing,
                     is_completed, is_truncated, metrics
    Optional fields: answer, info, error, stop_condition, trajectory, tool_defs,
                     token_usage
    Additional fields: arbitrary serializable state_columns
    """

    # Required fields
    example_id: int
    prompt: Messages | None
    completion: Messages | None
    reward: float
    timing: RolloutTiming
    is_completed: bool
    is_truncated: bool
    metrics: dict[str, float]
    # Optional fields
    answer: str
    info: Info
    error: ErrorInfo | None
    stop_condition: str | None
    trajectory: list["TrajectoryStep"]
    tool_defs: list[Tool]
    token_usage: TokenUsage


_MISSING = object()
_DefaultValue = TypeVar("_DefaultValue")
_BorrowTarget = Literal["model", "sandbox"]
_ToolTarget = str | Iterable[str]
_TranscriptMode = Literal["private", "append"]


class StateForTaskDescriptor:
    def __get__(
        self, instance: "State | None", owner: type["State"]
    ) -> Callable[..., "State"]:
        def create(
            task: Mapping[str, Any],
            *,
            borrow: _BorrowTarget | Iterable[_BorrowTarget] = (),
            tools: _ToolTarget = (),
            transcript: _TranscriptMode = "private",
        ) -> "State":
            state = _state_for_task(owner, task, source_state=instance)
            if instance is not None:
                _borrow_from_state(state, instance, borrow, tools, transcript)
            elif borrow or tools:
                raise ValueError("State.for_task borrow/tools requires a source state.")
            elif transcript != "private":
                raise ValueError(
                    "State.for_task transcript='append' requires a source state."
                )
            return state

        return create


class State(dict):
    for_task = StateForTaskDescriptor()

    INPUT_FIELDS = ["prompt", "answer", "info", "example_id"]
    INTERNAL_KEYS = {"is_completed", "stop_condition", "is_truncated", "error"}
    RUNTIME_HANDLE_KEYS = {"runtime_id", "client_key"}
    ENDPOINT_HANDLE_KEYS = {
        "endpoint_rollout_key",
        "endpoint_root_url",
        "endpoint_base_url",
    }

    # rollout inputs
    input: RolloutInput
    task: dict[str, Any]
    client: "Client"
    model: str
    sampling_args: SamplingArgs | None
    # created during rollout
    is_completed: bool
    is_truncated: bool
    stop_condition: str | None
    tool_defs: list[Tool]
    trajectory: list[TrajectoryStep]
    completion: Messages | None
    reward: float | None
    advantage: float | None
    metrics: dict[str, float] | None
    timing: RolloutTiming | None
    error: "Error | ErrorInfo | None"
    usage: TokenUsage | None
    usage_tracker: object
    _vf_state_contract: Literal["legacy", "v1"]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._vf_state_contract = "legacy"

    @property
    def uses_v1_contract(self) -> bool:
        return self._vf_state_contract == "v1"

    def _enable_v1_contract(self) -> "State":
        self._vf_state_contract = "v1"
        return self

    def __getitem__(self, key: str) -> Any:
        # forward to input if exists
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if key in input_obj:
                return input_obj[key]
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: str, value: Any) -> None:
        if self.uses_v1_contract and key in self.INTERNAL_KEYS:
            raise RuntimeError(_internal_key_error(key))
        # forward to input if exists
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if key in input_obj:
                input_obj[key] = value
                return
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        if self.uses_v1_contract and key in self.INTERNAL_KEYS:
            raise RuntimeError(_internal_key_error(key))
        super().__delitem__(key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        if self.uses_v1_contract:
            for key, value in values.items():
                self[str(key)] = value
            return
        super().update(values)

    @overload
    def pop(self, key: str) -> Any: ...

    @overload
    def pop(self, key: str, default: _DefaultValue) -> Any | _DefaultValue: ...

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if self.uses_v1_contract and key in self.INTERNAL_KEYS:
            raise RuntimeError(_internal_key_error(key))
        if default is _MISSING:
            return super().pop(key)
        return super().pop(key, default)

    def popitem(self) -> tuple[str, Any]:
        if not self.uses_v1_contract:
            return super().popitem()
        for key in reversed(self.keys()):
            if key not in self.INTERNAL_KEYS:
                return key, super().pop(key)
        raise RuntimeError("State.popitem() cannot remove framework-managed fields.")

    def clear(self) -> None:
        if self.uses_v1_contract:
            raise RuntimeError(
                "State.clear() cannot preserve framework-managed fields."
            )
        super().clear()

    def setdefault(self, key: object, default: Any = None, /) -> Any:
        if self.uses_v1_contract and isinstance(key, str) and key in self.INTERNAL_KEYS:
            raise RuntimeError(_internal_key_error(key))
        return super().setdefault(key, default)

    def __ior__(self, other: object) -> "State":
        self.update(other)
        return self

    def _set_internal(self, key: str, value: Any) -> None:
        if key not in self.INTERNAL_KEYS:
            raise KeyError(f"{key!r} is not a framework-managed state key.")
        super().__setitem__(key, value)

    def _set_completed(self, value: bool = True) -> None:
        self._set_internal("is_completed", value)

    def _set_error(self, value: Any) -> None:
        self._set_internal("error", value)

    def _set_stop_condition(
        self, value: str | None, *, overwrite: bool = False
    ) -> None:
        if overwrite or self.get("stop_condition") is None:
            self._set_internal("stop_condition", value)

    def _set_truncated(self, value: bool = True, *, overwrite: bool = False) -> None:
        current = bool(self.get("is_truncated", False))
        self._set_internal(
            "is_truncated", bool(value) if overwrite else current or bool(value)
        )

    def stop(self, condition: str = "state_done") -> None:
        if not isinstance(condition, str) or not condition:
            raise TypeError("State.stop condition must be a non-empty string.")
        super().__setitem__("done", True)
        self._set_completed(True)
        self._set_stop_condition(condition, overwrite=True)

    def runtime_state(self) -> dict[str, Any]:
        raw_runtime = self.setdefault("runtime", {})
        if not isinstance(raw_runtime, dict):
            raise TypeError("state.runtime must be a mapping.")
        return cast(dict[str, Any], raw_runtime)

    def _runtime(self) -> Any:
        from verifiers.v1.utils.runtime_registry import load_runtime_from_state

        return load_runtime_from_state(self)

    def get_model(self) -> str:
        runtime = self.get("runtime", {})
        if isinstance(runtime, Mapping):
            model = runtime.get("model")
            if isinstance(model, str) and model:
                return model
            resolved = runtime.get("resolved")
            if isinstance(resolved, Mapping):
                handle = resolved.get("model")
                if isinstance(handle, Mapping):
                    model = handle.get("model")
                    if isinstance(model, str) and model:
                        return model
        try:
            return self._runtime().model(self)
        except RuntimeError as exc:
            raise RuntimeError("State has no resolved model.") from exc

    def get_max_turns(self, default: int) -> int:
        runtime = self.get("runtime", {})
        if isinstance(runtime, Mapping) and "max_turns" in runtime:
            value = runtime["max_turns"]
            if value is None:
                return default
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("state.runtime.max_turns must be an integer.")
            return value
        return default

    def get_client(
        self,
        api: EndpointApi | ClientType = "chat_completions",
        *,
        sync: bool = False,
    ) -> object:
        from verifiers.v1.utils.endpoint_utils import client_from_state

        return client_from_state(self, api, sync=sync)

    def get_endpoint_config(
        self,
        api: EndpointApi | ClientType = "chat_completions",
    ) -> dict[str, str]:
        from verifiers.v1.utils.endpoint_utils import endpoint_config_from_state

        return endpoint_config_from_state(self, api)

    def get_tools(self) -> dict[str, Callable[..., Any]]:
        from verifiers.v1.utils.tool_utils import load_tools_from_state

        return load_tools_from_state(self)

    def _runtime_handles(self) -> dict[str, Any]:
        runtime = self.runtime_state()
        handles = runtime.setdefault("resolved", {})
        if not isinstance(handles, dict):
            raise TypeError("state.runtime.resolved must be a mapping.")
        return handles

    def _runtime_handle(self, name: str) -> dict[str, Any]:
        runtime = self.runtime_state()
        handles = runtime.get("resolved")
        if handles is not None:
            if not isinstance(handles, Mapping):
                raise TypeError("state.runtime.resolved must be a mapping.")
            existing = handles.get(name)
            if existing is not None:
                if not isinstance(existing, Mapping):
                    raise TypeError(f"state.runtime.resolved.{name} must be a mapping.")
                return dict(existing)

        runtime_id = runtime.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            raise RuntimeError("State has no live runtime id.")
        if name == "model":
            client_key = runtime.get("client_key")
            if not isinstance(client_key, str) or not client_key:
                raise RuntimeError("State has no resolved model client.")
            handle: dict[str, Any] = {
                "runtime_id": runtime_id,
                "client_key": client_key,
            }
            for key in ("model", "client_type", "sampling_args"):
                if key in runtime:
                    handle[key] = runtime[key]
            return handle
        if name == "endpoint":
            return {"runtime_id": runtime_id}
        if name == "trajectory":
            runtime_obj = self._runtime()
            runtime_obj.register_trajectory(self)
            trajectory = self.get("trajectory") or []
            if not isinstance(trajectory, list):
                raise TypeError("state.trajectory must be a list.")
            return {
                "runtime_id": runtime_id,
                "trajectory_id": str(self["trajectory_id"]),
                "start": len(trajectory),
            }
        if name == "sandbox":
            sandbox = runtime.get("sandbox")
            if not isinstance(sandbox, Mapping):
                raise RuntimeError("State has no resolved primary sandbox.")
            handle = dict(sandbox)
            handle["runtime_id"] = runtime_id
            return handle
        raise KeyError(f"Unknown runtime handle {name!r}.")

    def _tools_handle(self, names: _ToolTarget) -> dict[str, Any] | None:
        tool_names = tuple(_tool_names(names))
        if not tool_names:
            return None
        runtime = self._runtime()
        handle_id = runtime.register_tool_handle(self, tool_names)
        return {
            "runtime_id": runtime.runtime_id,
            "handle_id": handle_id,
            "names": list(tool_names),
        }

    def _use_runtime_handle(self, name: str, handle: Mapping[str, Any]) -> "State":
        self._runtime_handles()[name] = dict(handle)
        return self

    def strip_runtime_handles(self) -> None:
        _strip_runtime_handles(self)

    def finalize(self) -> "State":
        self.strip_runtime_handles()
        self.assert_serializable()
        return self

    @classmethod
    def _legacy_for_task(cls, task: Mapping[str, Any]) -> "State":
        state = cls(
            {
                "task": dict(task),
                "runtime": dict(task.get("runtime", {})),
                "trajectory": [],
                "trajectory_id": uuid.uuid4().hex,
                "artifacts": {},
                "metrics": {},
                "reward": 0.0,
                "is_completed": False,
                "is_truncated": False,
                "stop_condition": None,
                "completion": None,
                "error": None,
                "timing": {
                    "generation_ms": 0.0,
                    "scoring_ms": 0.0,
                    "total_ms": 0.0,
                    "start_time": time.time(),
                },
            }
        )
        for key in ("prompt", "answer", "info", "example_id"):
            if key in task:
                state[key] = deepcopy(task[key])
        return state

    def assert_serializable(self) -> None:
        assert_json_serializable(self)


def _internal_key_error(key: str) -> str:
    if key == "is_completed":
        return (
            "state['is_completed'] is framework-managed; use state.stop(...), "
            "state['done'], or @vf.stop."
        )
    if key == "stop_condition":
        return (
            "state['stop_condition'] is framework-managed; use state.stop(...), "
            "state['done'], or @vf.stop."
        )
    if key == "is_truncated":
        return (
            "state['is_truncated'] is framework-managed; raise an overlong-prompt "
            "error or let trajectory sync set it."
        )
    if key == "error":
        return "state['error'] is framework-managed; raise vf.Error instead."
    return f"state[{key!r}] is framework-managed."


def _state_for_task(
    cls: type[State], task: Mapping[str, Any], source_state: State | None = None
) -> State:
    if _uses_v1_contract(task, source_state):
        return _v1_state_for_task(cls, task)
    return cls._legacy_for_task(task)


def _uses_v1_contract(task: Mapping[str, Any], source_state: State | None) -> bool:
    if source_state is not None and source_state.uses_v1_contract:
        return True
    return getattr(task, "_vf_state_contract", "legacy") == "v1"


def _v1_state_for_task(cls: type[State], task: Mapping[str, Any]) -> State:
    from verifiers.v1.utils.timing_utils import timing_record

    state = cls(
        {
            "task": dict(task),
            "runtime": {},
            "trajectory": [],
            "trajectory_id": uuid.uuid4().hex,
            "artifacts": {},
            "metrics": {},
            "reward": 0.0,
            "completion": None,
            "timing": timing_record(),
        }
    )._enable_v1_contract()
    state._set_completed(False)
    state._set_truncated(False, overwrite=True)
    state._set_stop_condition(None, overwrite=True)
    state._set_error(None)
    for key in ("prompt", "info", "example_id"):
        if key in task:
            state[key] = deepcopy(task[key])
    return state


def _borrow_from_state(
    state: State,
    source: State,
    borrow: _BorrowTarget | Iterable[_BorrowTarget],
    tools: _ToolTarget,
    transcript: _TranscriptMode,
) -> None:
    if transcript not in {"private", "append"}:
        raise ValueError("transcript must be 'private' or 'append'.")
    for name in _borrow_targets(borrow):
        if name not in {"model", "sandbox"}:
            raise KeyError(f"Unknown borrow target {name!r}.")
        state._use_runtime_handle(name, source._runtime_handle(name))
    tools_handle = source._tools_handle(tools)
    if tools_handle is not None:
        state._use_runtime_handle("tools", tools_handle)
    if transcript == "append":
        state._use_runtime_handle("trajectory", source._runtime_handle("trajectory"))


def _borrow_targets(
    borrow: _BorrowTarget | Iterable[_BorrowTarget],
) -> Iterable[_BorrowTarget]:
    if isinstance(borrow, str):
        return (cast(_BorrowTarget, borrow),)
    return borrow


def _tool_names(tools: _ToolTarget) -> Iterable[str]:
    if isinstance(tools, str):
        return (tools,)
    return tools


def _strip_runtime_handles(value: object) -> None:
    if isinstance(value, State) or type(value) is dict:
        mapping = cast(dict[str, Any], value)
        for key in State.RUNTIME_HANDLE_KEYS:
            mapping.pop(key, None)
        runtime = mapping.get("runtime")
        if type(runtime) is dict:
            runtime_mapping = cast(dict[str, Any], runtime)
            runtime_mapping.pop("resolved", None)
            for key in State.RUNTIME_HANDLE_KEYS:
                runtime_mapping.pop(key, None)
            sandbox = runtime_mapping.get("sandbox")
            if type(sandbox) is dict:
                cast(dict[str, Any], sandbox).pop("lease_key", None)
        for key in State.ENDPOINT_HANDLE_KEYS:
            mapping.pop(key, None)
        for item in list(mapping.values()):
            _strip_runtime_handles(item)
        return
    if isinstance(value, list):
        for item in value:
            _strip_runtime_handles(item)


def assert_json_serializable(value: object) -> None:
    try:
        json.dumps(value)
    except TypeError as e:
        raise TypeError("Task and State values must be JSON-serializable.") from e


TASK_INPUT_FIELDS = {"prompt", "answer", "info", "example_id"}


def normalize_task_payload(value: object) -> dict[str, Any]:
    """Normalize a serialized task payload attached to rollout info."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(
                "Serialized task payloads must be JSON objects. Plain string task "
                "routes are no longer supported; use info['env_id'] for routing."
            ) from e
    if not isinstance(value, Mapping):
        raise TypeError("Serialized task payloads must decode to a mapping.")
    return dict(cast(Mapping[str, Any], value))


def task_payload_from_info(info: object) -> dict[str, Any] | None:
    """Return the canonical task payload from info.task if one is present."""
    if isinstance(info, str):
        info = json.loads(info)
    if not isinstance(info, Mapping):
        return None
    task_payload = cast(Mapping[str, Any], info).get("task")
    if task_payload is None:
        return None
    return normalize_task_payload(task_payload)


def flatten_task_input(input_data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical task payload for a rollout input."""
    task_payload = task_payload_from_info(input_data.get("info"))
    if task_payload is not None:
        return task_payload
    direct_task_payload = input_data.get("task")
    if direct_task_payload is not None:
        return normalize_task_payload(direct_task_payload)
    return dict(input_data)


# oai tools
JsonPrimitive = Literal["string", "number", "integer", "boolean", "array", "object"]

# callbacks
StartCallback = Callable[
    [list[RolloutInput], list[RolloutInput] | list[list[RolloutInput]]], None
]
ProgressCallback = Callable[
    [list[RolloutOutput], list[RolloutOutput], "GenerateMetadata"], None
]  # all_outputs, new_outputs, new_metadata
LogCallback = Callable[[str], None]  # log messages


class GenerateMetadata(TypedDict):
    """Pydantic model for generation metadata."""

    env_id: str
    name: NotRequired[str]
    env_args: dict
    model: str
    base_url: str
    num_examples: int
    rollouts_per_example: int
    sampling_args: SamplingArgs
    date: str
    time: float  # whole-eval wall-clock seconds
    avg_reward: float
    avg_metrics: dict[str, float]
    avg_error: float
    pass_at_k: dict[str, float]
    pass_all_k: dict[str, float]
    pass_threshold: float
    usage: TokenUsage | None
    cost: NotRequired[EvalCost]
    version_info: VersionInfo
    state_columns: list[str]
    path_to_save: Path
    tools: list[Tool] | None


class GenerateOutputs(TypedDict):
    """TypedDict for generation outputs (results)."""

    outputs: list[RolloutOutput]
    metadata: GenerateMetadata


class RolloutScore(TypedDict):
    """TypedDict for rollout scores."""

    reward: float
    metrics: dict[str, float]


class RolloutScores(TypedDict):
    """TypedDict for rubric outputs."""

    reward: list[float]
    metrics: dict[str, list[float]]


Endpoint = TypedDict(
    "Endpoint",
    {
        "key": str,
        "url": str,
        "model": str,
        "api_client_type": NotRequired[ClientType],
        "extra_headers": NotRequired[dict[str, str]],
    },
)
Endpoints = dict[str, list[Endpoint]]


def _validate_extra_headers_value(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("extra_headers must be a dict")
    out: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("extra_headers keys must be non-empty strings")
        if not isinstance(v, str):
            raise ValueError("extra_headers values must be strings")
        out[k] = v
    return out


class ClientConfig(BaseModel):
    """Pydantic model for client configuration."""

    client_idx: int = 0
    client_type: ClientType = "openai_chat_completions"
    renderer: str = "auto"
    renderer_model_name: str | None = None
    renderer_pool_size: int | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    preserve_all_thinking: bool = False
    preserve_thinking_between_tool_calls: bool = False
    api_key_var: str = "PRIME_API_KEY"
    api_base_url: str = "https://api.pinference.ai/api/v1"
    endpoint_configs: list["EndpointClientConfig"] = Field(default_factory=list)
    timeout: float = 3600.0
    connect_timeout: float = 5.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_headers_from_state: dict[str, str] = Field(
        default_factory=dict,
        description="Maps HTTP header names to state field names. "
        "For each request, the header value is read from the state dict. "
        'e.g. {"X-Session-ID": "example_id"} adds a X-Session-ID header '
        "with the value of state['example_id'].",
    )

    @field_validator("extra_headers", mode="before")
    @classmethod
    def validate_extra_headers(cls, value: object) -> dict[str, str]:
        return _validate_extra_headers_value(value)

    @field_validator("endpoint_configs", mode="before")
    @classmethod
    def validate_non_recursive_endpoints(cls, value):
        if not isinstance(value, list):
            return value

        normalized_endpoints = []
        for endpoint in value:
            if isinstance(endpoint, ClientConfig):
                if endpoint.endpoint_configs:
                    raise ValueError(
                        "ClientConfig.endpoint_configs entries cannot include endpoint_configs"
                    )
                normalized_endpoints.append(
                    endpoint.model_dump(
                        mode="python",
                        exclude={"endpoint_configs"},
                        exclude_unset=True,
                    )
                )
                continue

            if (
                isinstance(endpoint, dict)
                and "endpoint_configs" in endpoint
                and endpoint["endpoint_configs"]
            ):
                raise ValueError(
                    "ClientConfig.endpoint_configs entries cannot include endpoint_configs"
                )

            nested = getattr(endpoint, "endpoint_configs", None)
            if nested:
                raise ValueError(
                    "ClientConfig.endpoint_configs entries cannot include endpoint_configs"
                )

            normalized_endpoints.append(endpoint)

        return normalized_endpoints


class EndpointClientConfig(BaseModel):
    """Leaf endpoint config used inside ClientConfig.endpoint_configs."""

    client_idx: int = 0
    api_key_var: str = "PRIME_API_KEY"
    api_base_url: str = "https://api.pinference.ai/api/v1"
    timeout: float = 3600.0
    connect_timeout: float = 5.0
    max_connections: int = 28000
    max_keepalive_connections: int = 28000
    max_retries: int = 10
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_headers", mode="before")
    @classmethod
    def validate_extra_headers(cls, value: object) -> dict[str, str]:
        return _validate_extra_headers_value(value)


ClientConfig.model_rebuild()


class EvalConfig(BaseModel):
    """Pydantic model for evaluation configuration."""

    # environment
    env_id: str
    name: str | None = None
    env_args: dict
    env_dir_path: str
    # evaluation
    endpoint_id: str | None = None
    model: str
    client_config: ClientConfig
    sampling_args: SamplingArgs
    num_examples: int
    rollouts_per_example: int
    max_concurrent: int
    num_workers: int | str = "auto"
    independent_scoring: bool = False
    extra_env_kwargs: dict = {}
    max_retries: int = 0
    disable_env_server: bool = False
    # logging
    verbose: bool = False
    disable_tui: bool = False
    # saving
    output_dir: str | None = None
    state_columns: list[str] | None = None
    save_results: bool = False
    resume_path: Path | None = None
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None


class EvalRunConfig(BaseModel):
    """Pydantic model for evaluation run configuration."""

    evals: list[EvalConfig]
    heartbeat_url: str | None = None
