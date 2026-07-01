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

from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)
from renderers import RendererConfig

from verifiers.errors import Error

if TYPE_CHECKING:
    from anthropic.types import RedactedThinkingBlock
    from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
    from datasets import Dataset

    from verifiers.clients import Client
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
EndpointClient: TypeAlias = AsyncOpenAI | OpenAI | AsyncAnthropic | Anthropic
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


ToolLike: TypeAlias = str | Tool | Callable[..., object]


class Usage(CustomBaseModel):
    prompt_tokens: int
    reasoning_tokens: int
    completion_tokens: int
    total_tokens: int


class RoutedExpertsPayload(TypedDict):
    # Keep the raw response sidecar opaque so Pydantic does not validate memoryview.
    data: Any
    shape: list[int]
    start: int


class ResponseTokens(CustomBaseModel):
    prompt_ids: list[int]
    prompt_mask: list[int]
    completion_ids: list[int]
    completion_mask: list[int]
    completion_logprobs: list[float]
    routed_experts: RoutedExpertsPayload | None = None
    # Renderer-emitted multimodal sidecar (renderers.base.MultiModalData)
    # carrying processed pixel_values / placeholder ranges per modality.
    # Populated by the renderer client when the rollout went through a
    # multimodal-aware renderer; ``None`` otherwise. Stored as ``Any`` to
    # avoid a hard import dependency on ``renderers`` at this layer.
    multi_modal_data: Any | None = None
    # Renderer-emitted per-token prompt attribution
    # (``renderers.base.RenderedTokens``); ``None`` for non-renderer
    # clients. ``Any`` for the same reason as ``multi_modal_data``.
    prompt_attribution: Any | None = None


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
    routed_experts: RoutedExpertsPayload | None
    # Renderer-emitted multimodal sidecar (renderers.base.MultiModalData)
    # carrying processed pixel_values / placeholder ranges per modality.
    # ``NotRequired`` because text-only rollouts (and non-renderer client
    # types) never populate it.
    multi_modal_data: NotRequired[Any]
    # ``RenderedTokens`` as dict (rehydrate via ``RenderedTokens(**d)``);
    # only ``RendererClient`` rollouts populate it.
    prompt_attribution: NotRequired[Any]


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


class ErrorData(TypedDict):
    error: str
    message: str
    error_chain_repr: str
    error_chain_str: str


class JudgeRecord(TypedDict, total=False):
    """One LLM-as-judge call captured during rollout scoring.

    Recorded by ``JudgeRubric.judge`` so the platform can render the exact
    input/output of each judge invocation in the rollout view. Fields are all
    optional except ``judge_input`` and ``judge_output`` so older callers can
    omit metadata without breaking serialization.
    """

    judge_input: list[dict[str, Any]] | str
    judge_output: str
    rubric: str
    model: str
    score: float | None
    timestamp: float


class RolloutOutput(dict):
    """Serialized output from a rollout (mirrors RolloutInput).

    A dict subclass that allows typed access to known fields while supporting
    arbitrary additional fields from state_columns. All values must be
    JSON-serializable.

    Required fields: example_id, prompt, completion, reward, timing,
                     is_completed, is_truncated, metrics
    Optional fields: answer, info, error, stop_condition, trajectory, tool_defs,
                     token_usage, judges
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
    error: ErrorData | None
    stop_condition: str | None
    trajectory: list["TrajectoryStep"]
    tool_defs: list[Tool]
    token_usage: TokenUsage
    judges: list[JudgeRecord]


_MISSING = object()
_DefaultValue = TypeVar("_DefaultValue")
_BorrowTarget = Literal["model", "sandbox"]
_ToolTarget = str | Iterable[str]
_TranscriptMode = Literal["private", "append"]


class StateForTaskDescriptor:
    def __get__(
        self, instance: "State | None", owner: type["State"]
    ) -> Callable[..., "State"]:
        def create(task: Mapping[str, Any]) -> "State":
            return owner._legacy_for_task(task)

        return create


class State(dict):
    for_task = StateForTaskDescriptor()

    INPUT_FIELDS = ["prompt", "answer", "info", "example_id"]
    INTERNAL_KEYS = {"is_completed", "stop_condition", "is_truncated", "error"}

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
    error: Error | None
    usage: TokenUsage | None
    usage_tracker: object

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
        # forward to input if exists
        if key in self.INPUT_FIELDS and "input" in self:
            input_obj = super().__getitem__("input")
            if key in input_obj:
                input_obj[key] = value
                return
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(dict(*args, **kwargs))

    @overload
    def pop(self, key: str) -> Any: ...

    @overload
    def pop(self, key: str, default: _DefaultValue) -> Any | _DefaultValue: ...

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        if default is _MISSING:
            return super().pop(key)
        return super().pop(key, default)

    def popitem(self) -> tuple[str, Any]:
        return super().popitem()

    def clear(self) -> None:
        super().clear()

    def setdefault(self, key: object, default: Any = None, /) -> Any:
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

    def _set_error(self, value: Error | None) -> None:
        if value is not None and not isinstance(value, Error):
            raise TypeError("state.error must be a vf.Error or None.")
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

    def add_step_reward(self, reward: float | int | None) -> None:
        if reward is None:
            return
        if isinstance(reward, bool):
            raise TypeError("State.add_step_reward requires a numeric reward.")
        trajectory = self["trajectory"]
        if not isinstance(trajectory, list):
            raise TypeError("state.trajectory must be a list.")
        if not trajectory:
            raise RuntimeError("State.add_step_reward requires a trajectory step.")
        step = trajectory[-1]
        if not isinstance(step, dict):
            raise TypeError("trajectory steps must be mappings.")
        current = step.get("reward", 0.0) or 0.0
        if isinstance(current, bool) or not isinstance(current, int | float):
            raise TypeError("trajectory step reward must be numeric.")
        step["reward"] = float(current) + float(reward)

    def total_step_reward(self) -> float:
        trajectory = self.get("trajectory") or []
        if not isinstance(trajectory, list):
            raise TypeError("state.trajectory must be a list.")
        total = 0.0
        for step in trajectory:
            if not isinstance(step, dict):
                raise TypeError("trajectory steps must be mappings.")
            reward = step.get("reward", 0.0)
            if reward is None:
                continue
            if isinstance(reward, bool) or not isinstance(reward, int | float):
                raise TypeError("trajectory step reward must be numeric.")
            total += float(reward)
        return total

    def ensure_timing(self) -> dict[str, Any]:
        timing = self.setdefault("timing", _timing_record())
        if not isinstance(timing, dict):
            raise TypeError("state.timing must be a mapping.")
        timing = cast(dict[str, Any], timing)
        if "generation_ms" in timing or "total_ms" in timing:
            start = _float_value(timing.get("start_time"), time.time())
            elapsed = (
                _float_value(timing.get("total_ms", timing.get("generation_ms", 0.0)))
                / 1000
            )
            timing.clear()
            timing.update(_timing_record(start))
            if elapsed > 0.0:
                _set_timing_span(timing, "generation", start, start + elapsed)
                _set_timing_total(timing, start + elapsed)
        return timing

    def record_generation_timing(self) -> None:
        timing = self.ensure_timing()
        start_time = _float_value(timing.get("start_time"), time.time())
        end_time = time.time()
        _set_timing_span(timing, "generation", start_time, end_time)
        _set_timing_total(timing, end_time)

    def record_scoring_timing(self, start_time: float) -> None:
        timing = self.ensure_timing()
        end_time = time.time()
        _set_timing_span(timing, "scoring", start_time, end_time)
        _set_timing_total(timing, end_time)

    def record_model_timing(self, start_time: float, end_time: float) -> None:
        timing = self.ensure_timing()
        spans = timing.setdefault("model", _timing_spans_record())
        if not isinstance(spans, dict):
            raise TypeError("state.timing.model must be a mapping.")
        spans = cast(dict[str, Any], spans)
        span_list = spans.setdefault("spans", [])
        if not isinstance(span_list, list):
            raise TypeError("state.timing.model.spans must be a list.")
        span_list.append(_timing_span_record(start_time, end_time))
        spans["duration"] = _timing_duration(spans) + max(0.0, end_time - start_time)

    def finalize(self) -> "State":
        self.serialize_error()
        self.assert_serializable()
        return self

    def serialize_error(self) -> None:
        error = self.get("error")
        if isinstance(error, Error):
            from verifiers.utils.error_utils import error_data

            self._set_internal("error", error_data(error))

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


def _timing_span_record(start: float = 0.0, end: float = 0.0) -> dict[str, float]:
    return {
        "start": start,
        "end": end,
        "duration": max(0.0, end - start) if end > 0.0 else 0.0,
    }


def _timing_spans_record() -> dict[str, Any]:
    return {"spans": [], "duration": 0.0}


def _timing_record(start_time: float | None = None) -> dict[str, Any]:
    start = time.time() if start_time is None else start_time
    return {
        "start_time": start,
        "setup": _timing_span_record(),
        "generation": _timing_span_record(start=start),
        "scoring": _timing_span_record(),
        "model": _timing_spans_record(),
        "env": _timing_spans_record(),
        "total": 0.0,
        "overhead": 0.0,
    }


def _set_timing_span(
    timing: dict[str, Any], key: str, start_time: float, end_time: float
) -> None:
    timing[key] = _timing_span_record(start_time, end_time)


def _set_timing_total(timing: dict[str, Any], end_time: float) -> None:
    start_time = _float_value(timing.get("start_time"), end_time)
    total = max(0.0, end_time - start_time)
    timing["total"] = total
    timing["overhead"] = max(
        0.0,
        total
        - _timing_duration(timing.get("setup", {}))
        - _timing_duration(timing.get("model", {}))
        - _timing_duration(timing.get("env", {}))
        - _timing_duration(timing.get("scoring", {})),
    )


def _timing_duration(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    value = cast(dict[str, Any], value)
    return _float_value(value.get("duration", 0.0), 0.0)


def _float_value(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return default
    return float(value or 0.0)


def _json_gate_default(obj: object) -> object:
    # Gate parity with the trainer transport (serve_utils.msgpack_encoder):
    # accept what msgpack ships (it reaches the trainer via msgpack, not JSON),
    # as cheap stand-ins; json re-invokes this on nested values. Unknown raises.
    import dataclasses
    from datetime import date, datetime
    from enum import Enum

    if isinstance(obj, (Path, uuid.UUID)):
        return str(obj)  # filesystem path / uuid
    if isinstance(obj, Enum):
        return obj.value  # enum member
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()  # timestamps
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<{type(obj).__name__}>"  # raw buffers (e.g. routed_experts data)
    np = sys.modules.get("numpy")
    if np is not None and isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # numpy scalar -> python scalar
    if np is not None and isinstance(obj, np.ndarray):
        return {"dtype": str(obj.dtype), "shape": list(obj.shape)}  # array: shape only
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(obj, torch.Tensor):
        return {"dtype": str(obj.dtype), "shape": list(obj.shape)}  # tensor: shape only
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)  # pydantic model
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # renderer sidecars (MultiModalData / PlaceholderRange); shallow so json
        # recurses into fields without deep-copying pixel arrays.
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")


def assert_json_serializable(value: object) -> None:
    try:
        json.dumps(value, default=_json_gate_default)
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
    shuffle: NotRequired[bool]
    shuffle_seed: NotRequired[int | None]
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


class EndpointConfig(BaseModel):
    """Endpoint connection config with credentials carried by env-var reference."""

    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str
    api_key_var: str
    api_client_type: ClientType | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_headers", mode="before")
    @classmethod
    def validate_extra_headers(cls, value: object) -> dict[str, str]:
        return _validate_extra_headers_value(value)


Endpoints = dict[str, list[EndpointConfig]]


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
    renderer_config: RendererConfig | None = None
    """Typed renderer config (one of ``renderers.RendererConfig``'s variants).
    Drives the renderer pool when ``client_type == "renderer"``. Defaults
    to ``None`` so non-renderer clients aren't forced to declare it; the
    renderer client treats ``None`` as ``AutoRendererConfig()``."""
    renderer_model_name: str | None = None
    """Override the tokenizer model name used to instantiate the renderer
    pool. Defaults to the model used in API requests."""
    renderer_pool_size: int | None = None
    """Size of the shared renderer pool. ``None`` falls back to the
    ``RendererClient`` default."""
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
    env_id: str = Field(validation_alias=AliasChoices("env_id", "id"))
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
    shuffle: bool = False
    shuffle_seed: int | None = None
    max_concurrent: int
    num_workers: int | str = "auto"
    independent_scoring: bool = False
    extra_env_kwargs: dict = {}
    max_retries: int = 3
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
