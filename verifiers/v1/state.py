from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from pydantic import (
    BaseModel,
    computed_field,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

from verifiers.types import (
    ErrorData,
    FinishReason,
    Messages,
    Response,
    ResponseTokens,
    ToolCall,
    ToolMessage,
    Usage,
)
from verifiers.utils.error_utils import error_data, validate_error_data
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.response_utils import parse_response_message
from verifiers.utils.save_utils import serialize_messages_for_output

from .types import JsonData, ModelConfig
from .utils.json_utils import jsonable
from .utils.task_freeze_utils import assert_serializable

if TYPE_CHECKING:
    from .task import Task


class TimeSpan(BaseModel, extra="forbid"):
    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start) if self.end else 0.0

    def begin(self) -> None:
        self.start = time.time()

    def finish(self) -> None:
        self.end = time.time()


class Timing(BaseModel, extra="forbid"):
    start_time: float = Field(default_factory=time.time)
    setup: TimeSpan = Field(default_factory=TimeSpan)
    generation: TimeSpan = Field(default_factory=TimeSpan)
    scoring: TimeSpan = Field(default_factory=TimeSpan)
    cleanup: TimeSpan = Field(default_factory=TimeSpan)
    model: list[TimeSpan] = Field(default_factory=list)
    runtime: list[TimeSpan] = Field(default_factory=list)

    @property
    def total(self) -> float:
        end = max(
            self.setup.end,
            self.generation.end,
            self.scoring.end,
            self.cleanup.end,
            self.start_time,
        )
        return max(0.0, end - self.start_time)


class TurnUsage(BaseModel, extra="forbid"):
    prompt_tokens: StrictInt = 0
    reasoning_tokens: StrictInt = 0
    completion_tokens: StrictInt = 0
    total_tokens: StrictInt = 0

    @classmethod
    def from_usage(cls, usage: Usage | None) -> "TurnUsage | None":
        if usage is None:
            return None
        return cls(
            prompt_tokens=usage.prompt_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )


class TurnTokens(BaseModel, extra="forbid"):
    prompt_ids: list[StrictInt] = Field(default_factory=list)
    prompt_mask: list[StrictInt] = Field(default_factory=list)
    prompt_advantages: list[StrictFloat] | None = None
    completion_ids: list[StrictInt] = Field(default_factory=list)
    completion_mask: list[StrictInt] = Field(default_factory=list)
    completion_logprobs: list[StrictFloat] = Field(default_factory=list)
    completion_advantages: list[StrictFloat] | None = None
    overlong_prompt: StrictBool = False
    is_truncated: StrictBool = False

    @classmethod
    def from_response(
        cls, tokens: ResponseTokens | None, *, is_truncated: bool = False
    ) -> "TurnTokens | None":
        if tokens is None:
            return None
        return cls(
            prompt_ids=list(tokens.prompt_ids),
            prompt_mask=list(tokens.prompt_mask),
            completion_ids=list(tokens.completion_ids),
            completion_mask=list(tokens.completion_mask),
            completion_logprobs=list(tokens.completion_logprobs),
            is_truncated=is_truncated,
        )

    @model_validator(mode="after")
    def validate_lengths(self) -> "TurnTokens":
        if len(self.prompt_ids) != len(self.prompt_mask):
            raise ValueError("TurnTokens prompt_ids and prompt_mask lengths differ.")
        if self.prompt_advantages is not None and len(self.prompt_ids) != len(
            self.prompt_advantages
        ):
            raise ValueError(
                "TurnTokens prompt_ids and prompt_advantages lengths differ."
            )
        if len(self.completion_ids) != len(self.completion_mask):
            raise ValueError(
                "TurnTokens completion_ids and completion_mask lengths differ."
            )
        if len(self.completion_ids) != len(self.completion_logprobs):
            raise ValueError(
                "TurnTokens completion_ids and completion_logprobs lengths differ."
            )
        if self.completion_advantages is not None and len(self.completion_ids) != len(
            self.completion_advantages
        ):
            raise ValueError(
                "TurnTokens completion_ids and completion_advantages lengths differ."
            )
        return self


class Turn(BaseModel, extra="forbid"):
    """One model request/response boundary in a rollout transcript."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    prompt: Messages
    completion: Messages = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolMessage] = Field(default_factory=list)
    response_id: str | None = None
    model: str | None = None
    created: StrictInt | None = None
    finish_reason: FinishReason = None
    usage: TurnUsage | None = None
    tokens: TurnTokens | None = None
    reward: float | None = None
    is_truncated: bool = False
    timing: TimeSpan = Field(default_factory=TimeSpan)

    @model_validator(mode="after")
    def normalize_turn_messages(self) -> "Turn":
        self.prompt = normalize_messages(self.prompt, field_name="turn.prompt")
        self.completion = normalize_messages(
            self.completion, field_name="turn.completion"
        )
        return self


class State(BaseModel, extra="forbid"):
    """Strict serializable v1 rollout state."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str | None = None
    transcript: list[Turn] = Field(default_factory=list)
    scratch: JsonData = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    reward: float = 0.0
    advantage: float | None = None
    artifacts: JsonData = Field(default_factory=dict)
    usage: dict[str, float] = Field(default_factory=dict)
    timing: Timing = Field(default_factory=Timing)
    is_completed: bool = False
    is_truncated: bool = False
    stop_condition: str | None = None
    error: ErrorData | None = None
    group_id: str | None = None
    metadata: JsonData = Field(default_factory=dict)
    model: ModelConfig | None = None
    teacher: ModelConfig | None = None

    @computed_field
    @property
    def prompt(self) -> Messages:
        if not self.transcript:
            return []
        return self.transcript[-1].prompt

    @computed_field
    @property
    def completion(self) -> Messages:
        if not self.transcript:
            return []
        return self.transcript[-1].completion

    @computed_field
    @property
    def messages(self) -> Messages:
        messages: Messages = []
        for turn in self.transcript:
            messages.extend(turn.prompt)
            messages.extend(turn.completion)
        return messages

    def stop(self, condition: str = "state_done") -> None:
        if not condition:
            raise ValueError("State.stop condition must be non-empty.")
        self.is_completed = True
        self.stop_condition = condition

    def capture_error(self, error: BaseException) -> None:
        self.error = error_data(error)
        self.stop("has_error")

    def add_turn(self, turn: Turn) -> None:
        self.transcript.append(turn)
        if turn.is_truncated:
            self.is_truncated = True

    async def add_response_turn(
        self,
        prompt: Messages,
        response: Response,
        *,
        start: float | None = None,
        end: float | None = None,
    ) -> Turn:
        completion = await parse_response_message(response)
        turn = Turn(
            prompt=prompt,
            completion=completion,
            tool_calls=list(response.message.tool_calls or []),
            response_id=response.id,
            model=response.model,
            created=response.created,
            finish_reason=response.message.finish_reason,
            usage=TurnUsage.from_usage(response.usage),
            tokens=TurnTokens.from_response(
                response.message.tokens,
                is_truncated=bool(response.message.is_truncated),
            ),
            is_truncated=bool(response.message.is_truncated),
            timing=TimeSpan(start=start or 0.0, end=end or 0.0),
        )
        self.add_turn(turn)
        self._record_usage(response.usage)
        return turn

    def add_step_reward(self, reward: float | int | None) -> None:
        if reward is None:
            return
        if isinstance(reward, bool) or not isinstance(reward, int | float):
            raise TypeError("State.add_step_reward requires a numeric reward.")
        if not self.transcript:
            raise RuntimeError("State.add_step_reward requires a transcript turn.")
        turn = self.transcript[-1]
        turn.reward = float(turn.reward or 0.0) + float(reward)

    def total_step_reward(self) -> float:
        return sum(float(turn.reward or 0.0) for turn in self.transcript)

    def finalize(self) -> "State":
        self.assert_serializable()
        return self

    def assert_serializable(self) -> None:
        assert_serializable(self.model_dump(mode="json", exclude_none=True))

    def to_output(
        self, task: "Task", state_columns: list[str] | None = None
    ) -> dict[str, object]:
        prompt = self.prompt or task.prompt
        output: dict[str, object] = {
            "example_id": task.row_id,
            "prompt": self._serialized_messages(prompt),
            "completion": self._serialized_messages(self.completion),
            "reward": float(self.reward),
            "timing": self.timing.model_dump(mode="json"),
            "is_completed": self.is_completed,
            "is_truncated": self.is_truncated,
            "metrics": dict(self.metrics),
            "stop_condition": self.stop_condition,
            "transcript": [self.turn_record(turn) for turn in self.transcript],
        }
        answer = getattr(task, "answer", None)
        if answer is not None:
            output["answer"] = str(answer)
        info = getattr(task, "info", None)
        if isinstance(info, dict) and info:
            output["info"] = cast(dict[str, object], deepcopy(info))
        if self.error is not None:
            output["error"] = validate_error_data(self.error)
            output["error_chain"] = self.error["error_chain_repr"]
            output["long_error_chain"] = self.error["error_chain_str"]
        if self.usage:
            output["token_usage"] = {
                "input_tokens": float(self.usage.get("input_tokens", 0.0)),
                "output_tokens": float(self.usage.get("output_tokens", 0.0)),
            }
        if self.advantage is not None:
            output["advantage"] = self.advantage
        for key, value in self.metrics.items():
            output[key] = value
        for column in state_columns or []:
            output[column] = self.column_value(column)
        json.dumps(output)
        return output

    def column_value(self, column: str) -> object:
        if column == "prompt":
            return self._serialized_messages(self.prompt)
        if column == "completion":
            return self._serialized_messages(self.completion)
        if column == "messages":
            return self._serialized_messages(self.messages)
        if column == "scratch":
            return deepcopy(self.scratch)
        if column == "transcript":
            return [self.turn_record(turn) for turn in self.transcript]
        if column in type(self).model_fields:
            value = getattr(self, column)
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                return model_dump(mode="json", exclude_none=True)
            return deepcopy(value)
        if column in self.scratch:
            return deepcopy(self.scratch[column])
        return None

    def _record_usage(self, usage: Usage | None) -> None:
        if usage is None:
            return
        self.usage["input_tokens"] = self.usage.get("input_tokens", 0.0) + float(
            usage.prompt_tokens
        )
        self.usage["output_tokens"] = self.usage.get("output_tokens", 0.0) + float(
            usage.completion_tokens
        )

    def _serialized_messages(self, messages: Messages) -> list[dict[str, object]]:
        return [
            cast(dict[str, object], jsonable(message))
            for message in serialize_messages_for_output(messages)
        ]

    def turn_record(self, turn: Turn) -> dict[str, object]:
        return {
            "id": turn.id,
            "prompt": self._serialized_messages(turn.prompt),
            "completion": self._serialized_messages(turn.completion),
            "tool_calls": [
                cast(dict[str, object], jsonable(tool_call))
                for tool_call in turn.tool_calls
            ],
            "tool_results": self._serialized_messages(
                cast(Messages, turn.tool_results)
            ),
            "response_id": turn.response_id,
            "model": turn.model,
            "created": turn.created,
            "finish_reason": turn.finish_reason,
            "usage": turn.usage.model_dump(mode="json", exclude_none=True)
            if turn.usage is not None
            else None,
            "tokens": turn.tokens.model_dump(mode="json", exclude_none=True)
            if turn.tokens is not None
            else None,
            "reward": turn.reward,
            "is_truncated": turn.is_truncated,
            "timing": turn.timing.model_dump(mode="json", exclude_none=True),
        }
